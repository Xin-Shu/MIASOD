/*
 * One H.264 decode that returns, per picture, the decoded YUV420p pixels and the bit-cost (BC) tile scores.
 *
 * Links against the libavcodec built with h264_slice.patch. The BC score of a tile is the largest
 * macroblock bit cost inside it; tile origins follow the scoring grid of MIASOD (size T, stride S,
 * the last tile of each row and column shifted to end at the frame border).
 */
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libavutil/mem.h>

extern void *avpriv_mia_bc_create(void);
extern void avpriv_mia_bc_free(void *);
extern int avpriv_mia_bc_take(void *, int, int, int, int, uint32_t **);
extern int avpriv_mia_bc_finish(void *, int);
extern int avpriv_mia_bc_error(void *, char *, int);

typedef struct MiaResult {
    int width, height, index, coded_id, picture_type, tiles, mw, mh;
    int64_t pts;
    const uint8_t *yuv;
    const uint32_t *grid;
    const float *scores;
    size_t yuv_size;
} MiaResult;

typedef struct MiaDecoder {
    AVFormatContext *fmt;
    AVCodecContext *dec;
    AVPacket *pkt;
    AVFrame *frame;
    void *store;
    uint8_t *packed;
    uint32_t *grid;
    float *scores;
    int *bounds;
    int stream, width, height, tiles, pixels, frames, flushed, finished;
    int64_t last_pts;
    char error[256];
    MiaResult result;
} MiaDecoder;

size_t mia_result_size(void) { return sizeof(MiaResult); }

static int fail(MiaDecoder *d, const char *msg)
{
    if (d && !d->error[0]) snprintf(d->error, sizeof(d->error), "%s", msg);
    return -1;
}

static int store_bad(MiaDecoder *d)
{
    if (!d->store) return 0;
    return avpriv_mia_bc_error(d->store, d->error, sizeof(d->error));
}

const char *mia_error(void *opaque)
{
    MiaDecoder *d = opaque;
    if (!d) return "decoder allocation failed";
    store_bad(d);
    return d->error;
}

/* Origins along one axis: 0, S, 2S, ... and L - T as the last, as tile_origins() in util.py. */
static int axis(int L, int T, int S, int *out)
{
    int n = 0, x;
    if (L <= T) { if (out) out[0] = 0; return 1; }
    for (x = 0; x <= L - T; x += S) { if (out) out[n] = x; n++; }
    if ((n - 1) * S != L - T) { if (out) out[n] = L - T; n++; }
    return n;
}

void *mia_open(const char *path, int threads, int pixels, int bc, int tile, int stride)
{
    MiaDecoder *d = av_mallocz(sizeof(*d)); const AVCodec *codec; unsigned i; int x, y, nx, ny, j;
    int *xs = NULL, *ys = NULL;
    if (!d) return NULL;
    d->stream = -1; d->pixels = pixels; d->last_pts = AV_NOPTS_VALUE;
    if (threads < 1 || threads > 32) { fail(d, "threads must be in [1, 32]"); return d; }
    if (tile < 16 || stride < 16 || tile % 16 || stride % 16 || stride > tile) {
        fail(d, "tile and stride must be multiples of 16 with stride <= tile"); return d;
    }
    if (avformat_open_input(&d->fmt, path, NULL, NULL) < 0) { fail(d, "cannot open input"); return d; }
    for (i = 0; i < d->fmt->nb_streams; i++) if (d->fmt->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
        if (d->stream != -1) { fail(d, "multiple video streams unsupported"); return d; }
        d->stream = i;
    }
    if (d->stream < 0) { fail(d, "missing video stream"); return d; }
    AVCodecParameters *p = d->fmt->streams[d->stream]->codecpar;
    if (p->codec_id != AV_CODEC_ID_H264 || p->video_delay || !p->extradata_size || p->width < tile || p->height < tile ||
        p->width > 16384 || p->height > 16384 || p->width % 2 || p->height % 2) {
        fail(d, "requires H.264 without B-frames in a container (e.g. MP4), even dimensions, at least one tile per side");
        return d;
    }
    d->width = p->width; d->height = p->height;
    codec = avcodec_find_decoder(AV_CODEC_ID_H264);
    d->dec = avcodec_alloc_context3(codec);
    if (!d->dec || avcodec_parameters_to_context(d->dec, p) < 0) { fail(d, "codec allocation"); return d; }
    if (bc) {
        d->store = avpriv_mia_bc_create();
        if (!d->store) { fail(d, "BC store allocation"); return d; }
        d->dec->opaque = d->store;
    }
    d->dec->thread_count = threads; d->dec->err_recognition = AV_EF_CAREFUL | AV_EF_EXPLODE;
    if (avcodec_open2(d->dec, codec, NULL) < 0) { fail(d, "open decoder"); return d; }
    d->pkt = av_packet_alloc(); d->frame = av_frame_alloc();
    if (!d->pkt || !d->frame) { fail(d, "frame/packet allocation"); return d; }
    if (pixels) {
        d->packed = av_malloc((size_t)d->width * d->height * 3 / 2);
        if (!d->packed) { fail(d, "pixel buffer allocation"); return d; }
    }
    nx = axis(d->width, tile, stride, NULL); ny = axis(d->height, tile, stride, NULL);
    d->tiles = nx * ny;
    xs = av_malloc_array(nx, sizeof(int)); ys = av_malloc_array(ny, sizeof(int));
    d->bounds = av_malloc_array(d->tiles, 4 * sizeof(int)); d->scores = av_malloc_array(d->tiles, sizeof(float));
    if (!xs || !ys || !d->bounds || !d->scores) { av_free(xs); av_free(ys); fail(d, "tile allocation"); return d; }
    axis(d->width, tile, stride, xs); axis(d->height, tile, stride, ys);
    /* Macroblock window of each tile, row-major tile order. A tile origin that is not a multiple of 16
     * (only the last row or column of frames whose size is not a multiple of 16) is floored. */
    for (y = 0, j = 0; y < ny; y++) for (x = 0; x < nx; x++, j++) {
        int *b = d->bounds + 4 * j;
        b[0] = xs[x] / 16; b[1] = ys[y] / 16; b[2] = b[0] + tile / 16; b[3] = b[1] + tile / 16;
    }
    av_free(xs); av_free(ys);
    return d;
}

static int output(MiaDecoder *d)
{
    AVFrame *f = d->frame; MiaResult *r = &d->result; int i, mw = (d->width + 15) / 16, mh = (d->height + 15) / 16;
    if (f->width != d->width || f->height != d->height || f->format != AV_PIX_FMT_YUV420P || f->color_range == AVCOL_RANGE_JPEG ||
        f->interlaced_frame || f->decode_error_flags || (f->flags & AV_FRAME_FLAG_CORRUPT) ||
        (f->pict_type != AV_PICTURE_TYPE_I && f->pict_type != AV_PICTURE_TYPE_P))
        return fail(d, "unsupported pixel format or range, changed dimensions, or a corrupt or non-I/P picture");
    if (f->pts == AV_NOPTS_VALUE || (d->frames && f->pts <= d->last_pts)) return fail(d, "missing or non-monotonic timestamp");
    if (f->coded_picture_number != d->frames) return fail(d, "missing or reordered decoded picture");
    if (d->store) {
        if (avpriv_mia_bc_take(d->store, f->coded_picture_number, f->pict_type, mw, mh, &d->grid) < 0) {
            store_bad(d); return fail(d, "incomplete BC picture");
        }
        for (i = 0; i < d->tiles; i++) {
            int *b = d->bounds + 4 * i, x, y; uint32_t best = 0;
            for (y = b[1]; y < b[3]; y++) for (x = b[0]; x < b[2]; x++) {
                uint32_t v = d->grid[(y < mh ? y : mh - 1) * mw + (x < mw ? x : mw - 1)];
                if (v > best) best = v;
            }
            d->scores[i] = (float)best;
        }
    }
    if (d->pixels && av_image_copy_to_buffer(d->packed, (size_t)d->width * d->height * 3 / 2,
            (const uint8_t * const *)f->data, f->linesize, f->format, f->width, f->height, 1) < 0)
        return fail(d, "pixel packing failed");
    d->frames++; d->last_pts = f->pts;
    *r = (MiaResult){d->width, d->height, d->frames, f->coded_picture_number, f->pict_type, d->tiles, mw, mh,
                     f->pts, d->packed, d->grid, d->store ? d->scores : NULL,
                     d->pixels ? (size_t)d->width * d->height * 3 / 2 : 0};
    return 1;
}

/* 1: one picture in *result (valid until the next call), 0: end of stream, -1: error (see mia_error). */
int mia_next(void *opaque, MiaResult *result)
{
    MiaDecoder *d = opaque; int r;
    if (!d || d->error[0]) return -1;
    if (d->finished) return 0;
    av_freep(&d->grid); av_frame_unref(d->frame);
    for (;;) {
        r = avcodec_receive_frame(d->dec, d->frame);
        if (store_bad(d)) return -1;
        if (r >= 0) { r = output(d); if (r > 0) *result = d->result; return r; }
        if (r == AVERROR_EOF) {
            if (!d->frames || (d->store && avpriv_mia_bc_finish(d->store, d->frames) < 0)) {
                store_bad(d); return fail(d, "incomplete stream");
            }
            d->finished = 1; return 0;
        }
        if (r != AVERROR(EAGAIN)) return fail(d, "receive/decode failed");
        if (d->flushed) return fail(d, "unexpected decoder state after flush");
        do {
            r = av_read_frame(d->fmt, d->pkt);
            if (r < 0) break;
            if (d->pkt->stream_index == d->stream) break;
            av_packet_unref(d->pkt);
        } while (1);
        if (r == AVERROR_EOF) {
            d->flushed = 1; r = avcodec_send_packet(d->dec, NULL);
        } else if (r < 0) return fail(d, "demux failed");
        else {
            if (d->pkt->flags & AV_PKT_FLAG_CORRUPT) { av_packet_unref(d->pkt); return fail(d, "corrupt packet"); }
            r = avcodec_send_packet(d->dec, d->pkt); av_packet_unref(d->pkt);
        }
        if (store_bad(d)) return -1;
        if (r < 0) return fail(d, "send packet failed");
    }
}

void mia_close(void *opaque)
{
    MiaDecoder *d = opaque;
    if (!d) return;
    /* Join the codec threads before freeing the store they write into. */
    avcodec_free_context(&d->dec); avpriv_mia_bc_free(d->store);
    av_frame_free(&d->frame); av_packet_free(&d->pkt); avformat_close_input(&d->fmt);
    av_free(d->packed); av_free(d->grid); av_free(d->scores); av_free(d->bounds); av_free(d);
}
