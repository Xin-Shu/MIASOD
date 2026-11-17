/*
 * Per-decoder store of macroblock bit costs, compiled into libavcodec by h264_slice.patch.
 *
 * decode_slice() commits the bit cost of every macroblock of a slice; the reader takes the complete
 * grid of a picture when that picture leaves the decoder. One store belongs to one AVCodecContext
 * (passed through avctx->opaque), so frame-threaded decoding and several open clips are safe.
 */
#ifndef MIA_BC_STORE_H
#define MIA_BC_STORE_H

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "libavutil/mem.h"
#include "libavutil/error.h"

#define MIA_BC_PENDING 64

typedef struct MiaBCPicture {
    int occupied, id, type, mw, mh, count;
    uint32_t *grid;
    uint8_t *seen;
} MiaBCPicture;

typedef struct MiaBCStore {
    pthread_mutex_t mutex;
    char error[160];
    int last_id, returned;
    MiaBCPicture pic[MIA_BC_PENDING];
} MiaBCStore;

void *avpriv_mia_bc_create(void);
void avpriv_mia_bc_free(void *opaque);
int avpriv_mia_bc_commit(void *opaque, int id, int type, int mw, int mh, int first, int n, const uint32_t *bits);
int avpriv_mia_bc_take(void *opaque, int id, int type, int mw, int mh, uint32_t **grid);
int avpriv_mia_bc_finish(void *opaque, int frames);
int avpriv_mia_bc_error(void *opaque, char *message, int capacity);

static int mia_bc_fail(MiaBCStore *s, const char *message)
{
    if (s) {
        pthread_mutex_lock(&s->mutex);
        if (!s->error[0]) snprintf(s->error, sizeof(s->error), "%s", message);
        pthread_mutex_unlock(&s->mutex);
    }
    return AVERROR_INVALIDDATA;
}

void *avpriv_mia_bc_create(void)
{
    MiaBCStore *s = av_mallocz(sizeof(*s));
    if (!s) return NULL;
    if (pthread_mutex_init(&s->mutex, NULL)) { av_free(s); return NULL; }
    s->last_id = -1;
    return s;
}

void avpriv_mia_bc_free(void *opaque)
{
    MiaBCStore *s = opaque; int i;
    if (!s) return;
    for (i = 0; i < MIA_BC_PENDING; i++) { av_free(s->pic[i].grid); av_free(s->pic[i].seen); }
    pthread_mutex_destroy(&s->mutex);
    av_free(s);
}

int avpriv_mia_bc_error(void *opaque, char *message, int capacity)
{
    MiaBCStore *s = opaque; int bad;
    if (!s) return 0;
    pthread_mutex_lock(&s->mutex);
    bad = !!s->error[0];
    /* Never overwrite a decoder or demuxer error that the caller already holds. */
    if (bad && message && capacity > 0) snprintf(message, capacity, "%s", s->error);
    pthread_mutex_unlock(&s->mutex);
    return bad;
}

int avpriv_mia_bc_commit(void *opaque, int id, int type, int mw, int mh, int first, int n, const uint32_t *bits)
{
    MiaBCStore *s = opaque; MiaBCPicture *p; const char *why = NULL; int cells;
    if (!s || id < 0 || mw < 1 || mh < 1 || mw > 4096 || mh > 4096 || (type != 1 && type != 2))
        return mia_bc_fail(s, "invalid picture metadata");
    cells = mw * mh;
    if (!bits || first < 0 || n <= 0 || first > cells - n) return mia_bc_fail(s, "invalid slice range");
    pthread_mutex_lock(&s->mutex);
    p = &s->pic[id % MIA_BC_PENDING];
    if (s->error[0]) { pthread_mutex_unlock(&s->mutex); return AVERROR_INVALIDDATA; }
    if (id <= s->last_id) why = "late picture after output commit";
    else if (p->occupied && p->id != id) why = "too many pending pictures";
    else {
        if (!p->occupied) {
            p->grid = av_malloc_array(cells, sizeof(*p->grid)); p->seen = av_mallocz(cells);
            if (!p->grid || !p->seen) {
                av_freep(&p->grid); av_freep(&p->seen); why = "picture allocation failed";
            } else { p->occupied = 1; p->id = id; p->type = type; p->mw = mw; p->mh = mh; p->count = 0; }
        }
        if (!why) {
            if (p->mw != mw || p->mh != mh || p->type != type) why = "inconsistent slices in picture";
            else if (memchr(p->seen + first, 1, n)) why = "duplicate macroblock coverage";
            else {
                memcpy(p->grid + first, bits, (size_t)n * sizeof(*bits));
                memset(p->seen + first, 1, n); p->count += n;
            }
        }
    }
    if (why && !s->error[0]) snprintf(s->error, sizeof(s->error), "%s", why);
    pthread_mutex_unlock(&s->mutex);
    return why ? AVERROR_INVALIDDATA : 0;
}

int avpriv_mia_bc_take(void *opaque, int id, int type, int mw, int mh, uint32_t **grid)
{
    MiaBCStore *s = opaque; MiaBCPicture *p; const char *why = NULL;
    if (!s || !grid || id < 0) return mia_bc_fail(s, "invalid output request");
    *grid = NULL;
    pthread_mutex_lock(&s->mutex);
    p = &s->pic[id % MIA_BC_PENDING];
    if (s->error[0]) { pthread_mutex_unlock(&s->mutex); return AVERROR_INVALIDDATA; }
    if (id != s->last_id + 1) why = "missing or reordered output picture";
    else if (!p->occupied || p->id != id) why = "output picture has no macroblock record";
    else if (p->mw != mw || p->mh != mh || p->type != type) why = "output metadata disagrees with slice";
    else if (p->count != mw * mh) why = "incomplete macroblock coverage";
    else {
        *grid = p->grid; p->grid = NULL; av_freep(&p->seen); p->occupied = 0;
        s->last_id = id; s->returned++;
    }
    if (why && !s->error[0]) snprintf(s->error, sizeof(s->error), "%s", why);
    pthread_mutex_unlock(&s->mutex);
    return why ? AVERROR_INVALIDDATA : 0;
}

int avpriv_mia_bc_finish(void *opaque, int frames)
{
    MiaBCStore *s = opaque; int i, bad = 0;
    if (!s) return 0;
    pthread_mutex_lock(&s->mutex);
    if (s->returned != frames || frames <= 0) bad = 1;
    for (i = 0; i < MIA_BC_PENDING; i++) if (s->pic[i].occupied) bad = 1;
    if (bad && !s->error[0]) snprintf(s->error, sizeof(s->error), "uncommitted or missing pictures at end");
    bad = bad || !!s->error[0];
    pthread_mutex_unlock(&s->mutex);
    return bad ? AVERROR_INVALIDDATA : 0;
}

#endif /* MIA_BC_STORE_H */
