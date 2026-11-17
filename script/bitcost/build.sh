#!/usr/bin/env bash
# Build the bit-cost decoder library (libmia_bc.so).
#
#   1. fetch FFmpeg at the pinned commit (or use --ffmpeg-src DIR, a clean checkout of that commit),
#   2. apply h264_slice.patch and add mia_bc_store.h to libavcodec,
#   3. build a minimal static libavcodec / libavformat / libavutil (H.264 decoder, MP4 and MKV demuxers),
#   4. link mia_bc_decoder.c and those libraries into one self-contained <out>/libmia_bc.so.
#
# Usage: bash script/bitcost/build.sh [--out DIR] [--ffmpeg-src DIR] [--jobs N]
# Default output: script/bitcost/build. script/decode.py finds the library there, or via MIASOD_BC_LIB.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FFMPEG_URL="https://github.com/FFmpeg/FFmpeg.git"
FFMPEG_COMMIT="ea5b375e0e95b05c88ec728d5953c916b4925bef"
OUT="$HERE/build"
SRC=""
JOBS="$(( $(nproc) < 8 ? $(nproc) : 8 ))"

while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --ffmpeg-src) SRC="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
FF="$OUT/ffmpeg"

if [ -n "$SRC" ]; then
  [ -d "$SRC/libavcodec" ] || { echo "--ffmpeg-src: $SRC is not an FFmpeg tree" >&2; exit 1; }
  rm -rf "$FF"; mkdir -p "$FF"
  if [ -d "$SRC/.git" ]; then
    git -C "$SRC" archive "$FFMPEG_COMMIT" | tar -x -C "$FF"
  else
    cp -a "$SRC/." "$FF/"
  fi
elif [ ! -f "$FF/.mia_patched" ]; then
  rm -rf "$FF"; mkdir -p "$FF"
  git -C "$FF" init -q
  git -C "$FF" fetch -q --depth 1 "$FFMPEG_URL" "$FFMPEG_COMMIT"
  git -C "$FF" -c advice.detachedHead=false checkout -q FETCH_HEAD
fi

if [ ! -f "$FF/.mia_patched" ]; then
  patch -d "$FF" -p1 --forward < "$HERE/h264_slice.patch"
  cp "$HERE/mia_bc_store.h" "$FF/libavcodec/mia_bc_store.h"
  touch "$FF/.mia_patched"
fi

ASM="--enable-x86asm"
command -v nasm >/dev/null 2>&1 || ASM="--disable-x86asm"

if [ ! -f "$FF/ffbuild/config.mak" ]; then
  (cd "$FF" && ./configure \
      --disable-everything --disable-autodetect --disable-programs --disable-doc --disable-network \
      --disable-avdevice --disable-avfilter --disable-swscale --disable-swresample --disable-postproc \
      --disable-shared --enable-static --enable-pic --enable-pthreads $ASM \
      --enable-decoder=h264 --enable-parser=h264 --enable-demuxer=mov,matroska --enable-protocol=file)
fi
make -C "$FF" -j"$JOBS" libavutil/libavutil.a libavcodec/libavcodec.a libavformat/libavformat.a

# FFmpeg is linked statically and its symbols stay hidden, so the library does not clash with any other FFmpeg
# copy loaded in the same process (e.g. by OpenCV or PyAV).
gcc -O3 -Wall -Wextra -shared -fPIC -I"$FF" "$HERE/mia_bc_decoder.c" \
    "$FF/libavformat/libavformat.a" "$FF/libavcodec/libavcodec.a" "$FF/libavutil/libavutil.a" \
    -Wl,--exclude-libs,ALL -Wl,-Bsymbolic -lm -lpthread -o "$OUT/libmia_bc.so"

echo "built $OUT/libmia_bc.so"
