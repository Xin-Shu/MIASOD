"""Video decoding for MIASOD.

One H.264 decode yields, per picture, the decoded pixels and the bit-cost (BC) tile scores. The BC path needs the
patched libavcodec built by script/bitcost/build.sh (libmia_bc.so). Without BC, PyAV can decode the pixels instead.

Supported input: H.264 (CABAC, I- and P-frames only, progressive, 8-bit 4:2:0) in a container such as MP4. Encode,
for example, with
    ffmpeg -framerate 10 -i frame/img%04d.png -c:v libx264 -preset medium -crf 22 -g 30 -bf 0 -pix_fmt yuv420p clip.mp4
"""
import ctypes as C
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from util import SCRIPT, STRIDE, TILE

PICTURE_TYPE = {1: "I", 2: "P"}


@dataclass
class DecodedFrame:
    index: int                  # 1-based position in decode order
    picture_type: str           # 'I' or 'P'
    width: int
    height: int
    yuv: np.ndarray             # (height * 3 / 2, width) uint8, YUV420p planes stacked
    bc: np.ndarray = None       # (N,) float32 BC tile scores, None without BC
    _bgr: np.ndarray = field(default=None, repr=False)

    @property
    def y(self):
        """Luma plane (height, width) uint8 (limited range, as decoded)."""
        return self.yuv[:self.height]

    @property
    def bgr(self):
        """BGR image for the detectors (converted once, on first use)."""
        if self._bgr is None:
            import cv2
            self._bgr = cv2.cvtColor(self.yuv, cv2.COLOR_YUV2BGR_I420)
        return self._bgr


class _Result(C.Structure):
    _fields_ = [(k, C.c_int) for k in ("width", "height", "index", "coded_id", "picture_type", "tiles", "mw", "mh")] + [
        ("pts", C.c_int64), ("yuv", C.POINTER(C.c_uint8)), ("grid", C.POINTER(C.c_uint32)),
        ("scores", C.POINTER(C.c_float)), ("yuv_size", C.c_size_t)]


def bc_library_path():
    env = os.environ.get("MIASOD_BC_LIB")
    return Path(env) if env else SCRIPT / "bitcost" / "build" / "libmia_bc.so"


def bc_available():
    return bc_library_path().is_file()


_LIB = None


def _library():
    global _LIB
    if _LIB is None:
        path = bc_library_path()
        if not path.is_file():
            raise RuntimeError(f"{path} not found: build it with `bash script/bitcost/build.sh` or set MIASOD_BC_LIB")
        lib = C.CDLL(str(path))
        lib.mia_result_size.restype = C.c_size_t
        if lib.mia_result_size() != C.sizeof(_Result):
            raise RuntimeError(f"{path}: result layout differs from decode.py; rebuild the library")
        lib.mia_open.argtypes = [C.c_char_p, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int]
        lib.mia_open.restype = C.c_void_p
        lib.mia_error.argtypes = [C.c_void_p]
        lib.mia_error.restype = C.c_char_p
        lib.mia_next.argtypes = [C.c_void_p, C.POINTER(_Result)]
        lib.mia_next.restype = C.c_int
        lib.mia_close.argtypes = [C.c_void_p]
        lib.mia_close.restype = None
        _LIB = lib
    return _LIB


class NativeReader:
    """Decoded pixels and BC tile scores from one decode (libmia_bc.so). Arrays it returns own their memory."""

    def __init__(self, path, bitcost=True, pixels=True, tile=TILE, stride=STRIDE, threads=4, grids=False):
        self.handle = None
        self.lib, self.grids = _library(), grids
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        self.handle = self.lib.mia_open(str(path).encode(), threads, int(pixels), int(bitcost), tile, stride)
        msg = self.lib.mia_error(self.handle) if self.handle else b"decoder allocation failed"
        if msg:
            self.close()
            raise ValueError(f"{path}: {msg.decode()}")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()

    def close(self):
        if self.handle:
            self.lib.mia_close(self.handle)
            self.handle = None

    def __iter__(self):
        return self

    def __next__(self):
        if not self.handle:
            raise StopIteration
        r = _Result()
        status = self.lib.mia_next(self.handle, C.byref(r))
        if status < 0:
            msg = self.lib.mia_error(self.handle).decode()
            self.close()
            raise ValueError(msg)
        if status == 0:
            self.close()
            raise StopIteration
        yuv = np.ctypeslib.as_array(r.yuv, shape=(r.yuv_size,)).reshape(r.height * 3 // 2, r.width).copy() if r.yuv else None
        bc = np.ctypeslib.as_array(r.scores, shape=(r.tiles,)).copy() if r.scores else None
        frame = DecodedFrame(r.index, PICTURE_TYPE[r.picture_type], r.width, r.height, yuv, bc)
        if self.grids and r.grid:
            frame.grid = np.ctypeslib.as_array(r.grid, shape=(r.mh * r.mw,)).reshape(r.mh, r.mw).copy()
        return frame


class PyAVReader:
    """Pixel-only decode through PyAV, for runs without the BC cue."""

    def __init__(self, path, threads=4):
        import av
        self.container = av.open(str(path))
        stream = self.container.streams.video[0]
        stream.thread_count = threads
        self.frames, self.index = self.container.decode(stream), 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self.container is not None:
            self.container.close()
            self.container = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.container is None:
            raise StopIteration
        try:
            f = next(self.frames)
        except StopIteration:
            self.close()
            raise
        if f.format.name != "yuv420p":
            raise ValueError(f"expected yuv420p, got {f.format.name}")
        self.index += 1
        ptype = {"I": "I", "P": "P"}.get(str(f.pict_type).split(".")[-1], str(f.pict_type))
        return DecodedFrame(self.index, ptype, f.width, f.height, f.to_ndarray(format="yuv420p"))


def open_video(path, bitcost=True, tile=TILE, stride=STRIDE, threads=4):
    """Reader over the frames of `path`. BC needs libmia_bc.so; without BC, PyAV is the fallback decoder."""
    if bitcost or bc_available():
        return NativeReader(path, bitcost=bitcost, tile=tile, stride=stride, threads=threads)
    try:
        return PyAVReader(path, threads=threads)
    except ImportError as e:
        raise RuntimeError("no decoder: build script/bitcost/build.sh or install PyAV (pip install av)") from e


def count_frames(path, threads=4):
    """Number of decodable frames (one pixel-free decode pass when the BC library is available)."""
    if bc_available():
        with NativeReader(path, bitcost=False, pixels=False, threads=threads) as r:
            return sum(1 for _ in r)
    with PyAVReader(path, threads=threads) as r:
        return sum(1 for _ in r)
