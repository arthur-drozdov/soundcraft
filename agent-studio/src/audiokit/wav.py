"""WAV (RIFF/WAVE) reading and writing without third-party dependencies.

Audio is ``float32`` shaped ``(channels, frames)``. Files are treated as hostile: chunk
walking is bounded, sizes are checked against the file and against caps before any
allocation, and non-finite float samples are replaced with zero.
"""

from __future__ import annotations

import os
import struct
import tempfile
from pathlib import Path
from typing import BinaryIO

import numpy as np

MAX_FRAMES = 48_000 * 900  # 15 minutes at 48 kHz
MAX_CHANNELS = 64
MIN_SAMPLE_RATE = 1
MAX_SAMPLE_RATE = 768_000
_MAX_CHUNKS = 1000
_MAX_FMT_BYTES = 1024
_MAX_DATA_BYTES = 0xFFFF_FFFF - 1024

_TAG_PCM = 0x0001
_TAG_FLOAT = 0x0003
_TAG_EXTENSIBLE = 0xFFFE
_CHANNEL_MASKS = {3: 0x7, 4: 0x33, 5: 0x607, 6: 0x3F, 7: 0x70F, 8: 0x63F}
_PCM_GUID_TAIL = b"\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"


class WavError(ValueError):
    """Raised for malformed, unsupported or over-limit WAV data."""


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> None:
    """Write ``data`` to ``path`` via a temp file in the same folder and an atomic rename."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=".tmp-", suffix=target.suffix)
    renamed = False
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
        renamed = True
    finally:
        if not renamed:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass


def _read_exact(handle: BinaryIO, n: int, what: str) -> bytes:
    data = handle.read(n)
    if len(data) != n:
        raise WavError(f"truncated WAV file while reading {what}")
    return data


def _parse_fmt(raw: bytes) -> tuple[int, int, int, int]:
    """Return ``(format_tag, channels, sample_rate, bits)`` from a ``fmt `` chunk body."""
    if len(raw) < 16:
        raise WavError("fmt chunk too short")
    tag, channels, rate, _byte_rate, _align, bits = struct.unpack("<HHIIHH", raw[:16])
    if tag == _TAG_EXTENSIBLE:
        if len(raw) < 40:
            raise WavError("WAVE_FORMAT_EXTENSIBLE fmt chunk too short")
        tag = struct.unpack("<H", raw[24:26])[0]
    if tag not in (_TAG_PCM, _TAG_FLOAT):
        raise WavError(f"unsupported WAV format tag 0x{tag:04x}")
    if not MIN_SAMPLE_RATE <= rate <= MAX_SAMPLE_RATE:
        raise WavError(f"unsupported sample rate {rate}")
    if not 1 <= channels <= MAX_CHANNELS:
        raise WavError(f"unsupported channel count {channels}")
    ok = (tag == _TAG_PCM and bits in (8, 16, 24, 32)) or (tag == _TAG_FLOAT and bits in (32, 64))
    if not ok:
        raise WavError(f"unsupported sample format: tag {tag}, {bits} bits")
    return tag, channels, rate, bits


def _decode(raw: np.ndarray, tag: int, bits: int, channels: int) -> np.ndarray:
    """Decode interleaved little-endian bytes into float32 ``(channels, frames)``."""
    if tag == _TAG_FLOAT:
        samples = raw.view("<f4" if bits == 32 else "<f8").astype(np.float32)
        samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    elif bits == 8:
        samples = (raw.astype(np.float32) - 128.0) / 128.0
    elif bits == 16:
        samples = raw.view("<i2").astype(np.float32) / 32768.0
    elif bits == 24:
        b = raw.reshape(-1, 3).astype(np.int32)
        value = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        samples = ((value ^ 0x800000) - 0x800000).astype(np.float32) / 8388608.0
    else:
        samples = raw.view("<i4").astype(np.float32) / 2147483648.0
    return np.ascontiguousarray(samples.reshape(-1, channels).T)


def read(path: str | os.PathLike[str], max_frames: int = MAX_FRAMES) -> tuple[np.ndarray, int]:
    """Read a WAV file into ``(float32 audio[channels, frames], sample_rate)``.

    Supports 8/16/24/32-bit PCM and 32/64-bit float, mono or multichannel, including
    ``WAVE_FORMAT_EXTENSIBLE``. Files with more than ``max_frames`` frames are rejected.
    """
    with open(path, "rb") as handle:
        file_size = os.fstat(handle.fileno()).st_size
        riff = _read_exact(handle, 12, "RIFF header")
        if riff[:4] != b"RIFF" or riff[8:12] != b"WAVE":
            raise WavError("not a RIFF/WAVE file")
        fmt: tuple[int, int, int, int] | None = None
        for _ in range(_MAX_CHUNKS):
            header = handle.read(8)
            if len(header) < 8:
                break
            chunk_id, size = struct.unpack("<4sI", header)
            body_start = handle.tell()
            if chunk_id == b"fmt ":
                if size > _MAX_FMT_BYTES:
                    raise WavError("fmt chunk too large")
                fmt = _parse_fmt(_read_exact(handle, size, "fmt chunk"))
            elif chunk_id == b"data":
                if fmt is None:
                    raise WavError("data chunk before fmt chunk")
                tag, channels, rate, bits = fmt
                size = min(size, file_size - body_start)
                frame_bytes = channels * bits // 8
                frames = size // frame_bytes
                if frames > max_frames or frames * channels > max_frames * 8:
                    raise WavError(f"WAV too long: {frames} frames (cap {max_frames})")
                raw = np.fromfile(handle, dtype=np.uint8, count=frames * frame_bytes)
                if raw.size != frames * frame_bytes:
                    raise WavError("truncated WAV data")
                return _decode(raw, tag, bits, channels), rate
            handle.seek(body_start + size + (size & 1))
    raise WavError("no data chunk found" if fmt else "no fmt chunk found")


def _encode(audio: np.ndarray, bit_depth: str) -> bytes:
    interleaved = np.ascontiguousarray(audio.T)
    if bit_depth == "16":
        return np.clip(np.rint(interleaved * 32768.0), -32768, 32767).astype("<i2").tobytes()
    if bit_depth == "24":
        ints = np.clip(np.rint(interleaved.astype(np.float64) * 8388608.0), -8388608, 8388607)
        return ints.astype("<i4").view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
    return interleaved.astype("<f4").tobytes()


def write(path: str | os.PathLike[str], audio: np.ndarray, sr: int, bit_depth: str = "24") -> None:
    """Write ``audio`` as a WAV file (``bit_depth`` is ``"16"``, ``"24"`` or ``"32f"``), atomically.

    Integer depths clip to full scale; non-finite samples are written as zero. More than two
    channels use ``WAVE_FORMAT_EXTENSIBLE``.
    """
    if bit_depth not in ("16", "24", "32f"):
        raise WavError(f"unsupported bit depth {bit_depth!r}")
    if isinstance(sr, bool) or not isinstance(sr, (int, np.integer)):
        raise WavError("sample rate must be an integer")
    if not MIN_SAMPLE_RATE <= sr <= MAX_SAMPLE_RATE:
        raise WavError(f"unsupported sample rate {sr}")
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.ndim != 2 or not 1 <= arr.shape[0] <= MAX_CHANNELS:
        raise WavError("audio must be shaped (channels, frames)")
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    channels = arr.shape[0]
    data = _encode(arr, bit_depth)
    if len(data) > _MAX_DATA_BYTES:
        raise WavError("audio too large for a WAV file")
    bits = {"16": 16, "24": 24, "32f": 32}[bit_depth]
    is_float = bit_depth == "32f"
    block = channels * bits // 8
    rate = int(sr)
    common = struct.pack("<HIIHH", channels, rate, rate * block, block, bits)
    chunks = b""
    if channels > 2:
        guid = struct.pack("<H", _TAG_FLOAT if is_float else _TAG_PCM) + _PCM_GUID_TAIL
        ext = struct.pack("<HHI", 22, bits, _CHANNEL_MASKS.get(channels, 0)) + guid
        fmt_body = struct.pack("<H", _TAG_EXTENSIBLE) + common + ext
    elif is_float:
        fmt_body = struct.pack("<H", _TAG_FLOAT) + common + struct.pack("<H", 0)
        chunks += b"fact" + struct.pack("<II", 4, len(data) // block)
    else:
        fmt_body = struct.pack("<H", _TAG_PCM) + common
    fmt_chunk = b"fmt " + struct.pack("<I", len(fmt_body)) + fmt_body
    pad = b"\x00" if len(data) & 1 else b""
    body = b"WAVE" + fmt_chunk + chunks + b"data" + struct.pack("<I", len(data)) + data + pad
    atomic_write_bytes(path, b"RIFF" + struct.pack("<I", len(body)) + body)
