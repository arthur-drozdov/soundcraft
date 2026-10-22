from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from ak_signals import SR, sine
from audiokit import wav


def _riff(fmt_body: bytes, data: bytes, extra: bytes = b"") -> bytes:
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt_body)) + fmt_body + extra
    body += b"data" + struct.pack("<I", len(data)) + data + (b"\x00" if len(data) & 1 else b"")
    return b"RIFF" + struct.pack("<I", len(body)) + body


def _fmt(tag: int, ch: int, rate: int, bits: int) -> bytes:
    block = ch * bits // 8
    return struct.pack("<HHIIHH", tag, ch, rate, rate * block, block, bits)


@pytest.mark.parametrize("depth,tol", [("16", 1 / 32768), ("24", 1 / 8388608 * 1.01), ("32f", 0.0)])
def test_round_trip_depths(tmp_path: Path, depth: str, tol: float) -> None:
    rng = np.random.default_rng(3)
    audio = (0.9 * rng.uniform(-1, 1, size=(2, 5000))).astype(np.float32)
    path = tmp_path / "a.wav"
    wav.write(path, audio, 48000, depth)
    back, sr = wav.read(path)
    assert sr == 48000 and back.shape == audio.shape and back.dtype == np.float32
    assert np.max(np.abs(back - audio)) <= tol + 1e-9


def test_mono_and_multichannel_extensible(tmp_path: Path) -> None:
    for channels in (1, 3, 6):
        audio = np.stack([sine(100 * (i + 1), 0.05, amp=0.5)[0] for i in range(channels)])
        for depth in ("16", "24", "32f"):
            path = tmp_path / f"c{channels}{depth}.wav"
            wav.write(path, audio, SR, depth)
            back, sr = wav.read(path)
            assert back.shape == audio.shape and sr == SR
            assert np.max(np.abs(back - audio)) < 1e-4
    raw = (tmp_path / "c6" "24.wav").read_bytes()
    assert struct.unpack("<H", raw[20:22])[0] == 0xFFFE


def test_clipping_and_nonfinite(tmp_path: Path) -> None:
    audio = np.array([[2.0, -2.0, np.nan, np.inf, 0.25]], dtype=np.float32)
    path = tmp_path / "x.wav"
    wav.write(path, audio, 8000, "16")
    back, _ = wav.read(path)
    assert back[0, 0] == pytest.approx(32767 / 32768) and back[0, 1] == -1.0
    assert back[0, 2] == 0.0 and back[0, 3] == 0.0 and back[0, 4] == pytest.approx(0.25)


def test_read_8bit_and_32bit_int_and_64bit_float(tmp_path: Path) -> None:
    p8 = tmp_path / "8.wav"
    p8.write_bytes(_riff(_fmt(1, 1, 8000, 8), bytes([0, 128, 255, 192, 64])))
    a8, sr = wav.read(p8)
    assert sr == 8000
    assert a8[0] == pytest.approx([-1.0, 0.0, 127 / 128, 0.5, -0.5])

    values = np.array([0, 2**30, -(2**30), 2**31 - 1, -(2**31)], dtype="<i4")
    p32 = tmp_path / "32.wav"
    p32.write_bytes(_riff(_fmt(1, 1, 48000, 32), values.tobytes()))
    a32, _ = wav.read(p32)
    assert a32[0, :3] == pytest.approx([0.0, 0.5, -0.5]) and a32[0, 4] == -1.0

    f64 = np.array([0.5, -0.25, 1.5, np.nan], dtype="<f8")
    p64 = tmp_path / "64.wav"
    p64.write_bytes(_riff(_fmt(3, 1, 48000, 64), f64.tobytes()))
    a64, _ = wav.read(p64)
    assert a64[0] == pytest.approx([0.5, -0.25, 1.5, 0.0])


def test_read_extensible_24bit_stereo(tmp_path: Path) -> None:
    guid = struct.pack("<H", 1) + b"\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"
    fmt = struct.pack("<HHIIHH", 0xFFFE, 2, 44100, 44100 * 6, 6, 24) + struct.pack("<HHI", 22, 24, 3) + guid
    samples = [0x400000, -0x400000, 0x7FFFFF, -0x800000]
    data = b"".join(int(v & 0xFFFFFF).to_bytes(3, "little") for v in samples)
    path = tmp_path / "ext.wav"
    path.write_bytes(_riff(fmt, data))
    audio, sr = wav.read(path)
    assert sr == 44100 and audio.shape == (2, 2)
    assert audio[0] == pytest.approx([0.5, 0x7FFFFF / 8388608]) and audio[1] == pytest.approx([-0.5, -1.0])


def test_unknown_chunks_and_odd_padding_are_skipped(tmp_path: Path) -> None:
    junk = b"LIST" + struct.pack("<I", 3) + b"abc\x00"
    path = tmp_path / "j.wav"
    path.write_bytes(_riff(_fmt(1, 1, 8000, 16), np.array([1000, -1000], "<i2").tobytes(), extra=junk))
    audio, _ = wav.read(path)
    assert audio[0] == pytest.approx([1000 / 32768, -1000 / 32768])


def test_hostile_files_are_rejected(tmp_path: Path) -> None:
    def read_bytes(name: str, blob: bytes, **kw: int) -> None:
        p = tmp_path / name
        p.write_bytes(blob)
        wav.read(p, **kw)

    good = _riff(_fmt(1, 1, 8000, 16), np.zeros(100, "<i2").tobytes())
    with pytest.raises(wav.WavError):
        read_bytes("nothing.wav", b"hello")
    with pytest.raises(wav.WavError):
        read_bytes("short.wav", good[:20])
    with pytest.raises(wav.WavError):
        read_bytes("cap.wav", good, max_frames=50)
    with pytest.raises(wav.WavError):
        read_bytes("adpcm.wav", _riff(_fmt(2, 1, 8000, 4), b"\x00" * 8))
    with pytest.raises(wav.WavError):
        read_bytes("zero_ch.wav", _riff(_fmt(1, 0, 8000, 16), b"\x00" * 8))
    with pytest.raises(wav.WavError):
        read_bytes("no_data.wav", b"RIFF" + struct.pack("<I", 28) + b"WAVEfmt " + struct.pack("<I", 16) + _fmt(1, 1, 8000, 16))
    # data size larger than the file is clamped to what is there
    lying = bytearray(good)
    lying[40:44] = struct.pack("<I", 0xFFFFFFF0)
    p = tmp_path / "lying.wav"
    p.write_bytes(bytes(lying))
    audio, _ = wav.read(p)
    assert audio.shape == (1, 100)


def test_write_validation_and_atomicity(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "o.wav"
    wav.write(path, sine(440, 0.01), SR, "16")
    assert path.exists()
    with pytest.raises(wav.WavError):
        wav.write(path, sine(440, 0.01), SR, "8")
    with pytest.raises(wav.WavError):
        wav.write(path, sine(440, 0.01), 0, "16")
    with pytest.raises(wav.WavError):
        wav.write(path, np.zeros((2, 2, 2), np.float32), SR, "16")
    assert [p.name for p in path.parent.iterdir()] == ["o.wav"]  # no temp files left behind
