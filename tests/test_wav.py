"""Tests for WAV header normalization.

Shapes used here are modelled on real DashScope replies captured during live
testing (see ``core/wav.py`` for the measured numbers).

@author zexuan.peng <pengzexuan2001@gmail.com>
"""

from __future__ import annotations

import struct
import wave
from io import BytesIO

import pytest

from core.audio_store import AudioStore, detect_audio_suffix
from core.wav import normalize_wav_sizes, wav_duration_seconds
from tests import mock_dashscope

PLACEHOLDER_RIFF_SIZE = 2147483583
PLACEHOLDER_DATA_SIZE = 2147483547


def build_wav(payload: bytes = b"\x00\x01" * 12000, *, rate: int = 24000) -> bytes:
    header = (
        b"RIFF"
        + struct.pack("<I", 36 + len(payload))
        + b"WAVE"
        + b"fmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
        + b"data"
        + struct.pack("<I", len(payload))
    )
    return header + payload


def with_placeholder_sizes(data: bytes) -> bytes:
    """Reproduce the streaming header DashScope returns."""
    patched = bytearray(data)
    struct.pack_into("<I", patched, 4, PLACEHOLDER_RIFF_SIZE)
    struct.pack_into("<I", patched, 40, PLACEHOLDER_DATA_SIZE)
    return bytes(patched)


class TestNormalization:
    def test_placeholder_sizes_are_rewritten(self) -> None:
        real = build_wav()
        broken = with_placeholder_sizes(real)

        fixed = normalize_wav_sizes(broken)

        assert fixed[:4] == b"RIFF"
        assert struct.unpack_from("<I", fixed, 4)[0] == len(fixed) - 8
        assert struct.unpack_from("<I", fixed, 40)[0] == len(real) - 44
        assert fixed == real

    def test_a_correct_header_is_returned_untouched(self) -> None:
        real = build_wav()
        assert normalize_wav_sizes(real) is real

    def test_non_wav_data_is_untouched(self) -> None:
        mp3 = b"ID3\x04\x00\x00" + b"\x00" * 100
        assert normalize_wav_sizes(mp3) is mp3

    def test_short_buffers_are_untouched(self) -> None:
        assert normalize_wav_sizes(b"RIFF\x00\x00") == b"RIFF\x00\x00"
        assert normalize_wav_sizes(b"") == b""

    def test_a_truncated_payload_is_reported_honestly(self) -> None:
        """A short download must not keep claiming a 12-hour duration."""
        real = build_wav()
        truncated = real[: len(real) // 2]
        broken = with_placeholder_sizes(truncated)

        fixed = normalize_wav_sizes(broken)

        assert struct.unpack_from("<I", fixed, 40)[0] == len(fixed) - 44

    def test_a_legitimate_size_after_a_trailing_chunk_is_kept(self) -> None:
        # ``data`` followed by a LIST chunk: the declared size is meaningful, so
        # the buffer must be left alone.
        payload = b"\x00\x01" * 100
        base = build_wav(payload)
        list_chunk = b"LIST" + struct.pack("<I", 4) + b"INFO"
        combined = base + list_chunk
        patched = bytearray(combined)
        struct.pack_into("<I", patched, 4, len(combined) - 8)
        candidate = bytes(patched)
        assert normalize_wav_sizes(candidate) is candidate

    def test_wave_module_reports_a_sane_duration_after_fixing(self) -> None:
        real = build_wav(rate=24000)
        broken = with_placeholder_sizes(real)

        # Demonstrate the bug: the placeholder header makes ``wave`` claim a
        # ~12 hour clip (1,073,741,773 frames) for a half-second payload.
        with wave.open(BytesIO(broken), "rb") as handle:
            assert handle.getnframes() > 1_000_000

        fixed = normalize_wav_sizes(broken)
        with wave.open(BytesIO(fixed), "rb") as handle:
            assert handle.getnframes() == 12000
            assert handle.getframerate() == 24000


class TestDuration:
    def test_duration_of_a_known_payload(self) -> None:
        data = build_wav(b"\x00\x01" * 24000, rate=24000)
        assert wav_duration_seconds(data) == pytest.approx(1.0, abs=0.01)

    def test_duration_ignores_a_placeholder_header(self) -> None:
        broken = with_placeholder_sizes(build_wav(b"\x00\x01" * 12000))
        # 12000 frames at 24 kHz = 0.5 s, not 12 hours.
        assert wav_duration_seconds(broken) == pytest.approx(0.5, abs=0.01)

    def test_non_wav_is_zero(self) -> None:
        assert wav_duration_seconds(b"OggS\x00\x02" + b"\x00" * 100) == 0.0
        assert wav_duration_seconds(b"") == 0.0


class TestStoreIntegration:
    async def test_saved_audio_is_always_well_formed(self, tmp_path) -> None:
        store = AudioStore(tmp_path / "audio")
        broken = with_placeholder_sizes(build_wav())
        path = await store.save(broken)

        saved = path.read_bytes()
        assert struct.unpack_from("<I", saved, 4)[0] == len(saved) - 8
        with wave.open(str(path), "rb") as handle:
            assert handle.getnframes() == 12000

    async def test_a_healthy_file_is_stored_byte_for_byte(self, tmp_path) -> None:
        store = AudioStore(tmp_path / "audio")
        real = mock_dashscope.make_wav()
        path = await store.save(real)
        assert path.read_bytes() == real
        assert detect_audio_suffix(path.read_bytes()) == ".wav"
