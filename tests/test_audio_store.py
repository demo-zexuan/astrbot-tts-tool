"""Tests for the on-disk audio cache.

@author zexuan.peng <pengzexuan2001@gmail.com>
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from core.audio_store import AudioStore, detect_audio_suffix
from core.errors import TtsError
from tests import mock_dashscope


class TestDetection:
    def test_known_containers(self) -> None:
        assert detect_audio_suffix(mock_dashscope.make_wav()) == ".wav"
        assert detect_audio_suffix(b"ID3\x04\x00\x00") == ".mp3"
        assert detect_audio_suffix(b"OggS\x00\x02") == ".ogg"
        assert detect_audio_suffix(b"fLaC\x00\x00") == ".flac"

    def test_unknown_bytes_default_to_wav(self) -> None:
        assert detect_audio_suffix(b"\x01\x02\x03\x04") == ".wav"

    def test_riff_without_wave_marker_is_not_wav(self) -> None:
        assert detect_audio_suffix(b"RIFF\x00\x00\x00\x00AVI ") == ".wav"


class TestSave:
    async def test_saves_a_wav_file_under_the_cache_dir(self, tmp_path: Path) -> None:
        store = AudioStore(tmp_path / "audio")
        path = await store.save(mock_dashscope.make_wav(), prefix="tts")

        assert path.exists()
        assert path.suffix == ".wav"
        assert path.read_bytes().startswith(b"RIFF")
        assert path.parent == tmp_path / "audio"

    async def test_no_temporary_files_are_left_behind(self, tmp_path: Path) -> None:
        store = AudioStore(tmp_path / "audio")
        await store.save(mock_dashscope.make_wav())
        leftovers = [p.name for p in (tmp_path / "audio").iterdir() if p.name.startswith(".")]
        assert leftovers == []

    async def test_identical_payloads_get_distinct_names(self, tmp_path: Path) -> None:
        store = AudioStore(tmp_path / "audio")
        audio = mock_dashscope.make_wav()
        first = await store.save(audio)
        second = await store.save(audio)
        assert first != second
        assert first.read_bytes() == second.read_bytes()

    async def test_empty_payload_is_rejected(self, tmp_path: Path) -> None:
        store = AudioStore(tmp_path / "audio")
        with pytest.raises(TtsError):
            await store.save(b"")

    async def test_count_reports_saved_files(self, tmp_path: Path) -> None:
        store = AudioStore(tmp_path / "audio")
        assert store.count() == 0
        await store.save(mock_dashscope.make_wav())
        await store.save(mock_dashscope.make_wav())
        assert store.count() == 2


class TestPrune:
    async def test_files_older_than_the_ttl_are_removed(self, tmp_path: Path) -> None:
        store = AudioStore(tmp_path / "audio", ttl_hours=1.0)
        keep = await store.save(mock_dashscope.make_wav())
        old = await store.save(mock_dashscope.make_wav())

        stale = time.time() - 7200
        os.utime(old, (stale, stale))

        removed = await store.prune()
        assert removed == 1
        assert keep.exists()
        assert not old.exists()

    async def test_zero_ttl_disables_age_based_cleanup(self, tmp_path: Path) -> None:
        store = AudioStore(tmp_path / "audio", ttl_hours=0.0)
        path = await store.save(mock_dashscope.make_wav())
        stale = time.time() - 999999
        os.utime(path, (stale, stale))
        assert await store.prune() == 0
        assert path.exists()

    async def test_oldest_files_are_dropped_beyond_the_file_limit(self, tmp_path: Path) -> None:
        store = AudioStore(tmp_path / "audio", ttl_hours=0.0, max_files=3)
        paths = []
        for index in range(5):
            path = await store.save(mock_dashscope.make_wav(freq=200 + index * 50))
            stamp = time.time() - (10 - index)
            os.utime(path, (stamp, stamp))
            paths.append(path)

        # The store prunes opportunistically on save, so trim back to the limit.
        removed = await store.prune()
        assert removed >= 0
        assert store.count() <= 3
        assert paths[-1].exists()

    async def test_prune_on_a_missing_directory_is_a_noop(self, tmp_path: Path) -> None:
        store = AudioStore(tmp_path / "does-not-exist")
        assert await store.prune() == 0

    async def test_non_audio_files_are_ignored(self, tmp_path: Path) -> None:
        store = AudioStore(tmp_path / "audio", ttl_hours=0.0, max_files=1)
        await store.save(mock_dashscope.make_wav())
        stray = tmp_path / "audio" / "notes.txt"
        stray.write_text("keep me", encoding="utf-8")
        await store.prune()
        assert stray.exists()
