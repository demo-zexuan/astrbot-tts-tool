"""Local storage for synthesized audio files.

I. Why the files land on disk

1. ``Comp.Record`` is happiest with a local file: AstrBot then converts it to the
   format each platform needs (wav for OneBot, silk for WeChat, ...).
2. DashScope returns a URL that expires after 24 hours, so keeping our own copy
   also makes replays and debugging possible.

II. Where they land

Per the AstrBot plugin storage convention everything lives under
``data/plugin_data/<plugin_dir>/`` so it survives plugin upgrades. ``main.py``
passes that directory in; this module never guesses paths.

III. Housekeeping

Files older than ``ttl_hours`` are deleted, and the newest ``max_files`` are
kept. A background task in ``main.py`` calls :meth:`AudioStore.prune`
periodically; the store also prunes opportunistically after each save.

@author zexuan.peng <pengzexuan2001@gmail.com>
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
from pathlib import Path
from typing import Any

from .errors import TtsError
from .wav import normalize_wav_sizes

# Magic numbers we can recognise; anything else is stored as ``.wav`` because
# that is what the TTS endpoint is documented to return.
_MAGIC_PREFIXES: tuple[tuple[bytes, str], ...] = (
    (b"RIFF", ".wav"),
    (b"ID3", ".mp3"),
    (b"OggS", ".ogg"),
    (b"fLaC", ".flac"),
)

AUDIO_SUFFIXES = frozenset({".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac", ".amr", ".silk"})


def detect_audio_suffix(data: bytes, fallback: str = ".wav") -> str:
    """Guess the audio container from its magic bytes."""
    if len(data) >= 4:
        if data[:4] == b"RIFF" and len(data) >= 12 and data[8:12] == b"WAVE":
            return ".wav"
        for prefix, suffix in _MAGIC_PREFIXES:
            if data.startswith(prefix):
                return suffix
        # MPEG frame sync (MP3 without an ID3 tag).
        if data[0] == 0xFF and (data[1] & 0xE0) == 0xE0:
            return ".mp3"
    return fallback


class AudioStore:
    """Filesystem-backed cache of synthesized audio."""

    def __init__(
        self,
        directory: Path | str,
        *,
        ttl_hours: float = 6.0,
        max_files: int = 200,
        logger: Any | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.ttl_hours = max(0.0, float(ttl_hours))
        self.max_files = max(1, int(max_files))
        self._logger = logger

    def _log(self, level: str, message: str, *args: Any) -> None:
        if self._logger is None:
            return
        log_method = getattr(self._logger, level, None)
        if callable(log_method):
            log_method(message, *args)

    # ---------------------------------------------------------------- io ---

    def ensure_ready(self) -> None:
        """Create the cache directory if needed."""
        self.directory.mkdir(parents=True, exist_ok=True)

    async def save(self, data: bytes, *, prefix: str = "tts") -> Path:
        """Persist ``data`` and return the created path.

        I. What happens here

        1. Streaming WAV headers are corrected so downstream players see the
           real duration instead of a placeholder size (see ``core/wav.py``).
        2. The write is atomic (temp file + ``os.replace``) so a crash can never
           leave a half-written file that a later send would pick up.
        """
        if not data:
            raise TtsError("refusing to store empty audio", hint="合成音频为空，未发送。")

        data = normalize_wav_sizes(data)
        self.ensure_ready()
        suffix = detect_audio_suffix(data)
        digest = hashlib.sha1(data).hexdigest()[:10]
        stamp = f"{time.time():.6f}".replace(".", "")
        target = self.directory / f"{prefix}_{stamp}_{digest}{suffix}"

        await asyncio.to_thread(_atomic_write, target, data)
        await self.prune()
        return target

    # ------------------------------------------------------------- prune ---

    def _list_audio_files(self) -> list[Path]:
        if not self.directory.is_dir():
            return []
        files: list[Path] = []
        try:
            for entry in self.directory.iterdir():
                if entry.is_file() and entry.suffix.lower() in AUDIO_SUFFIXES:
                    files.append(entry)
        except OSError as exc:
            self._log("warning", "Failed to scan audio cache: %s", exc)
        return files

    def prune_sync(self) -> int:
        """Delete expired and surplus files. Returns the number removed."""
        removed = 0
        files = self._list_audio_files()
        if not files:
            return 0

        now = time.time()
        ttl_seconds = self.ttl_hours * 3600.0
        survivors: list[tuple[float, Path]] = []

        for path in files:
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if ttl_seconds > 0 and now - mtime > ttl_seconds:
                removed += _unlink(path)
            else:
                survivors.append((mtime, path))

        if len(survivors) > self.max_files:
            survivors.sort(key=lambda item: item[0])
            for _, path in survivors[: len(survivors) - self.max_files]:
                removed += _unlink(path)

        if removed:
            self._log("debug", "Pruned %s cached audio file(s)", removed)
        return removed

    async def prune(self) -> int:
        """Thread-offloaded variant of :meth:`prune_sync`."""
        return await asyncio.to_thread(self.prune_sync)

    def count(self) -> int:
        """Number of audio files currently cached."""
        return len(self._list_audio_files())


def _atomic_write(target: Path, data: bytes) -> None:
    tmp = target.with_name(f".{target.name}.part")
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, target)


def _unlink(path: Path) -> int:
    try:
        os.unlink(path)
    except OSError:
        return 0
    return 1
