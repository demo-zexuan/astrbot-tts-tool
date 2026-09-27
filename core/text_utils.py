"""Wake-word matching and text preparation for speech synthesis.

I. Responsibilities

1. Detect whether a message starts with a configured wake word and return the
   remaining payload text.
2. Turn chat-flavoured text (markdown, URLs, emoji, CQ codes) into text that a
   TTS model reads out loud naturally.
3. Cut overly long text into TTS-friendly segments at sentence boundaries.

II. Design notes

1. The module is pure: no network, no AstrBot imports, no global state.
2. Regexes are compiled once at import time.

@author zexuan.peng <pengzexuan2001@gmail.com>
"""

from __future__ import annotations

import re

# Characters that may legally follow a wake word when strict matching is on.
WAKE_SEPARATORS = " \t\u3000,，.。!！?？:：;；、~～-—…|/\\\"'“”‘’()（）[]【】<>《》*_+="

# Characters that are invisible but frequently injected by chat platforms.
INVISIBLE_CHARS = "\u200b\u200c\u200d\u2060\ufeff\u200e\u200f"

_LEADING_NOISE = re.compile(rf"^[\s{re.escape(INVISIBLE_CHARS)}]+")
_CQ_CODE = re.compile(r"\[CQ:[^\]]*\]")
_FENCED_CODE = re.compile(r"```[a-zA-Z0-9_+-]*\n?(.*?)```", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`]*)`")
_IMAGE_MD = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK_MD = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_URL = re.compile(r"(?:https?://|www\.)[^\s\u3000]+", re.IGNORECASE)
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_BLOCKQUOTE = re.compile(r"^\s{0,3}>\s?", re.MULTILINE)
_LIST_MARKER = re.compile(r"^\s{0,3}(?:[-*+]|\d{1,2}[.)])\s+", re.MULTILINE)
_HRULE = re.compile(r"^\s{0,3}(?:[-*_]\s*){3,}$", re.MULTILINE)
_EMPHASIS = re.compile(r"(\*\*|__|\*|_|~~)")
_TABLE_PIPE = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
#: Any run of whitespace (including newlines) becomes a single space, which is
#: the safest thing to hand to a speech model.
_WHITESPACE = re.compile(r"\s+")

# Emoji / pictograph ranges that are noise for a speech model.
_EMOJI = re.compile(
    "["
    "\U0001f000-\U0001faff"  # pictographs, emoticons, transport, symbols
    "\U00002600-\U000027bf"  # misc symbols and dingbats
    "\U0001f1e6-\U0001f1ff"  # regional indicators (flags)
    "\U00002b00-\U00002bff"  # misc symbols and arrows
    "\U00002190-\U000021ff"  # arrows
    "\U0000fe0f"  # variation selector-16
    "\U0000200d"  # zero-width joiner
    "\U00003030\U0000303d\U00002122\U00002139"
    "]+",
)

# Sentence terminators used when splitting long text.
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?；;…\n])")
# Preferred break points when a single sentence exceeds the limit.
_SOFT_BREAK = "，,、：:；; "


# ------------------------------------------------------------ wake word ---


def normalize_incoming(text: str) -> str:
    """Strip leading whitespace and invisible characters from ``text``."""
    if not text:
        return ""
    return _LEADING_NOISE.sub("", text)


def match_wake_word(
    text: str,
    wake_words: tuple[str, ...] | list[str],
    *,
    require_separator: bool = False,
    case_sensitive: bool = False,
) -> tuple[str | None, str]:
    """Return ``(matched_wake_word, remaining_text)``.

    I. Matching rules

    1. Only a prefix match counts; the wake word must be at the very start.
    2. Candidate wake words are expected to be pre-sorted longest-first, but the
       function sorts defensively so callers cannot get it wrong.
    3. With ``require_separator`` the character after the wake word must be a
       separator or the end of the string, which stops "小助手们" from waking the
       bot when the wake word is "小助手".

    II. Return value

    1. ``(None, original_text)`` when nothing matched; the remaining text is the
       normalised (not raw) message so callers never re-handle leading noise.
    """
    normalized = normalize_incoming(text or "")
    if not normalized or not wake_words:
        return None, normalized

    candidates = sorted({w for w in wake_words if w}, key=len, reverse=True)
    haystack = normalized if case_sensitive else normalized.casefold()

    for candidate in candidates:
        needle = candidate if case_sensitive else candidate.casefold()
        if not haystack.startswith(needle):
            continue
        rest = normalized[len(candidate) :]
        if require_separator and rest and rest[0] not in WAKE_SEPARATORS:
            continue
        return candidate, rest.lstrip(WAKE_SEPARATORS)
    return None, normalized


def strip_wake_word(text: str, wake_word: str) -> str:
    """Remove a leading ``wake_word`` plus any separator that follows it."""
    normalized = normalize_incoming(text or "")
    if not wake_word:
        return normalized
    if normalized.casefold().startswith(wake_word.casefold()):
        rest = normalized[len(wake_word) :]
        return rest.lstrip(WAKE_SEPARATORS)
    return normalized


def extract_command_argument(message_str: str) -> str:
    """Return everything after the leading command token of ``message_str``.

    AstrBot strips the wake prefix (``/`` by default) before handlers run, so
    ``/tts 你好 世界`` reaches a handler as ``tts 你好 世界``. Dropping the first
    whitespace-delimited token therefore yields the raw free-form argument,
    including internal spaces, without depending on AstrBot's parameter parser.
    """
    text = (message_str or "").strip()
    if not text:
        return ""
    parts = text.split(None, 1)
    return parts[1].strip() if len(parts) > 1 else ""


# -------------------------------------------------------------- sanitize ---


def sanitize_text(
    text: str,
    *,
    clean_markdown: bool = True,
    clean_url: bool = True,
    clean_emoji: bool = False,
) -> str:
    """Prepare chat text for speech synthesis.

    I. Order of operations

    1. Drop platform artefacts (CQ codes, invisible characters).
    2. Unwrap markdown constructs, keeping their human-readable content.
    3. Optionally remove URLs and emoji.
    4. Collapse whitespace and trim.

    The order matters: URLs inside markdown links must be resolved before the
    generic URL rule runs, otherwise the visible link text would be deleted too.
    """
    if not text:
        return ""

    result = str(text)
    result = result.replace("\r\n", "\n").replace("\r", "\n")
    result = _CQ_CODE.sub(" ", result)
    for char in INVISIBLE_CHARS:
        result = result.replace(char, "")

    if clean_markdown:
        result = _FENCED_CODE.sub(r"\1", result)
        result = _INLINE_CODE.sub(r"\1", result)
        # Alt text and link labels survive; the target URL does not.
        result = _IMAGE_MD.sub(r"\1", result)
        result = _LINK_MD.sub(r"\1", result)
        result = _HRULE.sub(" ", result)
        result = _HEADING.sub("", result)
        result = _BLOCKQUOTE.sub("", result)
        result = _LIST_MARKER.sub("", result)
        result = _TABLE_PIPE.sub(lambda m: m.group(0).strip("| "), result)
        result = _EMPHASIS.sub("", result)

    if clean_url:
        result = _URL.sub(" ", result)

    if clean_emoji:
        result = _EMOJI.sub(" ", result)

    result = _WHITESPACE.sub(" ", result)
    return result.strip()


# --------------------------------------------------------------- segment ---


def split_for_tts(
    text: str,
    *,
    max_chars: int,
    max_segments: int,
    split_long_text: bool = True,
) -> tuple[list[str], bool]:
    """Split ``text`` into segments no longer than ``max_chars``.

    Returns ``(segments, truncated)``. ``truncated`` is ``True`` when content had
    to be dropped, either because ``max_segments`` was reached or because
    splitting was disabled.

    I. Strategy

    1. A segment boundary is preferred at sentence terminators.
    2. Sentences are packed greedily until adding the next one would overflow.
    3. A single sentence longer than ``max_chars`` is cut at the last soft break
       (comma/space) inside the window, or hard-cut when no break exists.
    4. When ``split_long_text`` is disabled the text is cut once at
       ``max_chars`` and everything after it is dropped.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return [], False
    if max_chars < 1:
        return [cleaned], False
    if max_segments < 1:
        max_segments = 1

    if not split_long_text:
        if len(cleaned) <= max_chars:
            return [cleaned], False
        return [cleaned[:max_chars].rstrip()], True

    sentences = [s for s in _SENTENCE_SPLIT.split(cleaned) if s and s.strip()]
    segments: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current.strip():
            segments.append(current.strip())
        current = ""

    for sentence in sentences:
        piece = sentence.strip()
        while len(piece) > max_chars:
            # A single sentence is longer than the whole budget.
            if len(current) + 1 >= max_chars:
                flush()
            window = piece[:max_chars]
            cut = max((window.rfind(ch) for ch in _SOFT_BREAK), default=-1)
            if cut < max_chars // 2:
                cut = max_chars - 1
            head, piece = piece[: cut + 1].strip(), piece[cut + 1 :].strip()
            if head:
                if current:
                    flush()
                current = head
                flush()
        if not piece:
            continue
        if not current:
            current = piece
        elif len(current) + 1 + len(piece) <= max_chars:
            current = f"{current} {piece}" if _needs_space(current, piece) else f"{current}{piece}"
        else:
            flush()
            current = piece

    flush()

    truncated = False
    if len(segments) > max_segments:
        segments = segments[:max_segments]
        truncated = True
    return segments, truncated


def _needs_space(left: str, right: str) -> bool:
    """Insert a space only between two ASCII word characters."""
    if not left or not right:
        return False
    return left[-1].isascii() and left[-1].isalnum() and right[0].isascii() and right[0].isalnum()
