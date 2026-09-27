"""Framework-independent core of the wake-word TTS plugin.

Everything inside this package is deliberately free of ``astrbot`` imports so it
can be unit tested without a running AstrBot instance. ``main.py`` owns the only
AstrBot-facing glue code.

@author zexuan.peng <pengzexuan2001@gmail.com>
"""

from .audio_store import AudioStore
from .config import (
    HTTP_TTS_MODELS,
    PluginSettings,
    TtsProfile,
    mask_secret,
)
from .dashscope_client import DashScopeTtsClient, SynthesisResult
from .director import (
    DirectorResult,
    TtsCapabilities,
    build_messages,
    parse_director_response,
    strip_forbidden_tags,
)
from .errors import (
    LlmError,
    TtsApiError,
    TtsConfigError,
    TtsError,
    TtsNetworkError,
    TtsTextError,
)
from .llm_client import LlmChatResult, OpenAiCompatibleLlmClient
from .service import SynthesisOutcome, TtsService
from .text_utils import (
    extract_command_argument,
    match_wake_word,
    sanitize_text,
    split_for_tts,
    strip_wake_word,
)
from .wav import normalize_wav_sizes, wav_duration_seconds

__all__ = [
    "HTTP_TTS_MODELS",
    "AudioStore",
    "DashScopeTtsClient",
    "DirectorResult",
    "LlmChatResult",
    "LlmError",
    "OpenAiCompatibleLlmClient",
    "PluginSettings",
    "SynthesisOutcome",
    "SynthesisResult",
    "TtsApiError",
    "TtsCapabilities",
    "TtsConfigError",
    "TtsError",
    "TtsNetworkError",
    "TtsProfile",
    "TtsService",
    "TtsTextError",
    "build_messages",
    "extract_command_argument",
    "mask_secret",
    "match_wake_word",
    "normalize_wav_sizes",
    "parse_director_response",
    "sanitize_text",
    "split_for_tts",
    "strip_forbidden_tags",
    "strip_wake_word",
    "wav_duration_seconds",
]
