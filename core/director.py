"""The pre-synthesis "director" stage: LLM-driven script preparation.

I. What the director does

Before a single character reaches the TTS model, an LLM turns the raw chat
message into a *spoken script*:

1. **Free-style instruction following** – "温柔一点念，像哄睡一样" is turned into
   wording, punctuation and pacing that actually sounds that way.
2. **Fine-grained tag control** – when the target model supports inline tags
   (``<|EMO|>``-style or SSML), the director may insert them; the allowed tag
   vocabulary is injected into the prompt so the model cannot invent tags.
3. **Context awareness** – recent conversation turns are supplied so the delivery
   matches what is being discussed, without the LLM answering the message.

II. Two safety rules

1. The director must *never* change what is being said, only how it is said; the
   prompt states this and the plugin keeps the original text as a fallback.
2. The director must not emit controls the target model cannot handle. Withline
   tags are only offered when ``llm_tag_style`` is not ``none``, and
   ``instructions`` is only requested when the chosen model supports it.

@author zexuan.peng <pengzexuan2001@gmail.com>
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .config import (
    TAG_STYLE_CUSTOM,
    TAG_STYLE_NONE,
    TAG_STYLE_QWEN_AUDIO,
    TAG_STYLE_SSML,
    PluginSettings,
)

#: Tag vocabularies injected into the prompt. Values are documentation only —
#: nothing here is executed.
TAG_VOCABULARIES: dict[str, str] = {
    TAG_STYLE_QWEN_AUDIO: (
        "允许使用 Qwen-Audio-TTS / CosyVoice 风格的细粒度标签，只能从下列标签中选择，"
        "且必须原样书写（尖括号、竖线都不能省略）：\n"
        "- <|EMO|>：切换情绪，后接情绪词（如 <|EMO|>开心）\n"
        "- <|SPEED|>：切换语速，后接 slow/normal/fast\n"
        "- <|BREATH|>：插入一次换气停顿\n"
        "- <|PAUSE|>：插入一次短停顿\n"
        "- <|LAUGH|>：插入一次轻笑\n"
        "标签之间要有正常文本，不要连续堆叠，每 100 字内不超过 8 个标签。"
    ),
    TAG_STYLE_SSML: (
        "允许使用 SSML 标签，只能从下列标签中选择：\n"
        '- <break time="300ms"/>：停顿\n'
        '- <emphasis level="strong">词</emphasis>：加重\n'
        '- <prosody rate="slow" pitch="+2st">词</prosody>：语速与音高\n'
        "注意：标签必须成对闭合，不要产生嵌套超过两层的结构。"
    ),
}

#: Patterns removed from the final text when inline tags are disabled, so a
#: chatty LLM cannot make the TTS engine read tags out loud.
FORBIDDEN_TAG_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"<\|[^|<>]{1,40}\|>"),
    re.compile(r"</?(?:break|emphasis|prosody|speak|voice|phoneme|say-as)\b[^>]*>"),
    re.compile(r"\[\s*(?:EMO|SPEED|BREATH|PAUSE|LAUGH|TTS)\s*[^\]]*\]", re.IGNORECASE),
)

_CODE_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


# ------------------------------------------------------------ capabilities ---


@dataclass(frozen=True, slots=True)
class TtsCapabilities:
    """What the selected TTS model can actually be told to do."""

    model: str
    voice: str
    language_type: str
    max_chars: int
    max_segments: int
    tag_style: str = TAG_STYLE_NONE
    custom_tags: str = ""

    @property
    def supports_instructions(self) -> bool:
        """``instructions`` is documented for the instruct model family only.

        Sending it to ``qwen3-tts-flash`` can be rejected, so it is only offered
        when the model name says it supports it.
        """
        return "instruct" in self.model.lower()

    @property
    def tags_enabled(self) -> bool:
        return self.tag_style != TAG_STYLE_NONE

    def describe(self) -> str:
        """Render the capability block that goes into the system prompt."""
        budget = self.max_chars * max(1, self.max_segments)
        lines = [
            f"- 目标模型：{self.model}（音色 {self.voice}，语种 {self.language_type}）",
            f"- 输出文本上限：约 {budget} 字，单段不超过 {self.max_chars} 字",
        ]
        if self.supports_instructions:
            lines.append(
                "- 该模型支持 instructions 字段：可以另外输出一段自然语言语气指令"
                "（控制语速、情绪、风格）。",
            )
        else:
            lines.append(
                "- 该模型不支持 instructions 字段：语气只能通过文本本身（用词、标点、"
                "断句、拟声词）体现，不要指望额外指令生效。",
            )
        if self.tags_enabled:
            lines.append("- 允许使用细粒度标签：")
            lines.append(self._tag_help())
        else:
            lines.append(
                "- 严格禁止输出任何标签、SSML、音效标记或舞台提示"
                "（例如 <|EMO|>、[pause]、<break/>），它们会被原样朗读出来。",
            )
        return "\n".join(lines)

    def _tag_help(self) -> str:
        if self.tag_style == TAG_STYLE_CUSTOM:
            custom = self.custom_tags.strip()
            return custom or "（未填写自定义标签说明，请不要输出任何标签。）"
        return TAG_VOCABULARIES.get(self.tag_style, "（无）")

    @classmethod
    def from_settings(cls, settings: PluginSettings) -> TtsCapabilities:
        return cls(
            model=settings.model,
            voice=settings.voice,
            language_type=settings.language_type,
            max_chars=settings.max_chars,
            max_segments=settings.max_segments,
            tag_style=settings.llm_tag_style,
            custom_tags=settings.llm_custom_tags,
        )


# ---------------------------------------------------------------- result ---


@dataclass(slots=True)
class DirectorResult:
    """The script the TTS model will actually speak."""

    text: str
    instructions: str = ""
    tags: tuple[str, ...] = ()
    reason: str = ""
    raw: str = ""
    used_fallback: bool = False
    stripped_tags: tuple[str, ...] = ()
    model: str = ""


# --------------------------------------------------------------- prompting ---

DEFAULT_SYSTEM_PROMPT = """你是语音合成前的「文本导演」。你的唯一任务是把用户的一句话改写成\
适合朗读的台词，让语气、节奏和情绪贴合上下文。

硬性规则：
1. 只调整「怎么说」，绝不改变「说什么」。不得增加事实、不得回答用户问题、不得添加\
自己编造的内容或总结。
2. 必须保持原意与原语言，长度不要超过原文的两倍。
3. 只输出一个 JSON 对象，不要输出解释、Markdown 代码块以外的任何文字。

输出格式（严格遵守）：
{"text": "最终要朗读的文本", "instructions": "语气指令（若模型不支持则留空字符串）",\
 "tags": ["用到的标签"], "reason": "一句话说明你做了什么调整"}

改写手法（按需使用，不要为了改而改）：
- 用标点控制停顿：句号停顿长、逗号停顿短、省略号表示迟疑。
- 用语气词、拟声词与短句营造情绪（如「呀」「呢」「嗯——」）。
- 长句拆成短句，避免一口气念不完；数字与符号改成口头读法。
- 若原文已经是自然的口语，可以原样保留。"""


def _render_context(history: list[dict[str, Any]] | None) -> str:
    """Render recent turns as a compact, role-labelled transcript."""
    if not history:
        return "（无历史消息）"
    lines: list[str] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "user")
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        label = {"user": "用户", "assistant": "机器人", "system": "系统"}.get(role, role)
        lines.append(f"{label}：{content}")
    return "\n".join(lines) if lines else "（无历史消息）"


def build_messages(
    *,
    text: str,
    capabilities: TtsCapabilities,
    history: list[dict[str, Any]] | None = None,
    system_prompt: str = "",
    allow_rewrite: bool = True,
) -> list[dict[str, str]]:
    """Build the OpenAI-style message list for the director call."""
    system = (system_prompt or "").strip() or DEFAULT_SYSTEM_PROMPT
    if not allow_rewrite:
        system += (
            "\n\n附加约束（本次由配置强制）：不得改动原文字句，"
            '"text" 必须与用户原文完全一致，只允许填写 "instructions" 与 "tags"。'
        )

    user = (
        "【目标模型能力】\n"
        f"{capabilities.describe()}\n\n"
        "【最近的对话上下文】\n"
        f"{_render_context(history)}\n\n"
        "【需要朗读的原文】\n"
        f"{text}\n\n"
        "请输出 JSON。"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------- parsing ---


def strip_forbidden_tags(text: str) -> tuple[str, tuple[str, ...]]:
    """Remove inline tags the target model cannot interpret.

    Returns ``(cleaned_text, removed_tags)``. Whitespace left behind by a removed
    tag is collapsed so the spoken text does not gain odd gaps.
    """
    removed: list[str] = []
    cleaned = text
    for pattern in FORBIDDEN_TAG_PATTERNS:
        for match in pattern.finditer(cleaned):
            removed.append(match.group(0))
        cleaned = pattern.sub(" ", cleaned)
    if removed:
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    return cleaned, tuple(removed)


def _extract_json(raw: str) -> dict[str, Any] | None:
    """Pull the first JSON object out of an LLM reply, tolerating prose/fences."""
    text = (raw or "").strip()
    if not text:
        return None

    candidates: list[str] = []
    fenced = _CODE_FENCE.search(text)
    if fenced:
        candidates.append(fenced.group(1).strip())
    candidates.append(text)
    braced = _JSON_OBJECT.search(text)
    if braced:
        candidates.append(braced.group(0))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, int | float):
        return str(value)
    return ""


def parse_director_response(
    raw: str,
    *,
    original_text: str,
    capabilities: TtsCapabilities | None = None,
    model: str = "",
) -> DirectorResult:
    """Turn an LLM reply into a :class:`DirectorResult`.

    I. Fallback chain

    1. A JSON object with a ``text`` field is the happy path.
    2. If the reply is not JSON (or has no usable ``text``), the whole reply is
       treated as the script — a chatty model that ignores the format still
       produces something speakable rather than an error.
    3. If that is empty too, the original text is used and ``used_fallback`` is
       set, which lets the caller log the problem.
    """
    payload = _extract_json(raw or "")
    result = DirectorResult(text="", raw=raw or "", model=model)

    if payload is not None:
        result.text = _as_text(payload.get("text"))
        result.instructions = _as_text(payload.get("instructions"))
        result.reason = _as_text(payload.get("reason"))
        tags = payload.get("tags")
        if isinstance(tags, list):
            result.tags = tuple(_as_text(tag) for tag in tags if _as_text(tag))

    if not result.text:
        stripped = (raw or "").strip()
        # Accept a prose reply only when it can plausibly be spoken: a leftover
        # JSON fragment (`{...` or `[...]`) is a formatting failure, not a script.
        looks_like_json = stripped.startswith(("{", "["))
        if stripped and not looks_like_json and len(stripped) <= 2000:
            result.text = stripped
        else:
            result.text = original_text
            result.used_fallback = True

    if capabilities is not None:
        if not capabilities.supports_instructions:
            # The target model would reject or ignore it, so never forward it.
            result.instructions = ""
        if not capabilities.tags_enabled:
            cleaned, removed = strip_forbidden_tags(result.text)
            if cleaned:
                result.text = cleaned
            result.stripped_tags = removed
            if removed:
                result.tags = ()

    return result
