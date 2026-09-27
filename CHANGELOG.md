# 更新日志

本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)，
提交信息遵循 [Conventional Commits + Emoji](https://www.conventionalcommits.org/)。

---

## [v1.1.0] - 2026-09-27

### ✨ 新增：LLM 预处理（文本导演）

在调用 TTS 模型之前，先调用一次大模型，把聊天消息改写成更适合朗读的台词。

- **free-style 指令遵循**：把「温柔一点念」「像哄睡一样」这类自然语言要求，
  转成真正的用词、标点与断句。
- **细粒度标签控制**：可按目标模型能力注入标签（Qwen-Audio-TTS / CosyVoice 风格
  或 SSML），标签词表会写进提示词，模型无法自创标签。
- **上下文感知**：自动读取当前会话最近 N 条消息，让语气贴合对话氛围。
- **安全兜底**：提示词强制「只改怎么说、不改说什么」，并要求模型只输出 JSON；
  解析失败时回退为原文。目标模型不支持 `instructions` 或行内标签时，
  插件会自动丢弃对应内容，避免标签被逐字朗读。

### ✨ 新增：两种 LLM 接入方式

- **复用 AstrBot Provider（默认）**：通过 `Provider.text_chat` 调用，
  `api_base`、API Key 与 **`custom_headers`** 全部由 AstrBot 管理，
  Opencode 等需要特殊请求头的网关只要在 AstrBot 里配好即可直接使用。
- **插件内直连**：面向未注册到 AstrBot 的 OpenAI 兼容网关，支持
  `bearer` / `x-api-key` / 无鉴权三种方式，并可用「自定义请求头」
  补齐 `anthropic-version`、`X-Title` 等特殊头；同名 Header 会覆盖自动生成的鉴权头。
- 调用通过签名内省适配，Provider 只暴露 `prompt` 也能正常工作。

### ✨ 新增：新指令与观测

- `/tts_llm <文本>`：只运行 LLM 预处理并显示改写结果、语气指令、标签与模型说明，
  不发送语音，便于调参。
- `/tts_status` 增加 LLM 阶段详情（接入方式、Provider、改写开关、标签类型、
  上下文条数、失败兜底、直连地址与自定义请求头名称），API Key 始终掩码显示。

### ✅ 测试

- 新增 130+ 条离线测试（总计 323 条），覆盖提示词构建、响应解析、标签清理、
  直连客户端请求头（逐项断言）、两种接入方式的端到端流程、上下文读取与失败兜底。
- 新增 2 条真机测试：用真实大模型验证导演输出的 JSON 契约，
  以及「唤醒词 → 真实 LLM → 真实 TTS → 音频文件」的完整链路。
- AstrBot 源码一致性测试扩展到 Provider API：`get_provider_by_id` /
  `get_using_provider` / `text_chat` 参数 / `custom_headers` 传递 /
  `conversation_manager` / `select_provider`。

### 📝 文档

- README 新增「让 LLM 先改写台词」章节，含 Opencode 等网关的配置说明与 FAQ。

---

## [v1.0.0] - 2026-09-27

### ✨ 首次发布

- 唤醒词触发 TTS：消息以配置的唤醒词开头即合成语音并回复当前会话。
- 支持多唤醒词，并可为每个唤醒词绑定专属模型 / 音色 / 语种 / 语气指令。
- 指令：`/tts`、`/tts_voice`、`/tts_status`、`/tts_help`。
- 基于 DashScope Qwen-TTS HTTP 非实时接口，默认模型 `qwen3-tts-flash`。
- 文本清洗（Markdown / 链接 / Emoji / CQ 码）、长文本按句子自动分段。
- 失败重试与指数退避、并发上限、会话冷却、可读的中文错误提示。
- 音频缓存按时间与数量自动回收，数据存放于 `data/plugin_data/`。
- **WAV 文件头修正**：实测发现 Qwen-TTS 返回的 WAV 使用流式占位文件头
  （声明 `data` 长度 2,147,483,547 字节，真实仅 149,760 字节，
  `wave` 模块会读成约 12 小时），插件在保存时改写为真实长度。
- 323 条离线测试 + 源码一致性测试 + 真机联调测试。

### ⚠️ 已知限制

- `qwen-audio-3.1-tts-flash` 等 Qwen-Audio-TTS 系列只有实时（WebSocket）接口，
  无法通过本插件的 HTTP 通道调用；已内置针对该错误的可读提示。
