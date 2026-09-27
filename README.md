# astrbot_plugin_tts_wakeword

> 唤醒词触发的 AstrBot 语音合成插件：说一句「小助手 你好呀」，机器人先用大模型把这句话改写成自然的朗读台词，再用 Qwen-TTS 合成语音发到当前会话。

[![AstrBot](https://img.shields.io/badge/AstrBot-%3E%3D4.9.2-5B5BD6)](https://astrbot.app)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-323%20offline%20%2B%205%20live-brightgreen)](#开发与测试)

<img src="logo.png" alt="插件 Logo" width="128" />

---

## 目录

- [这个插件能做什么](#这个插件能做什么)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [使用方法](#使用方法)
- [让 LLM 先改写台词](#让-llm-先改写台词)
- [模型与音色](#模型与音色)
- [支持的消息平台](#支持的消息平台)
- [工作原理](#工作原理)
- [数据与隐私](#数据与隐私)
- [常见问题](#常见问题)
- [开发与测试](#开发与测试)
- [目录结构](#目录结构)
- [许可与致谢](#许可与致谢)

---

## 这个插件能做什么

| 能力 | 说明 |
| --- | --- |
| 唤醒词触发 | 消息以「小助手」等唤醒词开头，即自动合成语音并回复 |
| LLM 预处理 | 合成前先调用大模型改写台词，支持指令遵循、标签控制与上下文感知 |
| 两种 LLM 接入 | 复用 AstrBot Provider（自动继承 custom_headers）或插件内直连 |
| 唤醒词专属配置 | 每个唤醒词可绑定独立的模型 / 音色 / 语种 / 语气指令 |
| 指令触发 | 同时提供 `/tts`、`/tts_voice`、`/tts_status`、`/tts_help` |
| 模型可配置 | 默认 `qwen3-tts-flash`，支持百炼全部 HTTP 非实时 TTS 模型 |
| 长文本分段 | 超过字数上限时按句子边界自动切成多条语音依次发送 |
| 文本清理 | 自动去掉 Markdown 标记、链接、Emoji、CQ 码，避免朗读出符号 |
| 稳定可靠 | 失败重试 + 指数退避、并发上限、会话冷却、可读的中文错误提示 |
| 自动回收 | 音频缓存按时间与数量自动清理，数据放在 `data/plugin_data/` |

一句话总结：**把「说唤醒词」变成「收到语音」，其余全部可配置。**

---

## 快速开始

### 1. 安装插件

**方式 A：从插件市场安装（推荐）**

在 AstrBot WebUI 的「插件」页面搜索本插件名称安装。

**方式 B：手动安装**

```bash
cd AstrBot/data/plugins
git clone https://github.com/demo-zexuan/astrbot-tts-tool.git astrbot_plugin_tts_wakeword
```

然后在 WebUI 的「插件」页面点击刷新图标重载插件。

> 目录名可以自定义（例如 `astrbot-tts-tool`），插件数据目录会跟随实际目录名，
> 但推荐使用 `astrbot_plugin_tts_wakeword` 以符合 AstrBot 命名约定。

如果安装后提示缺少依赖，在插件目录执行：

```bash
pip install -r requirements.txt
```

### 2. 获取 API Key

1. 打开 [阿里云百炼控制台](https://bailian.console.aliyun.com/?tab=model#/api-key) 创建 API Key。
2. 在 WebUI「插件」页面打开本插件配置，把 Key 填进 **DashScope API Key**。
3. 确认「接口地域」与 Key 的地域一致（**北京和新加坡的 Key 不通用**）。

### 3. 开始使用

在群里发送：

```
小助手 今天天气不错，要不要一起出去走走？
```

机器人会回复一条语音。默认配置下会同时阻止该消息继续走 LLM，避免文字+语音重复回复。

---

## 配置说明

所有配置都在 AstrBot WebUI 的「插件 → 唤醒词 TTS 语音合成 → 配置」里可视化修改，
保存后**下一条消息立即生效**，无需重载插件。

### 基础与触发

| 配置项 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `enabled` | bool | `true` | 总开关，关闭后唤醒词与指令都不触发 |
| `wake_words` | list | `["小助手"]` | 唤醒词列表，可配置多个 |
| `wake_require_separator` | bool | `false` | 开启后唤醒词后必须跟分隔符，可避免「小助手们」误触发 |
| `wake_case_sensitive` | bool | `false` | 英文唤醒词是否区分大小写 |
| `strip_wake_word` | bool | `true` | 合成时是否去掉唤醒词本身 |
| `enable_command` | bool | `true` | 是否启用 `/tts` 系列指令 |
| `ignore_slash_commands` | bool | `true` | 忽略带 AstrBot 唤醒前缀的指令消息，建议保持开启 |
| `stop_event_on_wake` | bool | `true` | 触发后阻止消息继续走 LLM，避免重复回复 |
| `echo_source_text` | bool | `false` | 发送语音前是否先把文本也发一条 |
| `cooldown_seconds` | int | `0` | 同一会话触发冷却，`0` 表示不限制 |
| `max_concurrent` | int | `3` | 同时进行的合成请求上限 |

### 模型与音色

| 配置项 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `region` | string | `beijing` | 接口地域：`beijing` / `singapore` / `qwenai` / `custom` |
| `base_url` | string | 空 | 自定义地址，留空按地域自动选择 |
| `api_key` | string | 空 | 阿里云百炼 API Key（面板中掩码显示） |
| `model` | string | `qwen3-tts-flash` | TTS 模型名称，可填任意兼容模型 |
| `voice` | string | `Cherry` | 系统音色名 |
| `language_type` | string | `Auto` | 合成语种，单一语种文本建议明确指定 |
| `instructions` | text | 空 | 自然语言语气指令，**仅 instruct 系列模型支持** |
| `optimize_instructions` | bool | `false` | 由模型自动优化语气指令 |
| `timeout_seconds` | int | `60` | 单次请求超时 |
| `max_retries` | int | `2` | 失败重试次数（仅网络错误、超时、429/5xx） |

### 会话音色与文本处理

| 配置项 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `allow_session_voice_override` | bool | `true` | 允许用户用 `/tts_voice` 覆盖本会话音色 |
| `admin_only_voice_switch` | bool | `false` | 仅管理员可切换音色 |
| `max_chars` | int | `300` | 单段最大字符数，超出会自动分段 |
| `split_long_text` | bool | `true` | 长文本自动分段；关闭则直接截断 |
| `max_segments` | int | `3` | 一条消息最多合成几段语音 |
| `clean_markdown` | bool | `true` | 去掉 `**`、`` ` ``、`#` 等标记 |
| `clean_url` | bool | `true` | 移除链接，避免逐字符朗读 |
| `clean_emoji` | bool | `false` | 移除 Emoji 与特殊符号 |

### LLM 预处理（文本导演）

> 默认 **开启**。不需要时可在配置里关掉，或把「LLM 接入方式」指向一个可用的 Provider。

| 配置项 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `llm_enable` | bool | `true` | 是否在合成前调用大模型改写台词 |
| `llm_mode` | string | `astrbot_provider` | `astrbot_provider` 复用 AstrBot Provider；`direct` 插件内直连 |
| `llm_provider_id` | string | 空 | 复用方式下选择的 Provider，留空用当前会话默认对话模型 |
| `llm_model` | string | 空 | 模型名，留空用 Provider 自身配置；DeepSeek 系列或 Opencode 模型 ID |
| `llm_base_url` | string | `https://api.deepseek.com/v1` | 仅直连方式。插件会自动补 `/chat/completions` |
| `llm_api_key` | string | 空 | 仅直连方式（掩码显示） |
| `llm_auth_style` | string | `bearer` | 仅直连方式：`bearer` / `x-api-key` / `none` |
| `llm_extra_headers` | text | 空 | 仅直连方式。「名称: 值」逐行或 JSON，同名会覆盖自动鉴权头 |
| `llm_timeout_seconds` | int | `20` | LLM 超时 |
| `llm_max_retries` | int | `1` | LLM 重试次数（网络错误、超时、429/5xx） |
| `llm_temperature` | float | `0.3` | 温度，越低越稳定（仅直连方式透传） |
| `llm_max_tokens` | int | `800` | 限制输出长度，避免费用失控 |
| `llm_context_messages` | int | `6` | 取当前会话最近几条消息作为上下文，`0` 表示不用 |
| `llm_rewrite` | bool | `true` | 是否允许改写正文；关闭则只产出语气指令 |
| `llm_apply_instructions` | bool | `true` | LLM 产出的语气指令是否覆盖固定 `instructions` |
| `llm_max_input_chars` | int | `1200` | 送给大模型的最大字符数，超出截断 |
| `llm_fail_open` | bool | `true` | LLM 失败时回退为「原文直接合成」；关闭则回复错误 |
| `llm_system_prompt` | text | 空 | 自定义系统提示词，留空用内置 |
| `llm_tag_style` | string | `none` | 细粒度标签类型：`none` / `qwen-audio` / `ssml` / `custom` |
| `llm_custom_tags` | text | 空 | 标签类型选 `custom` 时，写在这里的说明会进提示词 |

> ⚠️ **标签类型务必与模型匹配**：`qwen3-tts-flash` 不支持行内标签，
> 此时必须保持 `none`。插件在 `none` 下还会主动清理模型硬塞进来的
> `<|EMO|>`、`[pause]`、`<break/>` 等标记，避免被逐字朗读。

### 缓存

| 配置项 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `cache_ttl_hours` | float | `6.0` | 音频缓存保留时长，`0` 表示不按时间清理 |
| `max_cache_files` | int | `200` | 缓存文件数量上限，超出后删除最旧的 |

### 唤醒词专属配置（`wake_profiles`）

为不同唤醒词绑定不同的模型 / 音色 / 语气。未填写的字段继承全局配置。

配置示例（等价 JSON）：

```json
"wake_profiles": [
  {
    "__template_key": "profile",
    "wake_word": "小美",
    "voice": "Serena",
    "instructions": "语气温柔亲切，语速稍慢"
  },
  {
    "__template_key": "profile",
    "wake_word": "解说员",
    "model": "qwen3-tts-instruct-flash",
    "voice": "Ethan",
    "language_type": "Chinese",
    "optimize_instructions": true
  }
]
```

**生效优先级**：`/tts_voice` 会话设置 > 唤醒词专属配置 > 全局配置。
当唤醒词存在包含关系时，**更长的唤醒词优先**（例如「小助手小美」优先于「小助手」）。

---

## 使用方法

### 唤醒词

```
小助手 今天天气不错
小助手 帮我念一段睡前故事
```

只发唤醒词不发内容时，机器人会提示用法，不会调用 TTS（避免误触发浪费额度）：

```
小助手
```

### 指令

| 指令 | 别名 | 说明 |
| --- | --- | --- |
| `/tts <文本>` | `/tts_say`、`/语音`、`/合成` | 合成指定文本；不带参数时会尝试读取被引用的消息 |
| `/tts_voice [音色名]` | `/音色` | 查看或设置当前会话音色；`reset` 恢复全局音色 |
| `/tts_llm <文本>` | `/tts_direct`、`/tts_script` | 只运行 LLM 预处理并显示改写结果，**不发送语音** |
| `/tts_status` | `/tts_info` | 查看模型、音色、接口地址、缓存与唤醒词配置（Key 已掩码） |
| `/tts_help` | `/tts_usage` | 显示使用说明 |

示例：

```
/tts 你好，世界
/tts_voice Serena
/tts_voice reset
/tts_llm 温柔一点念这段
/tts_status
```

---

## 让 LLM 先改写台词

插件默认会在调用 TTS 之前先跑一遍大模型，这一步叫「文本导演」。它解决三个问题：

| 能力 | 说明 | 例子 |
| --- | --- | --- |
| free-style 指令遵循 | 把自然语言要求转成真正的用词、标点与断句 | 「温柔一点念」→ 拆成短句、加「呀」「～」、用省略号表示迟疑 |
| 细粒度标签控制 | 按目标模型能力注入标签，词表写进提示词 | `<\|EMO\|>开心`、`<break time="300ms"/>` |
| 上下文感知 | 读取当前会话最近 N 条消息，让语气贴合氛围 | 正在哄睡 → 自动放轻放缓 |

### 数据流

```
消息（已去掉唤醒词）
      │
      ▼
┌─────────────────────────────────────┐
│ LLM 导演                            │
│  系统提示词：只改怎么说、不改说什么   │
│  用户提示词：目标模型能力 + 对话上下文 + 原文 │
│  要求输出 JSON：                     │
│  {"text","instructions","tags","reason"} │
└─────────────────────────────────────┘
      │  失败且开启兜底 → 用原文继续
      ▼
文本清洗 → 分段 → Qwen-TTS → 语音消息
```

### 方式 A：复用 AstrBot 的 Provider（默认，推荐）

在 AstrBot 的「服务提供商」里配好一个对话模型，本插件就能直接用。

这样做最关键的好处是：**AstrBot 会把该 Provider 的 `custom_headers` 应用到 HTTP 客户端**。
像 Opencode 这类需要特殊请求头的网关，只要在 AstrBot 里把 Header 配好，
插件无需任何额外配置即可调用——插件自己发请求反而容易漏掉这些头。

```
llm_mode: astrbot_provider
llm_provider_id: （留空则用当前会话默认对话模型）
llm_model: （留空则用 Provider 自己的模型）
```

### 方式 B：插件内直连

用于「不想在 AstrBot 里再注册一个 Provider」或「网关不在 AstrBot 支持列表里」的场景。

```
llm_mode: direct
llm_base_url: https://api.deepseek.com/v1
llm_api_key: sk-...
llm_auth_style: bearer          # 或 x-api-key / none
llm_extra_headers: |
  anthropic-version: 2023-06-01
  X-Title: astrbot-tts-wakeword
```

### 以 Opencode 为例

Opencode 一类的网关通常需要额外的请求头。两种接法：

**接法 A（推荐）**：在 AstrBot 中新增一个 OpenAI 兼容 Provider，把网关地址与所需 Header
填进该 Provider 的「自定义请求头（`custom_headers`）」，然后在本插件里选中它。

**接法 B**：本插件设为直连，`llm_auth_style` 按网关要求选择，
其余 Header 写进「自定义请求头」。同名 Header 会覆盖插件自动生成的鉴权头，
所以网关要求的任何特殊写法都能表达。

> 排查 401/403 时先看 `/tts_status`：它会显示接入方式、Provider、地址与
> 自定义请求头的**名称**（值不会回显），再配合 `/tts_llm` 单独验证 LLM 阶段。

### 调试技巧

```
/tts_llm 温柔一点念：宝宝该睡觉了
```

会显示原文、改写后的文本、语气指令、使用的标签，以及模型给出的调整说明，
**不会发送语音**，非常适合反复调整提示词与标签策略。

### 调优建议

- 延迟敏感：把 `llm_context_messages` 调小（或设 0）、`llm_max_tokens` 调小，
  或直接关闭 `llm_enable`。
- 保真优先：把 `llm_rewrite` 设为 `false`，只让模型给语气指令。
- 省钱：`llm_max_input_chars` 限制输入长度，`llm_fail_open` 保持开启以免报错打扰用户。
- 提示词不合口味：用 `llm_system_prompt` 覆盖内置提示词（记得仍然要求输出 JSON）。

---

## 模型与音色

### HTTP 非实时接口可用模型

以下模型已在本插件的 HTTP 通道上**实测可用**（百炼·北京地域）：

| 模型 | 说明 |
| --- | --- |
| `qwen3-tts-flash` | **默认模型**。低延迟，按字符计费，适合聊天播报 |
| `qwen3-tts-instruct-flash` | 支持 `instructions` 自然语言语气控制（语速/情绪/风格） |
| `qwen-tts` | 通义千问 TTS，按 Token 计费 |
| `qwen-tts-latest` | 指向当前最新的 `qwen-tts` |

### ⚠️ 关于 `qwen-audio-3.1-tts-flash`

`qwen-audio-3.1-tts-flash` 及整个 **Qwen-Audio-TTS 3.x 系列**（`qwen-audio-3.1-tts-next`、
`qwen-audio-3.0-tts-flash` 等）**目前只提供实时（WebSocket）接口**，
在 HTTP 非实时接口上调用会返回：

```json
{ "code": "InvalidParameter", "message": "url error, please check url！" }
```

本插件走的是 HTTP 非实时通道（**这是刻意设计**：AstrBot 每条回复只发送一条语音消息，
流式合成无法让用户更早听到完整音频，反而显著增加复杂度），因此**无法调用该系列模型**。

插件已经内置了针对这条错误的可读提示：

```
模型在所选接口地址上不可用：url error, please check url！
请确认「地域/接口地址」与 model 是否配套。HTTP 非实时接口可用模型：
qwen3-tts-flash、qwen3-tts-instruct-flash、qwen-tts。
注意：qwen-audio-3.1-tts-flash 等 Qwen-Audio-TTS 系列只提供实时（WebSocket）
接口，无法通过本插件当前的 HTTP 通道调用，请改用上述模型。
```

如果你需要这一系列模型，请改用 `qwen3-tts-flash`（音质与延迟表现接近），
或在 Issue 中说明需求，我们可以评估增加 WebSocket realtime 模式。

### 音色

系统音色名区分模型系列，`Cherry`、`Serena`、`Ethan`、`Chelsie`、`Dylan` 等
均在 `qwen3-tts-flash` 上可用。完整列表见
[Qwen-TTS 音色列表](https://platform.qianwenai.com/docs/developer-guides/speech/voice-list/qwen-tts)。

### 其他兼容平台

插件默认走百炼（DashScope）原生 HTTP 协议。任何实现同一协议的网关都可以使用：
把「接口地域」设为 `custom`，在「自定义 API 地址」里填入网关地址即可
（插件会自动补全 `/api/v1`）。

---

## 支持的消息平台

语音消息（`Record`）并非所有平台都支持，这是消息平台本身的限制：

| 平台 | 语音支持 | 说明 |
| --- | --- | --- |
| QQ 个人号（aiocqhttp） | ✅ | 推荐搭配 NapCat / Lagrange |
| Telegram | ✅ | — |
| 企业微信（wecom） | ✅ | — |
| QQ 官方接口 | ❌ | 平台不支持 `Record` 消息段 |
| 飞书 / 钉钉 | ❌ | 平台不支持 `Record` 消息段 |

在不支持语音的平台上，AstrBot 会退化为发送文件，或仅能使用文本功能。

---

## 工作原理

```
用户消息
   │
   ├─ 以唤醒词开头？ ──否──▶ 交给 AstrBot 后续流程（LLM 等）
   │         │是
   │         ▼
   │   匹配唤醒词专属配置（音色/模型/语气）
   │         │
   │         ▼
   │   ┌──────────────────────────────────────────┐
   │   │ LLM 预处理（文本导演，可关闭）              │
   │   │  Provider 复用 / 直连 OpenAI 兼容接口      │
   │   │  输入：目标模型能力 + 会话上下文 + 原文     │
   │   │  输出：{"text","instructions","tags",...} │
   │   │  失败且开启兜底 → 直接用原文继续            │
   │   └──────────────────────────────────────────┘
   │         │
   │         ▼
   │   文本清洗：去 Markdown / 链接 / Emoji / CQ 码 / 非法标签
   │         │
   │         ▼
   │   按句子切分，超过 max_chars 自动分段
   │         │
   │         ▼
   │   DashScope HTTP 合成（并发上限 + 重试退避）
   │         │
   │         ▼
   │   下载音频 → 修正 WAV 文件头 → 存入 data/plugin_data/<插件目录>/audio/
   │         │
   │         ▼
   └─▶ Comp.Record 语音消息 → 发送到当前会话
              └─ 停止事件传播，避免 LLM 重复回复
```

### 值得说明的两个实现细节

**1. 为什么修正 WAV 文件头？**

Qwen-TTS 的 HTTP 接口实际是流式返回，返回的 WAV 使用**占位文件头**。
实测一条 149,804 字节的音频，头部声明 `data` 长度为 2,147,483,547 字节，
真实负载只有 149,760 字节——`wave` 模块会把它读成 1,073,741,773 帧（约 12 小时）的音频。

宽容的播放器会忽略该字段读到文件末尾，但严格的解析器和部分 IM 客户端会计算出
错误的时长甚至拒绝播放。插件在保存音频时会把 RIFF 与 `data` 长度改写为真实值
（见 [`core/wav.py`](core/wav.py)），且仅在 `data` 块位于文件末尾且长度明显异常时改写。

**2. 为什么不做流式输出？**

AstrBot 的一条回复只能携带一条语音消息，用户必须等整段音频合成完成才能听到，
流式合成不会缩短这个时间，却需要处理 SSE 分片拼接、中间块格式等问题。
因此插件只实现非流式（单次请求返回完整音频）路径，换取更高的可靠性。

---

## 数据与隐私

- **API Key** 保存在 AstrBot 的插件配置文件中（`data/config/<插件目录>_config.json`），
  插件日志与 `/tts_status` 输出中**始终掩码显示**（如 `sk-abc****xyz9`）。
- **合成文本**会发送到你所配置的 TTS 服务（默认阿里云百炼）。请勿让机器人朗读敏感内容。
- **LLM 预处理**开启时，原始消息与当前会话最近 `llm_context_messages` 条消息会发送到
  你所配置的大模型服务。若不希望消息外发，请关闭 `llm_enable` 或把
  `llm_context_messages` 设为 `0`。
- **LLM API Key** 与 TTS Key 一样只在配置文件中保存，日志与 `/tts_status` 均掩码显示；
  自定义请求头的**值**不会回显，只显示名称。
- **音频缓存**保存在 `data/plugin_data/<插件目录>/audio/`，按 `cache_ttl_hours`
  与 `max_cache_files` 自动回收，不会写入插件自身的代码目录。
- **会话音色设置**保存在 `data/plugin_data/<插件目录>/session_voices.json`。
- 卸载插件时在 WebUI 勾选「删除插件数据」即可一并清理。

---

## 常见问题

<details>
<summary><b>报错 400：<code>url error, please check url！</code> / <code>Model not exist.</code></b></summary>

模型名称与接口地址不匹配。请检查：

1. 是否使用了 Qwen-Audio-TTS 系列（`qwen-audio-3.x-tts-*`）——该系列只有
   WebSocket 实时接口，本插件无法调用，请改用 `qwen3-tts-flash`。
2. 模型名是否拼写正确（常见错误：`qwen-tts-flash`、`qwen3-tts`）。
3. 「接口地域」与 Key 是否配套。

发送 `/tts_status` 可以查看当前实际使用的模型与接口地址。
</details>

<details>
<summary><b>报错 401：鉴权失败</b></summary>

- API Key 是否填写正确、是否已过期。
- **北京与新加坡的 Key 不通用**：北京 Key 必须选「百炼·北京」，
  新加坡 Key 必须选「百炼·新加坡」。
- 是否欠费或未开通百炼服务。
</details>

<details>
<summary><b>报错 429：调用频率超限</b></summary>

调低「最大并发合成数」，或为会话设置「触发冷却」。插件已内置指数退避重试，
偶发 429 会自动恢复。
</details>

<details>
<summary><b>报错：语音服务返回了空音频</b></summary>

通常是文本被清理后为空（例如只发了一个链接）。确认 `clean_url` 是否把内容全删掉了，
或检查文本是否触发内容安全策略。
</details>

<details>
<summary><b>群里不回复，私聊却正常</b></summary>

AstrBot 默认在群聊中需要 @机器人 或使用唤醒前缀才会响应 LLM。
本插件的唤醒词监听器不受此限制，但如果你需要机器人对普通消息也响应，
请检查 AstrBot 的唤醒设置。另外确认该消息没有以 AstrBot 唤醒前缀（默认 `/`）开头——

如果确实是以 `/` 开头（例如 `/小助手 你好`），插件会按「指令消息」处理并交给
AstrBot 的指令系统，这是为了避免劫持其他插件指令。
</details>

<details>
<summary><b>误触发：说了「小助手们」也被触发</b></summary>

把「唤醒词后必须跟分隔符」(`wake_require_separator`) 设为 `true`。
此时必须写成「小助手 你好」或「小助手，你好」。
</details>

<details>
<summary><b>回复了两条：一条文字一条语音</b></summary>

把「触发后阻止该消息继续走 LLM」(`stop_event_on_wake`) 设为 `true`（默认已开启）。
如果你打开了「同时发送文本内容」(`echo_source_text`)，那是有意为之的先文字后语音。
</details>

<details>
<summary><b>语音消息变成了文件</b></summary>

说明当前消息平台不支持 `Record` 消息段（如 QQ 官方接口、飞书、钉钉）。
请参考[支持的消息平台](#支持的消息平台)。
</details>

<details>
<summary><b>长文本只念了一半</b></summary>

一条消息默认最多合成 `max_segments`（3）段，超出部分会被丢弃并提示「文本过长」。
可以调大「最多合成段数」和「单段最大字符数」。
</details>

<details>
<summary><b>LLM 预处理报错「没有找到可用的大模型 Provider」</b></summary>

默认配置会先调用大模型。如果 AstrBot 里还没有配置对话模型，插件会**自动回退**用原文
合成语音（日志里会有 warning），不会影响基本功能。要启用改写，请：

1. 在 AstrBot「服务提供商」中添加一个对话模型；或
2. 把「LLM 接入方式」改为「插件内直连」，填写 Base URL 与 API Key。

也可以用 `/tts_llm 测试` 单独验证这一步。
</details>

<details>
<summary><b>直连方式报 401 / 403</b></summary>

网关要求的请求头没带对。按顺序检查：

1. `llm_auth_style` 是否正确（`bearer` / `x-api-key` / `none`）。
2. 网关是否需要额外 Header（`anthropic-version`、`X-Title`、组织 ID、签名头等）——
   写进「自定义请求头」，每行一个 `名称: 值`。
3. `llm_base_url` 是否需要 `/v1`，以及模型名是否属于该网关。

推荐做法：改用在 AstrBot 中配置 Provider 的方式——把 Header 填进 Provider 的
「自定义请求头」，然后本插件选它即可，AstrBot 会保证这些头被正确带上。
</details>

<details>
<summary><b>直连方式报 404</b></summary>

Base URL 不对。插件会自动补 `/chat/completions`，所以填 `https://host` 或
`https://host/v1` 都可以；但若网关路径不是标准的 `/v1/chat/completions`
（例如 `https://host/openai/v1`），请把完整前缀填进 `llm_base_url`。
</details>

<details>
<summary><b>标签被念出来了（听到「竖线 EMO 竖线」）</b></summary>

目标模型不支持行内标签。把「细粒度标签类型」设为 `none`（默认值）即可，
插件会自动清理 `<|EMO|>`、`[pause]`、`<break/>` 这类标记。
只有确认模型支持标签时（例如 Qwen-Audio-TTS / CosyVoice 系列或支持 SSML 的引擎）
才应开启对应选项。
</details>

<details>
<summary><b>延迟变高了</b></summary>

LLM 阶段会增加约 0.5~2 秒。可以：

- 把 `llm_context_messages` 调小或设为 `0`；
- 换用更快的模型（如 `llm_model: deepseek-v4-flash`）；
- 调小 `llm_max_tokens`；
- 直接关闭 `llm_enable`。
</details>

<details>
<summary><b>LLM 把内容改了 / 自己回答了问题</b></summary>

内置提示词已强制「只改怎么说、不改说什么」并且不回答问题，但模型仍可能跑偏。可以：

- 把 `llm_rewrite` 设为 `false`，只允许产出语气指令；
- 用 `llm_system_prompt` 覆盖提示词，并在其中再次强调不得改变原意；
- 用 `/tts_llm` 观察几轮输出后再决定。
</details>

<details>
<summary><b>如何让不同唤醒词用不同音色？</b></summary>

使用「唤醒词专属配置」`wake_profiles`，见[配置说明](#唤醒词专属配置wake_profiles)。
</details>

---

## 开发与测试

### 环境准备

```bash
python -m venv .venv && . .venv/bin/activate
pip install pytest pytest-asyncio aiohttp ruff
```

### 运行测试

```bash
pytest                      # 离线测试，不需要网络与 API Key
pytest -q -rs               # 显示跳过原因
```

离线测试共 **323 条**，覆盖配置解析、唤醒词匹配、文本清洗与分段、DashScope 客户端
（含重试与错误映射）、WAV 文件头修正、音频缓存回收、合成服务编排，以及：

- LLM 导演：能力描述、提示词构建、JSON/代码块/散文/异常响应解析、标签清理与
  非 instruct 模型的指令丢弃
- 直连 LLM 客户端：URL 归一化、三种鉴权方式、**自定义请求头逐项断言**、
  429 重试、401/404/网络错误提示
- 两种接入方式的端到端流程：Provider 复用与直连、上下文读取、改写结果进入 TTS、
  失败兜底开关、`/tts_llm` 与 `/tts_status`
- 插件端到端：唤醒词 / 指令 / 会话音色 / 错误提示 / 注册项

其中 `tests/astrbot_stub.py` 提供 AstrBot 的最小替身，
`tests/test_astrbot_api_conformance.py` 会**读取真实 AstrBot 源码**校验插件依赖的
符号、方法、配置类型与生命周期是否仍然存在（自动探测 `../AstrBot`，
也可用 `ASTRBOT_SOURCE=/path/to/AstrBot` 指定）：

```bash
ASTRBOT_SOURCE=/path/to/AstrBot pytest tests/test_astrbot_api_conformance.py -v
```

### 真机联调测试（可选，会消耗额度）

```bash
DASHSCOPE_API_KEY=sk-xxx pytest tests/test_live_dashscope.py -v -s
```

这 5 条真机测试包含：模型可用性与音频可播放性、WAV 文件头修正、`/tts_status` 掩码、
**真实大模型的导演 JSON 契约**，以及「唤醒词 → 真实 LLM → 真实 TTS → 音频文件」全链路。

可选环境变量：`DASHSCOPE_REGION`（`beijing`/`singapore`）、
`DASHSCOPE_TTS_MODEL`、`DASHSCOPE_TTS_VOICE`、`LIVE_LLM_MODEL`、`LIVE_LLM_BASE_URL`。

### 代码检查

```bash
ruff check .
ruff format --check .
```

### 本地调试插件

```bash
ln -s "$(pwd)" /path/to/AstrBot/data/plugins/astrbot_plugin_tts_wakeword
```

修改代码后在 WebUI「插件」页面点击刷新图标重载，或发送 `/tts_status` 查看生效配置。
需注意：**配置项**每次消息都会重新读取，改配置无需重载；**代码**改动需要重载插件。

---

## 目录结构

```
.
├── main.py                     # 插件入口：AstrBot 事件处理与消息发送
├── metadata.yaml               # 插件元数据（名称、版本、作者、仓库）
├── _conf_schema.json           # WebUI 可视化配置 Schema
├── requirements.txt            # 运行依赖（仅 aiohttp）
├── logo.png                    # 插件 Logo（插件市场卡片）
├── core/                       # 与 AstrBot 解耦的核心逻辑，可独立单测
│   ├── config.py               # 配置解析、校验、唤醒词专属配置、LLM 配置
│   ├── text_utils.py           # 唤醒词匹配、文本清洗、长文本分段
│   ├── dashscope_client.py     # DashScope Qwen-TTS HTTP 客户端（重试/退避）
│   ├── director.py             # LLM 导演：能力描述、提示词、响应解析、标签清理
│   ├── llm_client.py           # OpenAI 兼容 LLM 客户端（鉴权方式 + 自定义请求头）
│   ├── audio_store.py          # 音频落盘、格式识别、缓存回收
│   ├── wav.py                  # WAV 文件头修正与时长计算
│   ├── service.py              # 合成流程编排、并发控制、导演阶段
│   └── errors.py               # 带用户提示的错误类型
├── tests/                      # pytest 测试（离线 + AstrBot 源码一致性 + 真机可选）
│   ├── astrbot_stub.py         # AstrBot 最小替身（含 Provider / 会话历史）
│   ├── mock_dashscope.py       # 假 DashScope TTS 服务
│   ├── mock_llm.py             # 假 OpenAI 兼容 LLM 服务
│   ├── test_director.py        # 提示词与响应解析
│   ├── test_llm_client.py      # 请求头 / 重试 / 错误映射
│   └── test_llm_stage.py       # LLM 阶段端到端流程
├── tools/make_logo.py          # 生成 logo.png 的辅助脚本（开发用）
├── pyproject.toml              # ruff / pytest 配置（非打包配置）
├── README.md
├── CHANGELOG.md
└── LICENSE
```

---

## 许可与致谢

本项目基于 [MIT License](LICENSE) 开源。

设计与实现参考了以下文档与项目（按 AstrBot 插件开发原则在此声明来源）：

- [AstrBot 插件开发文档](https://docs.astrbot.app/dev/star/plugin-new.html)：
  插件结构、`filter` 装饰器、消息链与 `Record`、插件配置 Schema、插件数据目录规范
- [Qwen-TTS 非实时语音合成 API 参考](https://help.aliyun.com/zh/model-studio/qwen-tts-api)：
  请求/响应结构、`instructions` 参数、错误码语义
- [音频生成 API 参考](https://help.aliyun.com/zh/model-studio/audio-generation-api)：
  确认 `qwen-audio-3.1-tts-next` 使用的独立端点与 `text_prompt` 参数形态
- [Qwen-Audio-TTS WebSocket API](https://platform.qianwenai.com/docs/api-reference/speech-synthesis/cosyvoice/websocket-api)：
  确认 Qwen-Audio-TTS 系列仅提供实时接口，这是本插件默认模型的选型依据
- [AstrBot 调用 AI 指南](https://docs.astrbot.app/dev/star/guides/ai.html)：
  `Provider.text_chat` 的 `prompt` / `contexts` / `system_prompt` / `model` 用法
- AstrBot 源码 `astrbot/core/provider/sources/openai_source.py` 与
  `anthropic_source.py`：确认 Provider 的 `custom_headers` 会作为 HTTP 客户端
  `default_headers` 生效，这是「Opencode 等网关无需插件额外适配」的实现依据

未复制任何第三方代码；`core/` 全部为本项目原创实现。
