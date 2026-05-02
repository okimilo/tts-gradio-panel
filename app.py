import os
import base64
import mimetypes
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import gradio as gr


# ============================================================
# 环境变量配置
# ============================================================

API_BASE = os.getenv("TTS_API_BASE", "").rstrip("/")
API_KEY = os.getenv("TTS_API_KEY", "")

# NewAPI 一般用 Bearer；如果你直连小米官方，也可以设置 AUTH_MODE=api-key
AUTH_MODE = os.getenv("AUTH_MODE", "bearer").lower()

MIMO_TTS_MODEL = os.getenv("MIMO_TTS_MODEL", "mimo-v2.5-tts")
MIMO_VOICE_DESIGN_MODEL = os.getenv("MIMO_VOICE_DESIGN_MODEL", "mimo-v2.5-tts-voicedesign")
MIMO_VOICE_CLONE_MODEL = os.getenv("MIMO_VOICE_CLONE_MODEL", "mimo-v2.5-tts-voiceclone")

DEFAULT_VOICE = os.getenv("MIMO_DEFAULT_VOICE", "mimo_default")
DEFAULT_AUDIO_FORMAT = os.getenv("MIMO_AUDIO_FORMAT", "wav")

REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "300"))
MAX_CLONE_AUDIO_MB = int(os.getenv("MAX_CLONE_AUDIO_MB", "10"))

APP_USERNAME = os.getenv("APP_USERNAME", "")
APP_PASSWORD = os.getenv("APP_PASSWORD", "")

CHAT_COMPLETIONS_PATH = os.getenv("CHAT_COMPLETIONS_PATH", "/chat/completions")


# ============================================================
# 常量
# ============================================================

PRESET_VOICES = [
    "mimo_default",
    "冰糖",
    "茉莉",
    "苏打",
    "白桦",
    "Mia",
    "Chloe",
    "Milo",
    "Dean",
]

AUDIO_FORMATS = ["wav", "mp3", "pcm16"]


# ============================================================
# 基础工具函数
# ============================================================

def require_config() -> None:
    if not API_BASE:
        raise gr.Error("缺少环境变量 TTS_API_BASE，例如：https://你的-newapi-域名/v1")
    if not API_KEY:
        raise gr.Error("缺少环境变量 TTS_API_KEY")


def build_url(path: str) -> str:
    if path.startswith("http://") or path.startswith("https://"):
        return path
    if not path.startswith("/"):
        path = "/" + path
    return API_BASE + path


def make_headers() -> Dict[str, str]:
    headers = {
        "Content-Type": "application/json",
    }

    if AUTH_MODE == "api-key":
        headers["api-key"] = API_KEY
    else:
        headers["Authorization"] = f"Bearer {API_KEY}"

    return headers


def save_bytes_to_temp_file(data: bytes, suffix: str = ".wav") -> str:
    f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.write(data)
    f.close()
    return f.name


def normalize_base64_audio(audio_b64: str) -> bytes:
    """
    兼容：
    1. 纯 base64
    2. data:audio/wav;base64,xxxx
    """
    if "," in audio_b64 and audio_b64.strip().startswith("data:"):
        audio_b64 = audio_b64.split(",", 1)[1]

    return base64.b64decode(audio_b64)


def save_audio_base64(audio_b64: str, audio_format: str = "wav") -> str:
    suffix = f".{audio_format}" if audio_format != "pcm16" else ".pcm"
    audio_bytes = normalize_base64_audio(audio_b64)
    return save_bytes_to_temp_file(audio_bytes, suffix=suffix)


def download_audio_url(url: str, audio_format: str = "wav") -> str:
    r = requests.get(url, timeout=REQUEST_TIMEOUT)
    if r.status_code >= 400:
        raise gr.Error(f"下载音频失败：HTTP {r.status_code}\n{r.text[:1000]}")

    suffix = f".{audio_format}" if audio_format != "pcm16" else ".pcm"
    return save_bytes_to_temp_file(r.content, suffix=suffix)


def get_nested(data: Any, path: str) -> Any:
    current = data
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list):
            try:
                current = current[int(part)]
            except Exception:
                return None
        else:
            return None
    return current


def extract_audio_from_response(resp_json: Dict[str, Any], audio_format: str = "wav") -> str:
    """
    优先兼容 OpenAI/NewAPI 风格：
    choices[0].message.audio.data

    也兼容一些常见变体：
    choices[0].message.audio.url
    choices[0].delta.audio.data
    audio.data
    data.audio
    """
    possible_data_paths = [
        "choices.0.message.audio.data",
        "choices.0.delta.audio.data",
        "message.audio.data",
        "audio.data",
        "data.audio.data",
        "data.audio",
    ]

    possible_url_paths = [
        "choices.0.message.audio.url",
        "choices.0.delta.audio.url",
        "message.audio.url",
        "audio.url",
        "data.audio.url",
        "data.url",
        "url",
    ]

    for path in possible_data_paths:
        value = get_nested(resp_json, path)
        if value:
            return save_audio_base64(str(value), audio_format=audio_format)

    for path in possible_url_paths:
        value = get_nested(resp_json, path)
        if value:
            return download_audio_url(str(value), audio_format=audio_format)

    raise gr.Error(
        "没有在接口返回里找到音频数据。\n\n"
        "已尝试读取 choices[0].message.audio.data / audio.url 等字段。\n\n"
        f"接口返回预览：\n{str(resp_json)[:2000]}"
    )


def guess_mime_type(file_path: str) -> str:
    mime, _ = mimetypes.guess_type(file_path)

    if mime in ["audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav"]:
        if mime == "audio/x-wav":
            return "audio/wav"
        return mime

    ext = Path(file_path).suffix.lower()
    if ext == ".mp3":
        return "audio/mpeg"
    if ext == ".wav":
        return "audio/wav"

    # 小米文档目前主要支持 mp3 / wav
    return "audio/wav"


def file_to_data_url(file_path: str) -> str:
    size_mb = os.path.getsize(file_path) / 1024 / 1024
    if size_mb > MAX_CLONE_AUDIO_MB:
        raise gr.Error(
            f"参考音频过大：{size_mb:.2f} MB。\n"
            f"当前限制为 {MAX_CLONE_AUDIO_MB} MB。请上传更短的 mp3/wav。"
        )

    mime = guess_mime_type(file_path)

    with open(file_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime};base64,{audio_b64}"


def sanitize_payload_for_error(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    报错时避免把完整 base64 音频打印出来。
    """
    safe = dict(payload)

    audio = safe.get("audio")
    if isinstance(audio, dict) and isinstance(audio.get("voice"), str):
        voice = audio["voice"]
        if voice.startswith("data:audio/"):
            audio = dict(audio)
            audio["voice"] = voice[:80] + "...[base64 omitted]"
            safe["audio"] = audio

    return safe


# ============================================================
# MiMo Chat Completions 调用
# ============================================================

def call_mimo_chat_completions(
    model: str,
    messages: List[Dict[str, Any]],
    audio_format: str = "wav",
    voice: Optional[str] = None,
) -> str:
    """
    三种模型的关键差异：

    1. mimo-v2.5-tts:
       audio = {"format": "wav", "voice": "mimo_default"}

    2. mimo-v2.5-tts-voicedesign:
       audio = {"format": "wav"}
       不能传 audio.voice

    3. mimo-v2.5-tts-voiceclone:
       audio = {"format": "wav", "voice": "data:audio/mpeg;base64,xxx"}
    """
    require_config()

    audio_payload: Dict[str, Any] = {
        "format": audio_format,
    }

    if voice:
        audio_payload["voice"] = voice

    payload = {
        "model": model,
        "messages": messages,
        "audio": audio_payload,
    }

    r = requests.post(
        build_url(CHAT_COMPLETIONS_PATH),
        headers=make_headers(),
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )

    if r.status_code >= 400:
        safe_payload = sanitize_payload_for_error(payload)
        raise gr.Error(
            f"请求失败：HTTP {r.status_code}\n\n"
            f"接口返回：\n{r.text[:2000]}\n\n"
            f"请求体预览：\n{safe_payload}"
        )

    try:
        resp_json = r.json()
    except Exception:
        raise gr.Error(f"接口返回不是 JSON：\n{r.text[:2000]}")

    return extract_audio_from_response(resp_json, audio_format=audio_format)


# ============================================================
# 三个功能：普通 TTS / Voice Design / Voice Clone
# ============================================================

def normal_tts(
    instruction: str,
    text: str,
    voice: str,
    audio_format: str,
) -> Tuple[str, str]:
    instruction = (instruction or "").strip()
    text = (text or "").strip()
    voice = (voice or "").strip() or DEFAULT_VOICE
    audio_format = audio_format or DEFAULT_AUDIO_FORMAT

    if not text:
        raise gr.Error("请输入朗读文本。")

    messages: List[Dict[str, Any]] = []

    # user 消息是风格/语气指令，可选，不会被合成进音频
    if instruction:
        messages.append({
            "role": "user",
            "content": instruction,
        })

    # 合成目标文本必须放 assistant
    messages.append({
        "role": "assistant",
        "content": text,
    })

    audio_path = call_mimo_chat_completions(
        model=MIMO_TTS_MODEL,
        messages=messages,
        audio_format=audio_format,
        voice=voice,
    )

    status = (
        "✅ 普通 TTS 生成成功\n\n"
        f"- 模型：`{MIMO_TTS_MODEL}`\n"
        f"- 音色：`{voice}`\n"
        f"- 格式：`{audio_format}`"
    )

    return status, audio_path


def voice_design_tts(
    voice_prompt: str,
    text: str,
    audio_format: str,
) -> Tuple[str, str]:
    voice_prompt = (voice_prompt or "").strip()
    text = (text or "").strip()
    audio_format = audio_format or DEFAULT_AUDIO_FORMAT

    if not voice_prompt:
        raise gr.Error("请输入声音设计描述。")

    if not text:
        raise gr.Error("请输入试听文本。")

    messages = [
        # VoiceDesign 必须把音色描述放在 user 里
        {
            "role": "user",
            "content": voice_prompt,
        },
        # 目标合成文本放 assistant
        {
            "role": "assistant",
            "content": text,
        },
    ]

    # 重点：VoiceDesign 不能传 audio.voice
    audio_path = call_mimo_chat_completions(
        model=MIMO_VOICE_DESIGN_MODEL,
        messages=messages,
        audio_format=audio_format,
        voice=None,
    )

    status = (
        "✅ Voice Design 生成成功\n\n"
        f"- 模型：`{MIMO_VOICE_DESIGN_MODEL}`\n"
        f"- 格式：`{audio_format}`\n\n"
        "注意：Voice Design 不使用预置音色，声音由你的描述决定。"
    )

    return status, audio_path


def voice_clone_tts(
    reference_audio: str,
    instruction: str,
    text: str,
    audio_format: str,
) -> Tuple[str, str]:
    if not reference_audio:
        raise gr.Error("请上传参考音频。")

    instruction = (instruction or "").strip()
    text = (text or "").strip()
    audio_format = audio_format or DEFAULT_AUDIO_FORMAT

    if not text:
        raise gr.Error("请输入试听文本。")

    reference_voice_data_url = file_to_data_url(reference_audio)

    messages = [
        # user 可放风格指令，也可以为空字符串
        {
            "role": "user",
            "content": instruction or "",
        },
        # 目标合成文本放 assistant
        {
            "role": "assistant",
            "content": text,
        },
    ]

    # 重点：VoiceClone 的参考音频放 audio.voice
    audio_path = call_mimo_chat_completions(
        model=MIMO_VOICE_CLONE_MODEL,
        messages=messages,
        audio_format=audio_format,
        voice=reference_voice_data_url,
    )

    status = (
        "✅ Voice Clone 生成成功\n\n"
        f"- 模型：`{MIMO_VOICE_CLONE_MODEL}`\n"
        f"- 格式：`{audio_format}`\n"
        f"- 参考音频大小限制：`{MAX_CLONE_AUDIO_MB} MB`\n\n"
        "注意：请只克隆你自己或已获得明确授权的声音。"
    )

    return status, audio_path


# ============================================================
# 示例提示词
# ============================================================

NORMAL_STYLE_EXAMPLES = [
    "请用自然、清晰、适合短视频旁白的语气朗读，语速中等，情绪稳定。",
    "请用温柔但疲惫的语气朗读，声音轻一点，像深夜电台旁白。",
    "请用活泼、年轻、带一点惊喜感的语气朗读，适合短视频开头。",
    "请用低沉、稳重、有纪录片感的中文男声朗读，语速稍慢。",
]

VOICE_DESIGN_EXAMPLES = [
    "一位年轻女性，说标准普通话，声音温柔清晰，有亲和力，适合短剧旁白。",
    "五十多岁的中年男性，嗓音低沉、沙哑、有岁月感，像一位老故事讲述者。",
    "年轻男声，干净明亮，语速稍快，适合科技类短视频解说。",
    "冰冷、慵懒却极具威压的低音御姐，语速较慢，咬字清晰，有距离感。",
    "深夜电台女主播，声音轻柔、贴耳、安静，语速很慢，带一点呼吸感。",
]


def pick_example(example: str) -> str:
    return example


# ============================================================
# Gradio 页面
# ============================================================

custom_css = """
footer {visibility: hidden}
.gradio-container {max-width: 1120px !important}
"""

with gr.Blocks(
    title="MiMo V2.5 TTS 在线面板",
    css=custom_css,
) as demo:
    gr.Markdown(
        """
# 🎙️ MiMo V2.5 TTS 在线面板

通过 NewAPI 调用小米 MiMo-V2.5-TTS 系列，实现：

- **普通 TTS**：使用预置音色直接朗读
- **Voice Design**：输入声音描述，直接生成试听
- **Voice Clone**：上传参考音频，直接克隆试听

生成后会在页面里直接出现播放器，不需要手动下载文件。
"""
    )

    with gr.Accordion("后端配置检查", open=False):
        gr.Markdown(
            f"""
- `TTS_API_BASE`: `{API_BASE or "未配置"}`
- `AUTH_MODE`: `{AUTH_MODE}`
- `CHAT_COMPLETIONS_PATH`: `{CHAT_COMPLETIONS_PATH}`
- `MIMO_TTS_MODEL`: `{MIMO_TTS_MODEL}`
- `MIMO_VOICE_DESIGN_MODEL`: `{MIMO_VOICE_DESIGN_MODEL}`
- `MIMO_VOICE_CLONE_MODEL`: `{MIMO_VOICE_CLONE_MODEL}`
- `DEFAULT_VOICE`: `{DEFAULT_VOICE}`
- `DEFAULT_AUDIO_FORMAT`: `{DEFAULT_AUDIO_FORMAT}`
- `MAX_CLONE_AUDIO_MB`: `{MAX_CLONE_AUDIO_MB}`
"""
        )

    with gr.Tab("普通 TTS"):
        gr.Markdown(
            """
## 普通 TTS：预置音色朗读

适合快速试听内置音色。  
目标朗读文本放在「朗读文本」中；风格控制可选。
"""
        )

        with gr.Row():
            normal_voice = gr.Dropdown(
                label="预置音色",
                choices=PRESET_VOICES,
                value=DEFAULT_VOICE if DEFAULT_VOICE in PRESET_VOICES else "mimo_default",
                allow_custom_value=True,
            )
            normal_format = gr.Dropdown(
                label="音频格式",
                choices=AUDIO_FORMATS,
                value=DEFAULT_AUDIO_FORMAT if DEFAULT_AUDIO_FORMAT in AUDIO_FORMATS else "wav",
            )

        normal_instruction = gr.Textbox(
            label="风格控制，可选。内容不会被朗读，只影响语气/情绪/节奏",
            value="请用自然、清晰、适合短视频旁白的语气朗读，语速中等，情绪稳定。",
            lines=4,
        )

        with gr.Row():
            normal_example = gr.Dropdown(
                label="风格示例",
                choices=NORMAL_STYLE_EXAMPLES,
                value=NORMAL_STYLE_EXAMPLES[0],
            )
            normal_example_btn = gr.Button("填入风格示例")

        normal_text = gr.Textbox(
            label="朗读文本",
            value="你好，这是 MiMo V2.5 普通 TTS 的在线试听。你可以切换不同预置音色，比较声音效果。",
            lines=7,
        )

        normal_btn = gr.Button("生成并试听", variant="primary")
        normal_status = gr.Markdown()
        normal_audio = gr.Audio(label="试听结果", type="filepath", autoplay=True)

        normal_example_btn.click(
            fn=pick_example,
            inputs=[normal_example],
            outputs=[normal_instruction],
        )

        normal_btn.click(
            fn=normal_tts,
            inputs=[
                normal_instruction,
                normal_text,
                normal_voice,
                normal_format,
            ],
            outputs=[
                normal_status,
                normal_audio,
            ],
        )

    with gr.Tab("Voice Design 声音设计"):
        gr.Markdown(
            """
## Voice Design：用文字描述生成声音

这里不要选择预置音色。  
你只需要描述想要的声音，模型会根据描述直接生成音频。
"""
        )

        design_prompt = gr.Textbox(
            label="声音设计描述",
            value="一位年轻女性，说标准普通话，声音温柔清晰，有亲和力，适合短剧旁白。",
            lines=6,
        )

        with gr.Row():
            design_example = gr.Dropdown(
                label="声音设计示例",
                choices=VOICE_DESIGN_EXAMPLES,
                value=VOICE_DESIGN_EXAMPLES[0],
            )
            design_example_btn = gr.Button("填入声音示例")

        design_text = gr.Textbox(
            label="试听文本",
            value="你好，这是我根据文字描述设计出来的声音。你觉得这个音色适合做短剧旁白吗？",
            lines=7,
        )

        design_format = gr.Dropdown(
            label="音频格式",
            choices=AUDIO_FORMATS,
            value=DEFAULT_AUDIO_FORMAT if DEFAULT_AUDIO_FORMAT in AUDIO_FORMATS else "wav",
        )

        design_btn = gr.Button("设计声音并试听", variant="primary")
        design_status = gr.Markdown()
        design_audio = gr.Audio(label="试听结果", type="filepath", autoplay=True)

        design_example_btn.click(
            fn=pick_example,
            inputs=[design_example],
            outputs=[design_prompt],
        )

        design_btn.click(
            fn=voice_design_tts,
            inputs=[
                design_prompt,
                design_text,
                design_format,
            ],
            outputs=[
                design_status,
                design_audio,
            ],
        )

    with gr.Tab("Voice Clone 声音克隆"):
        gr.Markdown(
            """
## Voice Clone：上传参考音频克隆声音

建议上传 **10–60 秒** 的清晰人声，尽量少背景音乐、少噪音。  
目前推荐 mp3 / wav。参考音频会被转成 base64 后放入 `audio.voice`。
"""
        )

        clone_reference_audio = gr.Audio(
            label="上传参考音频，建议 mp3/wav，清晰单人声",
            type="filepath",
        )

        clone_instruction = gr.Textbox(
            label="风格控制，可选。内容不会被朗读，只影响语气/情绪/节奏",
            value="请保持参考音频中的说话人音色，用自然、清晰的中文朗读。",
            lines=4,
        )

        clone_text = gr.Textbox(
            label="试听文本",
            value="你好，这是根据参考音频克隆出来的声音，现在正在进行在线试听。",
            lines=7,
        )

        clone_format = gr.Dropdown(
            label="音频格式",
            choices=AUDIO_FORMATS,
            value=DEFAULT_AUDIO_FORMAT if DEFAULT_AUDIO_FORMAT in AUDIO_FORMATS else "wav",
        )

        clone_btn = gr.Button("克隆声音并试听", variant="primary")
        clone_status = gr.Markdown()
        clone_audio = gr.Audio(label="试听结果", type="filepath", autoplay=True)

        clone_btn.click(
            fn=voice_clone_tts,
            inputs=[
                clone_reference_audio,
                clone_instruction,
                clone_text,
                clone_format,
            ],
            outputs=[
                clone_status,
                clone_audio,
            ],
        )

    with gr.Accordion("使用说明与避坑", open=False):
        gr.Markdown(
            """
### 三个模型怎么用

| 功能 | 模型 | 关键规则 |
|---|---|---|
| 普通 TTS | `mimo-v2.5-tts` | `audio.voice` 填预置音色 |
| Voice Design | `mimo-v2.5-tts-voicedesign` | 不传 `audio.voice`，声音描述放 `user.content` |
| Voice Clone | `mimo-v2.5-tts-voiceclone` | 参考音频转 base64 data URL 后放 `audio.voice` |

### 常见报错

`audio.voice is not supported for voice design model`  
说明 Voice Design 请求里错误传入了 `audio.voice`。这个版本已经修复。

`Param Incorrect`  
通常是模型名、请求格式、NewAPI 渠道映射或上游参数不匹配。

`没有在接口返回里找到音频数据`  
说明 NewAPI 返回格式不是标准 `choices[0].message.audio.data`，需要看返回内容再适配。

### 安全提醒

声音克隆请只使用你自己或已获得明确授权的声音。
"""
        )


if APP_USERNAME and APP_PASSWORD:
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        auth=(APP_USERNAME, APP_PASSWORD),
    )
else:
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
    )
