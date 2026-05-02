import os
import json
import base64
import tempfile
from typing import Any, Optional, Tuple

import requests
import gradio as gr


# =========================
# 基础配置
# =========================

API_BASE = os.getenv("TTS_API_BASE", "").rstrip("/")
API_KEY = os.getenv("TTS_API_KEY", "")

TTS_PATH = os.getenv("TTS_PATH", "/audio/speech")
VOICE_DESIGN_PATH = os.getenv("VOICE_DESIGN_PATH", "/voice/design")
VOICE_CLONE_PATH = os.getenv("VOICE_CLONE_PATH", "/voice/clone")

DEFAULT_TTS_MODEL = os.getenv("TTS_MODEL", "tts-1")
DEFAULT_TTS_VOICE = os.getenv("TTS_VOICE", "alloy")

DEFAULT_DESIGN_MODEL = os.getenv("VOICE_DESIGN_MODEL", "tts-voicedesign")
DEFAULT_CLONE_MODEL = os.getenv("VOICE_CLONE_MODEL", "tts-voiceclone")

VOICE_ID_JSON_PATH = os.getenv("VOICE_ID_JSON_PATH", "voice_id")
AUDIO_URL_JSON_PATH = os.getenv("AUDIO_URL_JSON_PATH", "audio_url")
AUDIO_BASE64_JSON_PATH = os.getenv("AUDIO_BASE64_JSON_PATH", "audio_base64")

CLONE_FILE_FIELD = os.getenv("CLONE_FILE_FIELD", "file")

REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "300"))

APP_USERNAME = os.getenv("APP_USERNAME", "")
APP_PASSWORD = os.getenv("APP_PASSWORD", "")


# =========================
# 工具函数
# =========================

def require_config():
    if not API_BASE:
        raise gr.Error("缺少环境变量 TTS_API_BASE，例如：https://api.example.com/v1")
    if not API_KEY:
        raise gr.Error("缺少环境变量 TTS_API_KEY")


def build_url(path: str) -> str:
    path = path.strip()
    if path.startswith("http://") or path.startswith("https://"):
        return path
    if not path.startswith("/"):
        path = "/" + path
    return API_BASE + path


def auth_headers(extra: Optional[dict] = None) -> dict:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
    }
    if extra:
        headers.update(extra)
    return headers


def get_nested_value(data: Any, path: str) -> Any:
    """
    支持：
    voice_id
    data.voice_id
    result.voice.id
    """
    if not path:
        return None

    current = data
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None

    return current


def save_bytes_to_file(content: bytes, suffix: str = ".mp3") -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(content)
    tmp.close()
    return tmp.name


def guess_audio_suffix(content_type: str) -> str:
    content_type = content_type.lower()

    if "wav" in content_type:
        return ".wav"
    if "mpeg" in content_type or "mp3" in content_type:
        return ".mp3"
    if "ogg" in content_type:
        return ".ogg"
    if "flac" in content_type:
        return ".flac"
    if "aac" in content_type:
        return ".aac"

    return ".mp3"


def handle_audio_or_json_response(resp: requests.Response) -> Tuple[Optional[str], Optional[dict]]:
    """
    返回：
    audio_path, json_data

    如果接口直接返回音频，则 audio_path 有值。
    如果接口返回 JSON，则 json_data 有值。
    """
    content_type = resp.headers.get("content-type", "").lower()

    if resp.status_code >= 400:
        text = resp.text[:1000]
        raise gr.Error(f"API 请求失败：HTTP {resp.status_code}\n{text}")

    if "audio/" in content_type or "application/octet-stream" in content_type:
        suffix = guess_audio_suffix(content_type)
        return save_bytes_to_file(resp.content, suffix=suffix), None

    try:
        data = resp.json()
        return None, data
    except Exception:
        # 有些服务商不带正确 content-type，但实际返回的是音频
        if resp.content:
            return save_bytes_to_file(resp.content, suffix=".mp3"), None
        raise gr.Error("接口返回内容无法识别，不是音频，也不是 JSON。")


def download_audio_url(audio_url: str) -> str:
    r = requests.get(audio_url, timeout=REQUEST_TIMEOUT)
    if r.status_code >= 400:
        raise gr.Error(f"下载音频失败：HTTP {r.status_code}\n{r.text[:500]}")

    suffix = guess_audio_suffix(r.headers.get("content-type", ""))
    return save_bytes_to_file(r.content, suffix=suffix)


def save_base64_audio(audio_base64: str) -> str:
    if "," in audio_base64:
        audio_base64 = audio_base64.split(",", 1)[1]

    content = base64.b64decode(audio_base64)
    return save_bytes_to_file(content, suffix=".mp3")


def extract_audio_from_json(data: dict) -> Optional[str]:
    audio_url = get_nested_value(data, AUDIO_URL_JSON_PATH)
    if audio_url:
        return download_audio_url(str(audio_url))

    audio_base64 = get_nested_value(data, AUDIO_BASE64_JSON_PATH)
    if audio_base64:
        return save_base64_audio(str(audio_base64))

    return None


def extract_voice_id(data: dict) -> Optional[str]:
    value = get_nested_value(data, VOICE_ID_JSON_PATH)
    if value:
        return str(value)

    # 兼容一些常见返回格式
    possible_paths = [
        "id",
        "voice.id",
        "data.id",
        "data.voice_id",
        "result.voice_id",
        "result.id",
        "voice_id",
    ]

    for path in possible_paths:
        value = get_nested_value(data, path)
        if value:
            return str(value)

    return None


# =========================
# 普通 TTS
# =========================

def generate_tts(
    text: str,
    model: str,
    voice: str,
    speed: float,
    response_format: str,
) -> str:
    require_config()

    text = text.strip()
    if not text:
        raise gr.Error("请输入要朗读的文字。")

    model = model.strip() or DEFAULT_TTS_MODEL
    voice = voice.strip() or DEFAULT_TTS_VOICE
    response_format = response_format or "mp3"

    payload = {
        "model": model,
        "input": text,
        "voice": voice,
        "response_format": response_format,
    }

    if speed:
        payload["speed"] = speed

    resp = requests.post(
        build_url(TTS_PATH),
        headers=auth_headers({"Content-Type": "application/json"}),
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )

    audio_path, json_data = handle_audio_or_json_response(resp)

    if audio_path:
        return audio_path

    if json_data:
        audio_from_json = extract_audio_from_json(json_data)
        if audio_from_json:
            return audio_from_json

        raise gr.Error(
            "普通 TTS 接口返回了 JSON，但没有找到音频。\n"
            f"返回内容：{json.dumps(json_data, ensure_ascii=False)[:1000]}"
        )

    raise gr.Error("普通 TTS 生成失败。")


def ui_tts(text, model, voice, speed, response_format):
    audio = generate_tts(text, model, voice, speed, response_format)
    status = f"✅ 普通 TTS 生成成功。\n\n使用音色：`{voice}`"
    return status, audio


# =========================
# Voice Design
# =========================

def design_voice_and_preview(
    prompt: str,
    preview_text: str,
    design_model: str,
    tts_model: str,
    speed: float,
    response_format: str,
) -> Tuple[str, str, str]:
    require_config()

    prompt = prompt.strip()
    preview_text = preview_text.strip()

    if not prompt:
        raise gr.Error("请输入声音描述。")

    if not preview_text:
        raise gr.Error("请输入试听文本。")

    design_model = design_model.strip() or DEFAULT_DESIGN_MODEL
    tts_model = tts_model.strip() or DEFAULT_TTS_MODEL

    payload = {
        "model": design_model,
        "prompt": prompt,
    }

    resp = requests.post(
        build_url(VOICE_DESIGN_PATH),
        headers=auth_headers({"Content-Type": "application/json"}),
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )

    audio_path, json_data = handle_audio_or_json_response(resp)

    # 情况 1：Voice Design 接口直接返回音频
    if audio_path:
        status = "✅ Voice Design 接口直接返回了音频，已生成试听。"
        return status, audio_path, ""

    if not json_data:
        raise gr.Error("Voice Design 没有返回可识别结果。")

    # 情况 2：返回 JSON 里已经有音频
    audio_from_json = extract_audio_from_json(json_data)
    if audio_from_json:
        voice_id = extract_voice_id(json_data) or ""
        status = "✅ Voice Design 返回了音频，已生成试听。"
        if voice_id:
            status += f"\n\n生成的 voice_id：`{voice_id}`"
        return status, audio_from_json, voice_id

    # 情况 3：返回 voice_id，再调用普通 TTS 试听
    voice_id = extract_voice_id(json_data)
    if not voice_id:
        raise gr.Error(
            "Voice Design 返回了 JSON，但没有找到 voice_id。\n\n"
            f"请检查 VOICE_ID_JSON_PATH 环境变量。\n\n"
            f"当前返回：{json.dumps(json_data, ensure_ascii=False)[:1200]}"
        )

    audio = generate_tts(
        text=preview_text,
        model=tts_model,
        voice=voice_id,
        speed=speed,
        response_format=response_format,
    )

    status = (
        "✅ 声音设计成功，并已生成试听。\n\n"
        f"生成的 voice_id：`{voice_id}`"
    )

    return status, audio, voice_id


# =========================
# Voice Clone
# =========================

def clone_voice_and_preview(
    audio_file: str,
    preview_text: str,
    clone_model: str,
    voice_name: str,
    tts_model: str,
    speed: float,
    response_format: str,
) -> Tuple[str, str, str]:
    require_config()

    if not audio_file:
        raise gr.Error("请先上传一段参考音频。")

    preview_text = preview_text.strip()
    if not preview_text:
        raise gr.Error("请输入试听文本。")

    clone_model = clone_model.strip() or DEFAULT_CLONE_MODEL
    tts_model = tts_model.strip() or DEFAULT_TTS_MODEL
    voice_name = voice_name.strip() or "cloned_voice"

    with open(audio_file, "rb") as f:
        files = {
            CLONE_FILE_FIELD: (
                os.path.basename(audio_file),
                f,
                "application/octet-stream",
            )
        }

        data = {
            "model": clone_model,
            "name": voice_name,
        }

        resp = requests.post(
            build_url(VOICE_CLONE_PATH),
            headers=auth_headers(),
            files=files,
            data=data,
            timeout=REQUEST_TIMEOUT,
        )

    audio_path, json_data = handle_audio_or_json_response(resp)

    # 情况 1：Voice Clone 接口直接返回音频
    if audio_path:
        status = "✅ Voice Clone 接口直接返回了音频，已生成试听。"
        return status, audio_path, ""

    if not json_data:
        raise gr.Error("Voice Clone 没有返回可识别结果。")

    # 情况 2：返回 JSON 里已经有音频
    audio_from_json = extract_audio_from_json(json_data)
    if audio_from_json:
        voice_id = extract_voice_id(json_data) or ""
        status = "✅ Voice Clone 返回了音频，已生成试听。"
        if voice_id:
            status += f"\n\n克隆 voice_id：`{voice_id}`"
        return status, audio_from_json, voice_id

    # 情况 3：返回 voice_id，再调用普通 TTS 试听
    voice_id = extract_voice_id(json_data)
    if not voice_id:
        raise gr.Error(
            "Voice Clone 返回了 JSON，但没有找到 voice_id。\n\n"
            f"请检查 VOICE_ID_JSON_PATH 环境变量。\n\n"
            f"当前返回：{json.dumps(json_data, ensure_ascii=False)[:1200]}"
        )

    audio = generate_tts(
        text=preview_text,
        model=tts_model,
        voice=voice_id,
        speed=speed,
        response_format=response_format,
    )

    status = (
        "✅ 声音克隆成功，并已生成试听。\n\n"
        f"克隆 voice_id：`{voice_id}`"
    )

    return status, audio, voice_id


# =========================
# Gradio 页面
# =========================

custom_css = """
footer {visibility: hidden}
.gradio-container {max-width: 1100px !important}
"""

with gr.Blocks(
    title="TTS 在线面板",
    css=custom_css,
) as demo:
    gr.Markdown(
        """
# 🎙️ TTS 在线试听面板

支持：

- 普通 TTS：输入文字，直接播放
- Voice Design：输入声音描述，生成音色并试听
- Voice Clone：上传参考音频，克隆音色并试听

> API Key 保存在服务器环境变量里，不会显示在网页上。
"""
    )

    with gr.Accordion("当前后端配置", open=False):
        gr.Markdown(
            f"""
- API_BASE：`{API_BASE or "未配置"}`
- TTS_PATH：`{TTS_PATH}`
- VOICE_DESIGN_PATH：`{VOICE_DESIGN_PATH}`
- VOICE_CLONE_PATH：`{VOICE_CLONE_PATH}`
- VOICE_ID_JSON_PATH：`{VOICE_ID_JSON_PATH}`
- CLONE_FILE_FIELD：`{CLONE_FILE_FIELD}`
"""
        )

    with gr.Tab("普通 TTS"):
        gr.Markdown("## 输入文字，直接生成语音")

        with gr.Row():
            tts_model = gr.Textbox(
                label="TTS 模型名",
                value=DEFAULT_TTS_MODEL,
            )
            tts_voice = gr.Textbox(
                label="音色 / voice_id",
                value=DEFAULT_TTS_VOICE,
            )

        tts_text = gr.Textbox(
            label="要朗读的文字",
            value="你好，这是一次普通 TTS 在线试听。",
            lines=6,
        )

        with gr.Row():
            tts_speed = gr.Slider(
                label="语速",
                minimum=0.25,
                maximum=4.0,
                value=1.0,
                step=0.05,
            )
            tts_format = gr.Dropdown(
                label="音频格式",
                choices=["mp3", "wav", "aac", "flac", "opus"],
                value="mp3",
            )

        tts_btn = gr.Button("生成并试听", variant="primary")
        tts_status = gr.Markdown()
        tts_audio = gr.Audio(label="试听结果", type="filepath", autoplay=True)

        tts_btn.click(
            fn=ui_tts,
            inputs=[tts_text, tts_model, tts_voice, tts_speed, tts_format],
            outputs=[tts_status, tts_audio],
        )

    with gr.Tab("Voice Design 声音设计"):
        gr.Markdown("## 输入声音描述，生成一个新音色并试听")

        design_prompt = gr.Textbox(
            label="声音描述",
            value="一个年轻、温柔、清晰的中文女声，适合短剧旁白，情绪自然，有亲和力。",
            lines=5,
        )

        design_preview_text = gr.Textbox(
            label="试听文本",
            value="你好，这是我刚刚设计出来的声音。你觉得这个音色适合做短剧旁白吗？",
            lines=5,
        )

        with gr.Row():
            design_model = gr.Textbox(
                label="Voice Design 模型名",
                value=DEFAULT_DESIGN_MODEL,
            )
            design_tts_model = gr.Textbox(
                label="试听使用的 TTS 模型",
                value=DEFAULT_TTS_MODEL,
            )

        with gr.Row():
            design_speed = gr.Slider(
                label="试听语速",
                minimum=0.25,
                maximum=4.0,
                value=1.0,
                step=0.05,
            )
            design_format = gr.Dropdown(
                label="音频格式",
                choices=["mp3", "wav", "aac", "flac", "opus"],
                value="mp3",
            )

        design_btn = gr.Button("生成声音并试听", variant="primary")
        design_status = gr.Markdown()
        design_audio = gr.Audio(label="试听结果", type="filepath", autoplay=True)
        design_voice_id = gr.Textbox(label="生成的 voice_id，可复制到普通 TTS 使用")

        design_btn.click(
            fn=design_voice_and_preview,
            inputs=[
                design_prompt,
                design_preview_text,
                design_model,
                design_tts_model,
                design_speed,
                design_format,
            ],
            outputs=[design_status, design_audio, design_voice_id],
        )

    with gr.Tab("Voice Clone 声音克隆"):
        gr.Markdown("## 上传一段参考音频，克隆音色并试听")

        clone_audio_file = gr.Audio(
            label="上传参考音频，建议 10–60 秒清晰人声",
            type="filepath",
        )

        clone_preview_text = gr.Textbox(
            label="试听文本",
            value="你好，这是我根据参考音频克隆出来的声音，现在正在进行在线试听。",
            lines=5,
        )

        with gr.Row():
            clone_model = gr.Textbox(
                label="Voice Clone 模型名",
                value=DEFAULT_CLONE_MODEL,
            )
            clone_voice_name = gr.Textbox(
                label="给这个克隆音色起个名字",
                value="my_cloned_voice",
            )

        with gr.Row():
            clone_tts_model = gr.Textbox(
                label="试听使用的 TTS 模型",
                value=DEFAULT_TTS_MODEL,
            )
            clone_speed = gr.Slider(
                label="试听语速",
                minimum=0.25,
                maximum=4.0,
                value=1.0,
                step=0.05,
            )
            clone_format = gr.Dropdown(
                label="音频格式",
                choices=["mp3", "wav", "aac", "flac", "opus"],
                value="mp3",
            )

        clone_btn = gr.Button("克隆声音并试听", variant="primary")
        clone_status = gr.Markdown()
        clone_result_audio = gr.Audio(label="试听结果", type="filepath", autoplay=True)
        clone_voice_id = gr.Textbox(label="克隆得到的 voice_id，可复制到普通 TTS 使用")

        clone_btn.click(
            fn=clone_voice_and_preview,
            inputs=[
                clone_audio_file,
                clone_preview_text,
                clone_model,
                clone_voice_name,
                clone_tts_model,
                clone_speed,
                clone_format,
            ],
            outputs=[clone_status, clone_result_audio, clone_voice_id],
        )

    gr.Markdown(
        """
---

### 使用提醒

声音克隆请只克隆你自己或已获得授权的声音。  
如果接口报错，请先确认你的 API 是否真的是 OpenAI 兼容格式，尤其是 Voice Design 和 Voice Clone 的路径、字段名、返回格式。
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
