import os
import base64
import mimetypes
import tempfile
import requests
import gradio as gr

API_BASE = os.getenv("TTS_API_BASE", "").rstrip("/")
API_KEY = os.getenv("TTS_API_KEY", "")

MIMO_TTS_MODEL = os.getenv("MIMO_TTS_MODEL", "mimo-v2.5-tts")
MIMO_VOICE_DESIGN_MODEL = os.getenv("MIMO_VOICE_DESIGN_MODEL", "mimo-v2.5-tts-voicedesign")
MIMO_VOICE_CLONE_MODEL = os.getenv("MIMO_VOICE_CLONE_MODEL", "mimo-v2.5-tts-voiceclone")

APP_USERNAME = os.getenv("APP_USERNAME", "")
APP_PASSWORD = os.getenv("APP_PASSWORD", "")


def require_config():
    if not API_BASE:
        raise gr.Error("缺少 TTS_API_BASE，例如：https://你的newapi域名/v1")
    if not API_KEY:
        raise gr.Error("缺少 TTS_API_KEY")


def save_audio_b64(audio_b64: str, suffix=".wav") -> str:
    audio_bytes = base64.b64decode(audio_b64)
    f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.write(audio_bytes)
    f.close()
    return f.name


def file_to_data_url(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if not mime:
        mime = "audio/wav"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def extract_audio(resp_json: dict) -> str:
    try:
        audio_b64 = resp_json["choices"][0]["message"]["audio"]["data"]
        return save_audio_b64(audio_b64, ".wav")
    except Exception:
        raise gr.Error(f"没有找到音频数据，接口返回：\n{str(resp_json)[:1200]}")


def call_mimo_chat(model: str, messages: list, voice: str = "mimo_default") -> str:
    require_config()

    payload = {
        "model": model,
        "messages": messages,
        "audio": {
            "format": "wav",
            "voice": voice
        }
    }

    r = requests.post(
        f"{API_BASE}/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json=payload,
        timeout=300
    )

    if r.status_code >= 400:
        raise gr.Error(f"请求失败：HTTP {r.status_code}\n{r.text[:1500]}")

    return extract_audio(r.json())


def normal_tts(text, instruction, voice):
    text = text.strip()
    instruction = instruction.strip()
    voice = voice.strip() or "mimo_default"

    if not text:
        raise gr.Error("请输入朗读文本")

    messages = []

    if instruction:
        messages.append({
            "role": "user",
            "content": instruction
        })

    messages.append({
        "role": "assistant",
        "content": text
    })

    audio = call_mimo_chat(MIMO_TTS_MODEL, messages, voice)
    return "✅ 普通 TTS 生成成功", audio


def voice_design_tts(voice_prompt, text):
    voice_prompt = voice_prompt.strip()
    text = text.strip()

    if not voice_prompt:
        raise gr.Error("请输入声音设计描述")

    if not text:
        raise gr.Error("请输入试听文本")

    messages = [
        {
            "role": "user",
            "content": voice_prompt
        },
        {
            "role": "assistant",
            "content": text
        }
    ]

    # VoiceDesign 不需要预设 voice，声音由 user 里的描述决定
    audio = call_mimo_chat(MIMO_VOICE_DESIGN_MODEL, messages, "mimo_default")
    return "✅ 声音设计试听生成成功", audio


def voice_clone_tts(reference_audio, instruction, text):
    if not reference_audio:
        raise gr.Error("请上传参考音频")

    instruction = instruction.strip()
    text = text.strip()

    if not text:
        raise gr.Error("请输入试听文本")

    audio_data_url = file_to_data_url(reference_audio)

    # 这里采用常见的 MiMo/MCP 写法：reference_audio 以 base64 DataURL 传入。
    # 如果你的 NewAPI 上游做了不同适配，可能需要按它的文档微调字段名。
    user_content = []

    user_content.append({
        "type": "text",
        "text": instruction or "请参考上传音频中的说话人音色进行克隆，并自然朗读。"
    })

    user_content.append({
        "type": "input_audio",
        "input_audio": {
            "data": audio_data_url,
            "format": "wav"
        }
    })

    messages = [
        {
            "role": "user",
            "content": user_content
        },
        {
            "role": "assistant",
            "content": text
        }
    ]

    audio = call_mimo_chat(MIMO_VOICE_CLONE_MODEL, messages, "mimo_default")
    return "✅ 声音克隆试听生成成功", audio


with gr.Blocks(title="MiMo TTS 在线面板") as demo:
    gr.Markdown("# 🎙️ MiMo V2.5 TTS 在线面板")
    gr.Markdown("通过 NewAPI 调用 MiMo TTS / VoiceDesign / VoiceClone，生成后直接在线试听。")

    with gr.Tab("普通 TTS"):
        instruction = gr.Textbox(
            label="风格控制，可选",
            value="请用自然、清晰、适合短视频旁白的语气朗读。",
            lines=3
        )

        voice = gr.Textbox(
            label="内置音色",
            value="mimo_default"
        )

        text = gr.Textbox(
            label="朗读文本",
            value="你好，这是 MiMo V2.5 普通 TTS 的在线试听。",
            lines=6
        )

        btn = gr.Button("生成并试听", variant="primary")
        status = gr.Markdown()
        audio = gr.Audio(label="试听结果", type="filepath", autoplay=True)

        btn.click(normal_tts, [text, instruction, voice], [status, audio])

    with gr.Tab("Voice Design 声音设计"):
        voice_prompt = gr.Textbox(
            label="声音设计描述",
            value="一位年轻女性，说标准普通话，声音温柔清晰，有亲和力，适合短剧旁白。",
            lines=5
        )

        design_text = gr.Textbox(
            label="试听文本",
            value="你好，这是我根据文字描述设计出来的声音。你觉得这个音色适合做短剧旁白吗？",
            lines=6
        )

        design_btn = gr.Button("设计声音并试听", variant="primary")
        design_status = gr.Markdown()
        design_audio = gr.Audio(label="试听结果", type="filepath", autoplay=True)

        design_btn.click(voice_design_tts, [voice_prompt, design_text], [design_status, design_audio])

    with gr.Tab("Voice Clone 声音克隆"):
        reference_audio = gr.Audio(
            label="上传参考音频，建议 10–60 秒清晰人声",
            type="filepath"
        )

        clone_instruction = gr.Textbox(
            label="风格控制，可选",
            value="请保持参考音频中的说话人音色，用自然、清晰的中文朗读。",
            lines=3
        )

        clone_text = gr.Textbox(
            label="试听文本",
            value="你好，这是根据参考音频克隆出来的声音，现在正在进行在线试听。",
            lines=6
        )

        clone_btn = gr.Button("克隆声音并试听", variant="primary")
        clone_status = gr.Markdown()
        clone_audio = gr.Audio(label="试听结果", type="filepath", autoplay=True)

        clone_btn.click(
            voice_clone_tts,
            [reference_audio, clone_instruction, clone_text],
            [clone_status, clone_audio]
        )

    gr.Markdown(
        """
---
### 注意
声音克隆请只使用你自己或已获得明确授权的声音。参考音频会通过 NewAPI/上游接口上传处理。
"""
    )


if APP_USERNAME and APP_PASSWORD:
    demo.launch(server_name="0.0.0.0", server_port=7860, auth=(APP_USERNAME, APP_PASSWORD))
else:
    demo.launch(server_name="0.0.0.0", server_port=7860)
