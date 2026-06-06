"""
AIGC Studio Agent — Gradio Web UI

启动方式:
    python agent/webui.py

首次启动会自动检查依赖，如缺少 gradio 请执行:
    pip install gradio openai
"""
import os
import sys

# 确保项目根目录在 path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def check_dependencies():
    """检查并提示安装依赖"""
    missing = []
    try:
        import gradio  # noqa: F401
    except ImportError:
        missing.append("gradio")
    try:
        import openai  # noqa: F401
    except ImportError:
        missing.append("openai")

    if missing:
        print(f"缺少依赖: {', '.join(missing)}")
        print(f"请执行: pip install {' '.join(missing)}")
        sys.exit(1)


check_dependencies()

import gradio as gr
from agent.agent import AIGCAgent


# ── 全局 Agent 实例 ──
_agent: AIGCAgent = None


def _get_agent(api_key: str, base_url: str, model: str) -> AIGCAgent:
    global _agent
    if _agent is None or api_key != getattr(_agent, "_cached_key", None):
        kwargs = {"api_key": api_key, "model": model}
        if base_url and base_url.strip():
            kwargs["base_url"] = base_url.strip()
        _agent = AIGCAgent(**kwargs, verbose=False)
        _agent._cached_key = api_key  # type: ignore
    return _agent


# ── CSS 样式 ──
CUSTOM_CSS = """
.gradio-container {
    max-width: 100% !important;
    margin: 0 !important;
    padding: 8px 12px !important;
}
.header-title {
    text-align: center;
    font-size: 1.6em;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.1em;
}
.header-subtitle {
    text-align: center;
    color: #888;
    font-size: 0.85em;
    margin-bottom: 0.8em;
}
.sidebar-box {
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 10px 12px;
    margin-bottom: 8px;
    background: #fafbfc;
}
.sidebar-box h3 {
    margin: 0 0 6px 0;
    font-size: 0.9em;
    color: #555;
}
.chatbot-container {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid #e5e7eb !important;
}
.image-preview {
    border-radius: 12px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
}
.log-box {
    font-family: 'Consolas', 'Courier New', monospace !important;
    font-size: 0.8em !important;
}
"""


def build_ui():
    with gr.Blocks(title="AIGC Studio Agent — SD1.5 智能助手") as demo:
        # ── 顶部标题 ──
        gr.HTML(
            '<div class="header-title">🎨 AIGC Studio Agent</div>'
            '<div class="header-subtitle">Stable Diffusion 1.5 + LoRA 智能图像生成 · 自然语言驱动</div>'
        )

        state = gr.State({"last_image": None, "last_prompt": None, "tool_logs": []})

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=260):
                gr.Markdown("### ⚙️ 模型配置")

                # LLM 预设
                with gr.Group(elem_classes=["sidebar-box"]):
                    gr.Markdown("**🖥️ LLM 大脑**")
                    local_preset = gr.Dropdown(
                        choices=[
                            "🦙 Ollama + Qwen2.5 1.5B",
                            "🦙 Ollama + Qwen2.5 3B",
                            "🚀 vLLM + Qwen2.5 1.5B (WSL2)",
                            "🚀 vLLM + Qwen2.5 3B (WSL2)",
                            "🚀 vLLM + Qwen2.5 7B (WSL2)",
                            "（手动配置）",
                        ],
                        value="🦙 Ollama + Qwen2.5 1.5B",
                        label="选择方案",
                    )
                    model = gr.Dropdown(
                        choices=["qwen2.5:1.5b","qwen2.5:3b","qwen2.5:7b","gpt-4o-mini","gpt-4o","deepseek-chat"],
                        value="qwen2.5:1.5b", label="模型", allow_custom_value=True,
                    )
                    api_key = gr.Textbox(label="API Key", value="ollama", type="password")
                    base_url = gr.Textbox(label="API URL", value="http://localhost:11434/v1")
                    with gr.Row():
                        detect_local_btn = gr.Button("🔍 检测连接", size="sm")
                        status_indicator = gr.Textbox(label="", value="⚪", interactive=False, lines=1, max_lines=1, show_label=False, scale=3)

                # SD 底模
                with gr.Group(elem_classes=["sidebar-box"]):
                    gr.Markdown("**🎨 SD 底模**")
                    model_path = gr.Textbox(
                        label="路径", value="./models/anything-v5",
                        placeholder="./models/anything-v5",
                    )
                    with gr.Row():
                        scan_models_btn = gr.Button("📂 扫描", size="sm")
                        available_models_display = gr.Textbox(
                            label="", lines=2, max_lines=3, interactive=False,
                            placeholder="可用底模", show_label=False, scale=3,
                        )

                # LoRA 管理
                with gr.Group(elem_classes=["sidebar-box"]):
                    gr.Markdown("**📦 LoRA 管理**")
                    lora_action = gr.Dropdown(
                        choices=["加载", "卸载", "调整强度"],
                        value="加载", label="操作",
                    )
                    lora_name_input = gr.Textbox(label="LoRA 名称", value="final_lora2", placeholder="final_lora2")
                    lora_strength_slider = gr.Slider(0.1, 1.5, value=1.0, step=0.1, label="强度")
                    lora_btn = gr.Button("✅ 执行", variant="secondary", size="sm")
                    lora_status = gr.Textbox(label="", interactive=False, lines=1, max_lines=1, show_label=False)

                # 快捷操作
                with gr.Group(elem_classes=["sidebar-box"]):
                    gr.Markdown("**⚡ 快捷操作**")
                    quick_clear = gr.Button("🗑️ 清空对话", size="sm")

            # ────── 右侧 4/5：工作区 ──────
            with gr.Column(scale=4):
                # 上排：图片 + 日志 并排
                with gr.Row():
                    with gr.Column(scale=3):
                        image_display = gr.Image(
                            label="🖼️ 生成的图片",
                            height=400, interactive=False,
                            elem_classes=["image-preview"],
                        )
                    with gr.Column(scale=2):
                        tool_log = gr.Textbox(
                            label="🔧 工具调用日志",
                            lines=18, max_lines=25, interactive=False,
                            placeholder="等待操作...",
                            elem_classes=["log-box"],
                        )

                # 对话区
                chatbot = gr.Chatbot(
                    elem_classes=["chatbot-container"],
                    height=300,
                    placeholder="👋 你好！输入需求开始创作，例如：「加载 final_lora2 并画 leisai」",
                    label="💬 对话",
                )

                # 输入行
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="输入你的需求...",
                        scale=9, show_label=False, container=False,
                    )
                    send_btn = gr.Button("🚀 发送", variant="primary", scale=1)

        # ══════════════════════════════════════════════════════
        #  本地模型预设逻辑
        # ══════════════════════════════════════════════════════

        # 预设配置表
        LOCAL_PRESETS = {
            "🦙 Ollama + Qwen2.5 1.5B": {
                "base_url": "http://localhost:11434/v1",
                "model": "qwen2.5:1.5b",
                "api_key": "ollama",
            },
            "🦙 Ollama + Qwen2.5 3B": {
                "base_url": "http://localhost:11434/v1",
                "model": "qwen2.5:3b",
                "api_key": "ollama",
            },
            "🚀 vLLM + Qwen2.5 1.5B (WSL2)": {
                "base_url": "http://localhost:8000/v1",
                "model": "Qwen/Qwen2.5-1.5B-Instruct",
                "api_key": "vllm",
            },
            "🚀 vLLM + Qwen2.5 3B (WSL2)": {
                "base_url": "http://localhost:8000/v1",
                "model": "Qwen/Qwen2.5-3B-Instruct",
                "api_key": "vllm",
            },
            "🚀 vLLM + Qwen2.5 7B (WSL2)": {
                "base_url": "http://localhost:8000/v1",
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "api_key": "vllm",
            },
        }

        def on_preset_change(preset_name):
            """切换本地模型预设时自动填充配置"""
            if preset_name == "（不使用本地模型）":
                return "", "gpt-4o-mini", "", "⚪ 已切换到云端模式"
            preset = LOCAL_PRESETS.get(preset_name)
            if preset:
                return preset["api_key"], preset["model"], preset["base_url"], "🟡 预设已加载，点击「自动检测」验证连接"
            return gr.update(), gr.update(), gr.update(), "⚪ 未知预设"

        def on_auto_detect(base_url_val, model_val, api_key_val):
            """自动检测本地 LLM 服务是否可用"""
            import json
            from urllib.request import Request, urlopen
            from urllib.error import URLError

            url = base_url_val.strip()
            if not url:
                return "⚪ 未设置 API 地址"

            # 尝试 /v1/models 端点
            models_url = url.rstrip("/") + "/models"
            try:
                req = Request(models_url)
                req.add_header("Authorization", f"Bearer {api_key_val or 'ollama'}")
                resp = urlopen(req, timeout=5)
                data = json.loads(resp.read().decode())
                model_list = [m.get("id", str(m)) for m in data.get("data", [])]
                if model_list:
                    return f"🟢 连接成功！可用模型: {', '.join(model_list[:5])}"
                return "🟢 连接成功（未返回模型列表）"
            except URLError as e:
                return f"🔴 连接失败: {str(e.reason) if hasattr(e, 'reason') else str(e)}"
            except Exception as e:
                return f"🔴 检测异常: {str(e)[:80]}"

        local_preset.change(
            on_preset_change,
            [local_preset],
            [api_key, model, base_url, status_indicator],
        )

        detect_local_btn.click(
            on_auto_detect,
            [base_url, model, api_key],
            [status_indicator],
        )

        # ── 底模扫描 ──
        def on_scan_models():
            """扫描 models/ 目录下可用的 diffusers 格式底模"""
            import os as _os
            models_dir = _os.path.join(PROJECT_ROOT, "models")
            if not _os.path.exists(models_dir):
                return "models/ 目录不存在"

            found = []
            for name in sorted(_os.listdir(models_dir)):
                full = _os.path.join(models_dir, name)
                if not _os.path.isdir(full):
                    continue
                unet_dir = _os.path.join(full, "unet")
                has_unet = _os.path.exists(_os.path.join(unet_dir, "diffusion_pytorch_model.safetensors"))
                has_config = _os.path.exists(_os.path.join(unet_dir, "config.json"))
                if has_unet or has_config:
                    rel = _os.path.relpath(full, PROJECT_ROOT)
                    found.append(f"✅ {rel}")

            if not found:
                return "未找到。用 python scripts/convert_checkpoint.py -i 模型.safetensors -o ./models/模型名 转换"
            return "\n".join(found)

        scan_models_btn.click(
            on_scan_models,
            [],
            [available_models_display],
        )

        # ══════════════════════════════════════════════════════
        #  事件处理
        # ══════════════════════════════════════════════════════

        def on_send(message, history, current_state, api_key_val, base_url_val, model_val, model_path_val):
            """处理用户消息"""
            if not message or not message.strip():
                return "", history, current_state, "请输入内容"

            # 同步底模路径到 tools 层
            if model_path_val and model_path_val.strip():
                from agent.tools import set_model_path as _set_mp
                _set_mp(model_path_val.strip())

            history = history or []
            tool_logs = current_state.get("tool_logs", [])

            try:
                agent = _get_agent(
                    api_key=api_key_val or os.getenv("OPENAI_API_KEY", ""),
                    base_url=base_url_val,
                    model=model_val,
                )
            except ValueError as e:
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": f"❌ 初始化失败: {str(e)}"})
                return "", history, current_state, str(e)

            history.append({"role": "user", "content": message})

            # 流式处理
            reply_parts = []
            new_logs = []
            last_result = current_state.copy()

            for event in agent.stream_chat(message):
                if event["type"] == "thinking":
                    status = f"🤔 {event['content']}"
                    yield "", history + [{"role": "assistant", "content": status}], last_result, "\n".join(tool_logs + [status])

                elif event["type"] == "tool_call":
                    tool_name = event["tool"]
                    args = event.get("args", {})
                    display_args = {k: v for k, v in args.items() if k not in ("prompt", "negative_prompt")}
                    log_entry = f"🔧 {tool_name}({display_args})"
                    new_logs.append(log_entry)
                    tool_logs.append(log_entry)
                    status = f"🔧 调用 {tool_name}..."
                    yield "", history + [{"role": "assistant", "content": status}], last_result, "\n".join(tool_logs[-15:])

                elif event["type"] == "tool_result":
                    result = event["result"]
                    if result.get("success"):
                        data = result.get("data", {})
                        # 如果是生成图片
                        if "image_path" in (data or {}):
                            last_result["last_image"] = data["image_path"]
                            last_result["last_prompt"] = data.get("prompt", "")
                        # 如果是列出LoRA
                        if "loras" in (data or {}):
                            lora_info = data.get("loras", [])
                            last_result["loras"] = lora_info

                    status = "✅ " + result.get("message", "OK")
                    log_entry = status[:100]
                    new_logs.append(log_entry)
                    tool_logs.append(log_entry)
                    yield "", history, last_result, "\n".join(tool_logs[-15:])

                elif event["type"] == "reply":
                    reply_parts.append(event["content"])
                    reply_text = "".join(reply_parts)
                    yield "", history + [{"role": "assistant", "content": reply_text}], last_result, "\n".join(tool_logs[-15:])

                elif event["type"] == "error":
                    history.append({"role": "assistant", "content": f"❌ {event['content']}"})
                    yield "", history, last_result, "\n".join(tool_logs)

                elif event["type"] == "done":
                    pass

            # 返回最终状态
            final_reply = "".join(reply_parts) if reply_parts else "操作完成，请查看结果。"
            if not any(h.get("role") == "assistant" and h.get("content") == final_reply for h in history):
                history.append({"role": "assistant", "content": final_reply})

            final_log = "\n".join(tool_logs[-15:])
            yield "", history, last_result, final_log

        def on_clear():
            global _agent
            if _agent:
                _agent.clear_history()
            return [], {"last_image": None, "last_prompt": None, "tool_logs": []}, None, "对话已清空"

        def update_image_display(state):
            """同步右侧图片显示"""
            if state and state.get("last_image") and os.path.exists(state["last_image"]):
                return state["last_image"]
            return None

        # 绑定事件
        msg_input.submit(
            on_send,
            [msg_input, chatbot, state, api_key, base_url, model, model_path],
            [msg_input, chatbot, state, tool_log],
        ).then(
            update_image_display, [state], [image_display]
        )

        send_btn.click(
            on_send,
            [msg_input, chatbot, state, api_key, base_url, model, model_path],
            [msg_input, chatbot, state, tool_log],
        ).then(
            update_image_display, [state], [image_display]
        )

        quick_clear.click(
            on_clear,
            [], [chatbot, state, image_display, tool_log],
        )

        # LoRA 快捷操作
        def on_lora_action(action, lora_name, strength_val):
            from agent.tools import switch_lora
            act_map = {"加载": "load", "卸载": "unload", "调整强度": "set_strength"}
            result = switch_lora(action=act_map.get(action, "load"), lora_name=lora_name, strength=strength_val)
            return result.get("message", "未知结果")

        lora_btn.click(
            on_lora_action,
            [lora_action, lora_name_input, lora_strength_slider],
            [lora_status],
        )

    return demo


def main():
    print("=" * 60)
    print("  🎨 AIGC Studio Agent — SD1.5 智能助手")
    print("=" * 60)
    print()
    print("  启动前请确保:")
    print("  1. pip install gradio openai")
    print("  2. 已下载 SD1.5 模型: python download_models.py")
    print("  3. 设置 API Key: export OPENAI_API_KEY=sk-xxxx")
    print()
    print("=" * 60)

    demo = build_ui()
    demo.queue(default_concurrency_limit=1, max_size=10)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(primary_hue="violet"),
    )


if __name__ == "__main__":
    main()
