"""
AIGC Agent CLI — 命令行交互入口

启动方式:
    python agent/cli.py

适合没有 GUI 的环境或快速测试。
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from agent.agent import AIGCAgent
except ImportError as e:
    print(f"依赖缺失: {e}")
    print("请执行: pip install openai")
    sys.exit(1)


def _detect_local_llm():
    """自动检测本地是否有 Ollama 或 vLLM 在运行"""
    import json
    from urllib.request import Request, urlopen
    from urllib.error import URLError

    candidates = [
        ("Ollama", "http://localhost:11434/v1", "ollama"),
        ("vLLM", "http://localhost:8000/v1", "vllm"),
    ]

    for name, url, key in candidates:
        try:
            req = Request(url.rstrip("/") + "/models")
            req.add_header("Authorization", f"Bearer {key}")
            resp = urlopen(req, timeout=3)
            data = json.loads(resp.read().decode())
            models = [m.get("id", str(m)) for m in data.get("data", [])]
            if models:
                return {
                    "name": name,
                    "base_url": url,
                    "api_key": key,
                    "models": models,
                }
        except Exception:
            continue
    return None


def main():
    print("=" * 60)
    print("  🎨 AIGC Studio Agent CLI")
    print("=" * 60)
    print("  输入 'exit' 或 'quit' 退出")
    print("  输入 'clear' 清空对话")
    print("  输入 'help' 查看帮助")
    print("=" * 60)

    # ── 自动检测本地 LLM ──
    local = _detect_local_llm()
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "")
    model = os.getenv("AIGC_MODEL", "")

    if local and not api_key:
        print(f"\n🔍 检测到本地 {local['name']} 服务！")
        print(f"   可用模型: {', '.join(local['models'][:5])}")
        use_local = input(f"是否使用本地 {local['name']}？[Y/n]: ").strip().lower()
        if use_local != "n":
            api_key = local["api_key"]
            base_url = local["base_url"]
            model = model or local["models"][0]
            print(f"✅ 使用本地 {local['name']}: {model}")
    else:
        print(f"\n💡 提示: 启动 Ollama 或 vLLM 后可自动检测本地模型。")

    if not api_key:
        api_key = input("请输入 API Key (或设置环境变量 OPENAI_API_KEY): ").strip()
        if not api_key:
            print("❌ 需要 API Key 才能运行。")
            return

    if not model:
        model = "gpt-4o-mini"

    print(f"\n📡 模型: {model}")
    print(f"🔗 API: {base_url or 'OpenAI 官方'}")
    print()

    agent = AIGCAgent(
        api_key=api_key,
        base_url=base_url or None,
        model=model,
        verbose=False,
    )

    print("✅ Agent 已就绪！试试输入: 画一个可爱的橘猫，动漫风格\n")

    while True:
        try:
            user_input = input("🧑 You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "q"):
            print("👋 再见！")
            break

        if user_input.lower() == "clear":
            agent.clear_history()
            print("✅ 对话历史已清空")
            continue

        if user_input.lower() == "help":
            print("""
📖 使用帮助:
  你可以直接用自然语言告诉 Agent 你想做什么，例如：

  🎨 生成图片:
    - "画一个紫色头发的动漫女孩"
    - "生成一张赛博朋克风格的城市夜景"
    - "create a cute cat sitting on a sofa, realistic style"

  ✨ 优化提示词:
    - "帮我把「一只猫」优化成专业prompt再生成"
    - "用更好的提示词描述：海边日落"

  📋 管理 LoRA:
    - "查看有哪些可用的 LoRA"
    - "加载 final_lora 模型"
    - "把 LoRA 强度调到 0.8"

  🏋️ 训练 LoRA:
    - "用 ./data 目录训练一个 LoRA，触发词是 mystyle"

  🔄 迭代优化:
    - "上次生成的图太暗了，帮我调整再生成"
            """)
            continue

        print("🤖 Agent > ", end="", flush=True)
        reply = agent.chat(user_input)
        print(reply)
        print()


if __name__ == "__main__":
    main()
