"""
本地 LLM 连接测试工具

用法:
    python scripts/test_llm_connection.py
    python scripts/test_llm_connection.py --url http://localhost:8000/v1
"""
import json
import sys
from urllib.request import Request, urlopen
from urllib.error import URLError


def test_connection(base_url: str, api_key: str = "ollama", timeout: int = 5) -> dict:
    """测试 LLM API 是否可达"""
    base = base_url.rstrip("/")

    # 1. 测试 /models 端点
    result = {"endpoint": base, "models": [], "chat_ok": False, "error": None}

    try:
        req = Request(f"{base}/models")
        req.add_header("Authorization", f"Bearer {api_key}")
        resp = urlopen(req, timeout=timeout)
        data = json.loads(resp.read().decode())
        result["models"] = [m.get("id", str(m)) for m in data.get("data", [])]
        print(f"✅ /models 端点正常: {len(result['models'])} 个模型可用")
    except URLError as e:
        result["error"] = f"连接 /models 失败: {e.reason}"
        print(f"❌ {result['error']}")
        return result
    except Exception as e:
        result["error"] = str(e)
        print(f"❌ {result['error']}")
        return result

    # 2. 测试 /chat/completions 端点（发送一个简单请求）
    try:
        chat_data = json.dumps({
            "model": result["models"][0] if result["models"] else "unknown",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 10,
        }).encode()
        req = Request(f"{base}/chat/completions", data=chat_data)
        req.add_header("Authorization", f"Bearer {api_key}")
        req.add_header("Content-Type", "application/json")
        resp = urlopen(req, timeout=timeout)
        chat_result = json.loads(resp.read().decode())
        reply = chat_result["choices"][0]["message"]["content"]
        result["chat_ok"] = True
        print(f"✅ /chat/completions 正常: 回复 '{reply[:30]}...'")
    except Exception as e:
        result["error"] = f"对话测试失败: {str(e)[:100]}"
        print(f"⚠️  {result['error']}")

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="本地 LLM 连接测试")
    parser.add_argument("--url", default="http://localhost:11434/v1",
                        help="API 地址 (默认 Ollama: http://localhost:11434/v1)")
    parser.add_argument("--key", default="ollama", help="API Key")
    parser.add_argument("--timeout", type=int, default=5, help="超时秒数")
    args = parser.parse_args()

    print("=" * 50)
    print("  🔍 本地 LLM 连接测试")
    print("=" * 50)
    print(f"  地址: {args.url}")
    print()

    result = test_connection(args.url, args.key, args.timeout)

    print()
    print("=" * 50)
    if result["chat_ok"]:
        print("  ✅ 一切正常！可以启动 AIGC Agent 了。")
        print()
        print("  Agent 设置:")
        print(f"    API Base URL: {args.url}")
        print(f"    模型: {result['models'][0] if result['models'] else '（自行填写）'}")
        print(f"    API Key: {args.key}")
    elif result["models"]:
        print("  ⚠️  模型列表可获取但对话测试失败，可尝试启动 Agent。")
    else:
        print("  ❌ 无法连接本地 LLM。请检查服务是否启动。")
        print()
        print("  启动方式:")
        print("    Ollama:  ollama serve    (然后 ollama pull qwen2.5:1.5b)")
        print("    vLLM:    vllm serve Qwen/Qwen2.5-1.5B-Instruct")
    print("=" * 50)


if __name__ == "__main__":
    main()
