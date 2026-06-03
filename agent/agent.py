"""
AIGC Agent 核心调度器 — 基于 OpenAI Function Calling

支持:
- OpenAI 官方 API（原生 Function Calling）
- 任何 OpenAI 兼容接口（vLLM / Ollama / 国产模型等）
- 自动多轮工具调用循环
- 文本回退解析（兼容不支持原生 Function Calling 的模型）
"""

import json
import os
import re
import uuid
from typing import Optional
from openai import OpenAI

from .tools import ToolsRegistry
from .prompts import SYSTEM_PROMPT


def _parse_text_tool_calls(text: str) -> list[dict]:
    """
    从文本中提取 JSON 格式的工具调用（回退方案）
    支持格式:
        {"name": "generate_image", "arguments": {...}}
        {"function": "generate_image", "arguments": {...}}
        <tool_call>{"name": "generate_image", ...}</tool_call>
    """
    if not text:
        return []

    calls = []
    seen = set()  # 去重
    # 匹配 JSON 对象，包含 "name" 或 "function" 键
    patterns = [
        r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^{}]*\}[^{}]*\}',
        r'\{[^{}]*"function"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^{}]*\}[^{}]*\}',
        r'\{[^{}]*"name"\s*:\s*"[^"]+"[^{}]*"arguments"\s*:\s*\{[^{}]*\}[^{}]*\}',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                obj = json.loads(match)
                name = obj.get("name") or obj.get("function")
                args = obj.get("arguments", {})
                if name and name in ToolsRegistry.TOOLS:
                    call_key = (name, json.dumps(args, sort_keys=True))
                    if call_key not in seen:
                        seen.add(call_key)
                        calls.append({"name": name, "arguments": args})
            except (json.JSONDecodeError, KeyError):
                continue

    return calls


def _parse_tool_calls_from_text(text: str) -> list[dict]:
    """
    更激进的文本解析：也尝试匹配模型输出的自然语言工具调用
    例如: "我将使用 generate_image 工具，参数是..."
    """
    calls = _parse_text_tool_calls(text)
    if calls:
        return calls

    # 尝试匹配 "generate_image" 后面跟着 JSON 参数的模式
    for tool_name in ToolsRegistry.TOOLS:
        # 匹配: generate_image({"prompt": "..."})
        pattern = rf'{re.escape(tool_name)}\s*\(\s*\{{[^}}]*\}}\s*\)'
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                json_str = re.search(r'\{[^}]*\}', match).group()
                args = json.loads(json_str)
                calls.append({"name": tool_name, "arguments": args})
            except (json.JSONDecodeError, AttributeError):
                continue

    return calls


class AIGCAgent:
    """AIGC 智能助手 Agent"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o",
        max_tool_rounds: int = 5,
        verbose: bool = True,
    ):
        """
        初始化 Agent

        参数:
            api_key: OpenAI API Key。默认从环境变量 OPENAI_API_KEY 读取
            base_url: API 地址。默认 OpenAI 官方；可用 https://api.openai.com/v1
                      国内可用: https://dashscope.aliyuncs.com/compatible-mode/v1 (通义千问)
            model: 模型名。推荐 gpt-4o / gpt-4o-mini / qwen-plus / deepseek-chat
            max_tool_rounds: 最大工具调用轮数，防止死循环
            verbose: 是否打印调试信息
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "请设置 OPENAI_API_KEY 环境变量或传入 api_key 参数。\n"
                "例如: export OPENAI_API_KEY=sk-xxxx"
            )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
        )
        self.model = model
        self.max_tool_rounds = max_tool_rounds
        self.verbose = verbose

        self.tools = ToolsRegistry.get_schemas()
        self.messages: list[dict] = []

        # 初始化对话
        self._reset_conversation()

    def _reset_conversation(self):
        """重置对话历史"""
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

    def _log(self, message: str):
        if self.verbose:
            print(f"[Agent] {message}")

    def chat(self, user_message: str) -> str:
        """
        处理用户消息并返回回复

        参数:
            user_message: 用户输入的自然语言

        返回:
            Agent 的文本回复
        """
        # 对1.5B等小模型强提醒：必须输出JSON
        enforced = user_message + "\n\n【重要】请用JSON格式调用工具，禁止只说文字不输出JSON。"
        self.messages.append({"role": "user", "content": enforced})
        self._log(f"用户: {user_message[:80]}...")

        for round_idx in range(self.max_tool_rounds):
            self._log(f"→ 第 {round_idx + 1} 轮推理...")

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=self.tools,
                    tool_choice="auto",
                    temperature=0.7,
                )
            except Exception as e:
                self._log(f"API 调用失败: {e}")
                return f"抱歉，AI 服务暂时不可用: {str(e)}"

            choice = response.choices[0]
            message = choice.message

            # 如果 LLM 决定直接回复（没有原生 tool_calls）
            if not message.tool_calls:
                content = message.content or ""

                # ── 文本回退解析：尝试从文本中提取工具调用 ──
                text_calls = _parse_tool_calls_from_text(content)
                if text_calls:
                    self._log(f"🔄 从文本中解析到 {len(text_calls)} 个工具调用")
                    # 构造假的 tool_calls 继续执行
                    fake_tool_calls = []
                    for i, tc in enumerate(text_calls):
                        fake_id = f"call_{uuid.uuid4().hex[:8]}"
                        fake_tool_calls.append({
                            "id": fake_id,
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["arguments"], ensure_ascii=False),
                            },
                        })
                        # 直接执行
                        self._log(f"🔧 调用工具: {tc['name']}({json.dumps(tc['arguments'], ensure_ascii=False)[:120]})")
                        result = ToolsRegistry.execute(tc["name"], tc["arguments"])
                        result_str = json.dumps(result, ensure_ascii=False, default=str)
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": fake_id,
                            "content": result_str,
                        })
                        self._log(f"✅ 工具返回: {result.get('message', 'OK')[:100]}")

                    # 让 LLM 总结结果
                    try:
                        summary_resp = self.client.chat.completions.create(
                            model=self.model,
                            messages=self.messages + [{"role": "user", "content": "请用中文简洁总结上述操作结果。"}],
                            temperature=0.7,
                        )
                        reply = summary_resp.choices[0].message.content or "操作已完成。"
                    except Exception:
                        reply = "操作已完成，请查看结果。"

                    self.messages.append({"role": "assistant", "content": reply})
                    return reply

                # 没有工具调用，直接返回文本
                reply = content
                self.messages.append({"role": "assistant", "content": reply})
                self._log(f"← 回复: {reply[:100]}...")
                return reply

            # ── 处理工具调用 ──
            self.messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ],
            })

            for tc in message.tool_calls:
                tool_name = tc.function.name
                tool_id = tc.id

                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}

                self._log(f"🔧 调用工具: {tool_name}({json.dumps(arguments, ensure_ascii=False)[:120]})")

                result = ToolsRegistry.execute(tool_name, arguments)
                result_str = json.dumps(result, ensure_ascii=False, default=str)

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": result_str,
                })

                self._log(f"✅ 工具返回: {result.get('message', 'OK')[:100]}")

        # 超出最大轮数
        self._log("⚠️ 达到最大工具调用轮数，强制结束")
        return "操作步骤较多，已尽力完成。请查看上述结果，如有需要可以继续告诉我。"

    def stream_chat(self, user_message: str):
        """
        流式对话生成器（逐步返回状态更新，适合 Gradio 等 UI）

        每次 yield 一个 dict:
            {"type": "thinking", "content": "..."}
            {"type": "tool_call", "tool": "...", "args": {...}}
            {"type": "tool_result", "tool": "...", "result": {...}}
            {"type": "reply", "content": "..."}
            {"type": "done"}
        """
        self.messages.append({"role": "user", "content": user_message + "\n\n【重要】请用JSON格式调用工具，禁止只说文字不输出JSON。"})
        yield {"type": "thinking", "content": f"收到消息: {user_message[:60]}..."}

        for round_idx in range(self.max_tool_rounds):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=self.tools,
                    tool_choice="auto",
                    temperature=0.7,
                )
            except Exception as e:
                yield {"type": "error", "content": f"API 调用失败: {str(e)}"}
                return

            choice = response.choices[0]
            message = choice.message

            if not message.tool_calls:
                content = message.content or ""

                # ── 文本回退解析 ──
                text_calls = _parse_tool_calls_from_text(content)
                if text_calls:
                    yield {"type": "thinking", "content": f"从文本解析到 {len(text_calls)} 个工具调用"}
                    for tc in text_calls:
                        yield {"type": "tool_call", "tool": tc["name"], "args": tc["arguments"]}
                        result = ToolsRegistry.execute(tc["name"], tc["arguments"])
                        yield {"type": "tool_result", "tool": tc["name"], "result": result}
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": f"call_{uuid.uuid4().hex[:8]}",
                            "content": json.dumps(result, ensure_ascii=False, default=str),
                        })

                    # 总结
                    try:
                        summary_resp = self.client.chat.completions.create(
                            model=self.model,
                            messages=self.messages + [{"role": "user", "content": "用中文一句话总结操作结果。"}],
                            temperature=0.7,
                        )
                        reply = summary_resp.choices[0].message.content or "操作完成。"
                    except Exception:
                        reply = "操作已完成！"

                    self.messages.append({"role": "assistant", "content": reply})
                    yield {"type": "reply", "content": reply}
                    yield {"type": "done"}
                    return

                reply = content
                self.messages.append({"role": "assistant", "content": reply})
                yield {"type": "reply", "content": reply}
                yield {"type": "done"}
                return

            # Tool calls
            tool_call_data = []
            for tc in message.tool_calls:
                tool_call_data.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                })

            self.messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": tool_call_data,
            })

            for tc in message.tool_calls:
                tool_name = tc.function.name
                tool_id = tc.id

                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}

                yield {"type": "tool_call", "tool": tool_name, "args": arguments}

                result = ToolsRegistry.execute(tool_name, arguments)
                result_str = json.dumps(result, ensure_ascii=False, default=str)

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": result_str,
                })

                yield {"type": "tool_result", "tool": tool_name, "result": result}

        yield {"type": "reply", "content": "操作步骤较多，已尽力完成。如有需要请继续告诉我。"}
        yield {"type": "done"}

    def clear_history(self):
        """清空对话历史"""
        self._reset_conversation()
        self._log("对话历史已清空")
