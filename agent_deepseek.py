"""
================================================================================
  Minimal DeepSeek Agent — Stage 1 完整示例 / Complete walkthrough
  从「调一次 LLM」到「跑通一个带保险的 agent loop」
  From a single LLM call to a hardened observe -> think -> act loop.

  DeepSeek 兼容 OpenAI 接口，所以用 openai 这个库，只改 base_url。
  DeepSeek is OpenAI-compatible: use the openai SDK, just point base_url at it.
================================================================================

环境准备 / Setup:
    python3 -m venv .venv && source .venv/bin/activate
    pip install openai python-dotenv
    echo 'DEEPSEEK_API_KEY=sk-xxxx' > .env        # platform.deepseek.com

运行 / Run:
    python agent.py
================================================================================
"""

import os
import json
import concurrent.futures
from dotenv import load_dotenv
from openai import OpenAI

# 从 .env 读取 key / load DEEPSEEK_API_KEY from .env
load_dotenv()

# DeepSeek 用 OpenAI 客户端，只改 base_url / OpenAI client pointed at DeepSeek
client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

MODEL = "deepseek-chat"   # DeepSeek 的对话模型 / DeepSeek's chat model


# ==============================================================================
# Step 1 — 普通对话 / Plain conversation
# ------------------------------------------------------------------------------
# 最小调用：messages 是一个 role/content 列表。返回里取 choices[0].message。
# Smallest call: messages is a list of role/content dicts.
# ==============================================================================
def step1_plain_chat():
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "用一句话解释什么是 agent loop"}],
    )
    print("[Step 1]", resp.choices[0].message.content)


# ==============================================================================
# Step 2 — 结构化 JSON 输出 / Structured JSON output
# ------------------------------------------------------------------------------
# 用 response_format={"type": "json_object"} 强制返回合法 JSON。
# 注意：用此模式时，提示词里必须出现 "json" 字样并说明字段，否则会报错。
# Force valid JSON with response_format. The prompt MUST mention "json".
# ==============================================================================
def step2_structured_json():
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": (
                "给我电影《盗梦空间》的信息，"
                "以 json 返回，字段：title(片名), year(年份, 整数), director(导演)"
            ),
        }],
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content
    print("[Step 2] raw :", raw)
    data = json.loads(raw)        # 安全解析 / parse
    print("[Step 2] year:", data["year"])


# ==============================================================================
# Step 3 — 定义工具函数 + schema / Define tools + schema
# ------------------------------------------------------------------------------
# OpenAI 风格接口需要你手写 JSON schema 描述工具（不像 Gemini 自动从 docstring 生成）。
# description 就是模型看到的"说明书"，决定它何时调、传什么参数。
# Unlike Gemini, the OpenAI-style API needs an explicit JSON schema per tool.
# ==============================================================================
def calculator(expression: str) -> float:
    """计算一个数学表达式 / evaluate an arithmetic expression."""
    return eval(expression, {"__builtins__": {}}, {})   # 仅演示 / demo only


def read_file(path: str) -> str:
    """读取文本文件内容 / read a text file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# 工具名 -> 真实函数 / name -> callable registry
TOOLS = {"calculator": calculator, "read_file": read_file}

# 工具的 JSON schema，发给模型看 / tool schemas sent to the model
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算一个数学表达式并返回结果",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "合法算术表达式，例如 (128 + 64) * 3",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取一个文本文件的内容并返回",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件的相对或绝对路径"}
                },
                "required": ["path"],
            },
        },
    },
]


# ==============================================================================
# Step 4 — 解析 tool call / Parse the tool call
# ------------------------------------------------------------------------------
# 模型不执行函数，只返回"意图": message.tool_calls。执行权在你手里 = act 入口。
# arguments 是 JSON 字符串，要 json.loads 解析。
# The model returns tool_calls (an intent), not a result. arguments is a string.
# ==============================================================================
def step4_parse_tool_call():
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "帮我算一下 (128 + 64) * 3 是多少"}],
        tools=TOOL_SCHEMAS,
    )
    msg = resp.choices[0].message
    # 坑 / Pitfall: tool_calls 可能为 None(直接回答)或含多个(并行)。遍历 + 判空。
    if msg.tool_calls:
        for tc in msg.tool_calls:
            print("[Step 4] name:", tc.function.name)              # "calculator"
            print("[Step 4] args:", json.loads(tc.function.arguments))  # {'expression': ...}


# ==============================================================================
# Step 5 — 执行工具并把结果喂回 / Execute + feed result back
# ------------------------------------------------------------------------------
# 一次工具调用 = 至少两次 API 调用。模型无记忆，每次带完整 messages。
# 关键：assistant 的 tool_calls 消息 与 role:"tool" 的结果消息 必须配对，
#       且 tool 消息要带上对应的 tool_call_id。
# assistant tool_calls msg and role:"tool" result msg must be paired by id.
# ==============================================================================
def step5_execute_and_feed_back():
    messages = [{"role": "user", "content": "帮我算一下 (128 + 64) * 3 是多少"}]

    # 1) 第一次请求 / first call
    resp = client.chat.completions.create(
        model=MODEL, messages=messages, tools=TOOL_SCHEMAS)
    msg = resp.choices[0].message

    # 2) 把 assistant 这轮回复(含 tool_calls)加回历史 / append assistant turn
    messages.append(msg)

    # 3) 逐个执行工具，并按 tool_call_id 配对塞回结果 / run + append paired results
    for tc in msg.tool_calls:
        args = json.loads(tc.function.arguments)
        result = TOOLS[tc.function.name](**args)        # 真正执行 / execute
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,                      # 必须对上 / must match the call id
            "content": str(result),
        })

    # 4) 再请求一次，拿最终答案 / second call -> final answer
    final = client.chat.completions.create(
        model=MODEL, messages=messages, tools=TOOL_SCHEMAS)
    print("[Step 5]", final.choices[0].message.content)


# ==============================================================================
# Step 6 — 完整 agent loop + 四道保险 / Full loop with four safeguards
# ------------------------------------------------------------------------------
# observe -> think -> act 的完整闭环。
# 保险: (1) 最大步数 (2) 未知工具 (3) 单工具超时 (4) 异常不炸循环(喂回模型)
# ==============================================================================
def run_agent(user_input: str, max_steps: int = 8, tool_timeout: int = 10) -> str:
    messages = [{"role": "user", "content": user_input}]

    for step in range(max_steps):                       # 保险1 / max steps
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOL_SCHEMAS)
        msg = resp.choices[0].message
        messages.append(msg)                            # assistant 回复入历史

        if not msg.tool_calls:                          # 无调用 = 最终答案 / done
            return msg.content

        for tc in msg.tool_calls:
            try:
                fn = TOOLS.get(tc.function.name)
                if fn is None:                          # 保险2 / hallucinated tool
                    raise ValueError(f"未知工具 unknown tool: {tc.function.name}")

                args = json.loads(tc.function.arguments)

                # 保险3 / per-tool timeout
                with concurrent.futures.ThreadPoolExecutor() as ex:
                    future = ex.submit(fn, **args)
                    result = future.result(timeout=tool_timeout)

                content = str(result)
            except Exception as e:
                # 保险4 / 错误喂回模型让它自我纠正 / feed error back
                content = f"ERROR: {e}"

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": content,
            })

    return "达到最大步数仍未完成 / hit max_steps without finishing"


# ==============================================================================
# 入口 / Entry point
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    step1_plain_chat()
    print("=" * 60)
    step2_structured_json()
    print("=" * 60)
    step4_parse_tool_call()
    print("=" * 60)
    step5_execute_and_feed_back()
    print("=" * 60)
    print("[Step 6]", run_agent("先算 (128 + 64) * 3，再把结果加上 100"))
    print("=" * 60)
