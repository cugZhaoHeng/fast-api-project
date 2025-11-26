from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
import json
from typing import Any, Dict, List, Optional
from langchain_core.messages import AIMessage, ToolMessage


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

def _msg_type(obj) -> Optional[str]:
    t = _get(obj, "type") or _get(obj, "role")
    if t is None:
        cls_name = obj.__class__.__name__.lower()
        if "human" in cls_name:
            return "human"
        if "ai" in cls_name or "assistant" in cls_name:
            return "ai"
        if "tool" in cls_name:
            return "tool"
    return t

def _flatten_content(c):
    if isinstance(c, list):
        parts = []
        for p in c:
            if isinstance(p, dict):
                if "text" in p:
                    parts.append(str(p["text"]))
                elif "content" in p:
                    parts.append(str(p["content"]))
                else:
                    parts.append(str(p))
            else:
                parts.append(str(p))
        return "\n".join(parts)
    return c

def _maybe_json_load(s):
    if isinstance(s, str):
        try:
            return json.loads(s)
        except Exception:
            return s
    return s

def _parse_tool_calls_structured(ai_msg) -> List[Dict[str, Any]]:
    """
    同时兼容两种形态：
      1) 规范化形态（AIMessage.tool_calls）：{'name': 'tool', 'args': {...}, 'id': ...}
      2) OpenAI 原始形态（additional_kwargs/response_metadata）：
         {'id': ..., 'function': {'name': ..., 'arguments': '...'}}
    优先使用 ai_msg.tool_calls；若无则回退 additional_kwargs/response_metadata。
    """
    calls: List[Dict[str, Any]] = []

    def add_from_list(lst):
        if not lst:
            return
        for call in lst:
            if not isinstance(call, dict):
                continue
            if "function" in call and isinstance(call["function"], dict):
                # OpenAI 原始形态
                func = call["function"]
                name = func.get("name")
                args = _maybe_json_load(func.get("arguments"))
                cid = call.get("id")
            else:
                # 规范化形态
                name = call.get("name") or call.get("tool_name")
                args = call.get("args")
                if isinstance(args, str):
                    args = _maybe_json_load(args)
                if args is None and "arguments" in call:
                    args = _maybe_json_load(call.get("arguments"))
                cid = call.get("id") or call.get("tool_call_id")
            calls.append({"id": cid, "name": name, "args": args})

    # 1) 优先使用规范化的 AIMessage.tool_calls
    tc_attr = _get(ai_msg, "tool_calls")
    if tc_attr:
        add_from_list(tc_attr)
        return calls

    # 2) 回退 additional_kwargs
    ak = _get(ai_msg, "additional_kwargs", {}) or {}
    add_from_list(ak.get("tool_calls"))

    fc = ak.get("function_call")
    if isinstance(fc, dict):
        calls.append({
            "id": None,
            "name": fc.get("name"),
            "args": _maybe_json_load(fc.get("arguments")),
        })

    # 3) 再回退 response_metadata
    rm = _get(ai_msg, "response_metadata", {}) or {}
    add_from_list(rm.get("tool_calls"))

    fc2 = rm.get("function_call")
    if isinstance(fc2, dict):
        calls.append({
            "id": None,
            "name": fc2.get("name"),
            "args": _maybe_json_load(fc2.get("arguments")),
        })

    return calls

def parse_tool_steps(messages: List[Any]) -> List[Dict[str, Any]]:
    """
    逐个 AIMessage 建“窗口”：将该 AIMessage 的调用计划与其后、下一条 AIMessage 之前的
    所有 ToolMessage 按顺序一一配对（不依赖 id）。这能修复同一 id 被复用导致的错配。
    返回结构：[{index, tool_call_id, name, args, result}]
    """
    steps: List[Dict[str, Any]] = []

    # 找到每个 AIMessage 的索引
    ai_indices = [i for i, m in enumerate(messages) if isinstance(m, AIMessage) or _msg_type(m) == "ai"]
    if not ai_indices:
        # 没有 AIMessage，兜底把所有 ToolMessage 列出来
        tool_msgs = [m for m in messages if isinstance(m, ToolMessage) or _msg_type(m) == "tool"]
        for i, tm in enumerate(tool_msgs, start=1):
            steps.append({
                "index": i,
                "tool_call_id": _get(tm, "tool_call_id"),
                "name": _get(tm, "name"),
                "args": None,
                "result": _flatten_content(_get(tm, "content")),
            })
        return steps

    # 逐个 AIMessage 建窗口
    for ai_idx_pos, ai_idx in enumerate(ai_indices):
        ai_msg = messages[ai_idx]
        calls = _parse_tool_calls_structured(ai_msg)
        if not calls:
            continue  # 这个 AIMessage 没有工具调用计划

        # 窗口右边界：下一条 AIMessage 的索引，否则到消息末尾
        next_ai_idx = ai_indices[ai_idx_pos + 1] if (ai_idx_pos + 1) < len(ai_indices) else len(messages)

        # 收集该窗口内的 ToolMessage（严格按出现顺序）
        window_tool_msgs = []
        for k in range(ai_idx + 1, next_ai_idx):
            m = messages[k]
            if isinstance(m, ToolMessage) or _msg_type(m) == "tool":
                window_tool_msgs.append(m)

        # 逐一顺序配对
        pair_count = max(len(calls), len(window_tool_msgs))
        for j in range(pair_count):
            call = calls[j] if j < len(calls) else None
            tm = window_tool_msgs[j] if j < len(window_tool_msgs) else None

            steps.append({
                "index": len(steps) + 1,
                "tool_call_id": (_get(tm, "tool_call_id") if tm is not None else (call.get("id") if call else None)),
                "name": (call.get("name") if call else _get(tm, "name")),
                "args": (call.get("args") if call else None),
                "result": (_flatten_content(_get(tm, "content")) if tm is not None else None),
            })

    # 兜底：如果完全没解析到调用计划但有 ToolMessage
    if not steps:
        tool_msgs = [m for m in messages if isinstance(m, ToolMessage) or _msg_type(m) == "tool"]
        for i, tm in enumerate(tool_msgs, start=1):
            steps.append({
                "index": i,
                "tool_call_id": _get(tm, "tool_call_id"),
                "name": _get(tm, "name"),
                "args": None,
                "result": _flatten_content(_get(tm, "content")),
            })

    return steps


@tool(description="加法工具，计算 a + b")
def add(a: int, b: int) -> int:
    # 若需普通加法改为：return a + b
    return a + b + b


# 初始化模型（替换为你的端点配置）
model = init_chat_model(
    model="qwen2.5-instruct",
    model_provider="openai",
    base_url="http://10.10.3.92:9998/v1",
    api_key="dummy_key",
    temperature=0.0,
    timeout=None,
)

# 创建 ReAct agent（会在需要时自动执行 add）
agent = create_react_agent(model, [add])

# 运行一次示例
result = agent.invoke({"messages": [{"role": "user", "content": "先用工具算 2+3，再用工具算 5+3"}]})
messages = result["messages"]

print(messages)
print(len(messages))
for i, m in enumerate(messages):
    print(f"[{i}] {m}")

steps = parse_tool_steps(messages)
print("本次工具调用次数:", len(steps))
for s in steps:
    print(f"第{s['index']}次调用: 工具={s['name']} 参数={s['args']} 返回={s['result']}")
