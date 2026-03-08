
import utils.json_util

import json
from typing import List, Any, Optional, Iterator

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.graph import CompiledStateGraph
import utils.json_util

# llm_with_tool.py
import json
from typing import List, Any, Optional, Iterator

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
import utils.json_util


class FixToolCallsChatModel(BaseChatModel):
    """包装原始 LLM，自动修复 tool_calls 中的 args 字符串问题"""

    # 注意：这里我们允许 base_llm 是任何 Runnable（包括 bound 后的对象）
    base_llm: Any  # 不限定为 ChatOpenAI，因为 bind_tools 后是 RunnableBinding

    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager=None,
            **kwargs
    ) -> ChatResult:
        raw_result: ChatResult = self.base_llm._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        print(f"原始的raw_result：{utils.json_util.to_json(raw_result)}")
        fixed_messages = []
        for gen in raw_result.generations:
            ai_msg = gen.message
            if hasattr(ai_msg, 'tool_calls') and ai_msg.tool_calls:
                fixed_tool_calls = []
                for tc in ai_msg.tool_calls:
                    args = tc["args"]
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError as e:
                            print(f"⚠️ 无法解析 tool_call args: {args} | 错误: {e}")
                            args = {}
                    fixed_tool_calls.append({
                        "name": tc["name"],
                        "args": args,
                        "id": tc["id"],
                        "type": tc.get("type", "tool_call")
                    })
                fixed_msg = AIMessage(
                    content=ai_msg.content,
                    tool_calls=fixed_tool_calls,
                    additional_kwargs=ai_msg.additional_kwargs,
                    response_metadata=ai_msg.response_metadata,
                    id=ai_msg.id,
                )
                fixed_messages.append(fixed_msg)
            else:
                fixed_messages.append(ai_msg)

        return ChatResult(
            generations=[ChatGeneration(message=msg) for msg in fixed_messages],
            llm_output=raw_result.llm_output
        )

    @property
    def _llm_type(self) -> str:
        return "fix_tool_calls_chat_model"

    def _stream(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager=None,
            **kwargs
    ) -> Iterator:
        raise NotImplementedError("Streaming not supported")

    # ✅ 显式重写 bind_tools，返回一个新的 FixToolCallsChatModel，包装 bound 对象
    def bind_tools(self, tools, **kwargs):
        print("🔧 bind_tools 被调用了！tools =", [t.name for t in tools])
        bound_runnable = self.base_llm.bind_tools(tools, **kwargs)
        # 创建新实例，base_llm 现在是 bound_runnable（RunnableBinding）
        return FixToolCallsChatModel(base_llm=bound_runnable)

    # 可选：代理其他属性（避免某些场景下缺失）
    def __getattr__(self, name: str) -> Any:
        if name == "base_llm":
            raise AttributeError()
        return getattr(self.base_llm, name)


# ----------------------------
# 工具定义
# ----------------------------
@tool
def add(a: int, b: int) -> int:
    """加法工具：返回 a + b 的结果"""
    print(f"计算{a}+{b}")
    return a + b


TOOLS = [add]


# ----------------------------
# 初始化 LLM
# ----------------------------
def get_llm(
        model_name: str = "qwen2.5-instruct",
        base_url: str = "http://10.10.3.92:9998/v1",
        api_key: str = "dummy_key",
        temperature: float = 0.0,
        max_tokens: int = 512,
) -> ChatOpenAI:
    return ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )


# ----------------------------
# 创建 Agent
# ----------------------------
def create_tool_agent(llm, tools) -> CompiledStateGraph:
    return create_react_agent(llm, tools)


# ----------------------------
# 运行 Agent
# ----------------------------
def run_agent(agent, user_input: str):
    result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
    print(f"result: {utils.json_util.to_json(result)}")
    return result["messages"][-1].content


# ----------------------------
# 主程序
# ----------------------------
if __name__ == "__main__":
    # ChatOpenAI这个class是langchain_openai库里的
    llm:ChatOpenAI = get_llm()
    fixed_llm = FixToolCallsChatModel(base_llm=llm)
    agent: CompiledStateGraph = create_tool_agent(fixed_llm, TOOLS)

    question = "请用工具计算 123 + 456"
    print(f"👤 用户: {question}")
    answer = run_agent(agent, question)
    print(f"🤖 Agent: {answer}")
