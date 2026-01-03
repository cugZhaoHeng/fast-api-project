# 示例：使用 LoggingLLMWrapper
from llm_utils_with_callbacks import load_chat_model_with_callbacks
from logging_llm_wrapper import LoggingLLMWrapper
from langchain_core.messages import AIMessage, BaseMessage
from langgraph_with_logging import create_logging_llm_node, AgentState, print_call_summary
from loguru import logger
from typing import List, TypedDict
from datetime import datetime

# 加载基础 LLM
base_llm = load_chat_model_with_callbacks()

# 用包装器包装
llm_with_logging = LoggingLLMWrapper(base_llm, log_level="DEBUG")

# 现在使用 llm_with_logging，所有调用都会被记录
response = llm_with_logging.invoke("你好，请介绍一下你自己")
print(f"响应: {response.content}")

# 在 LangGraph 中使用
from langgraph.graph import StateGraph, END

# 定义状态
class AgentState(TypedDict):
    messages: List[BaseMessage]
    llm_calls: List[dict]

# 创建带日志的节点
llm_node = create_logging_llm_node(llm_with_logging, "assistant_node")

# 构建工作流
workflow = StateGraph(AgentState)
workflow.add_node("assistant", llm_node)
workflow.set_entry_point("assistant")
workflow.add_edge("assistant", END)

app = workflow.compile()

# 执行
result = app.invoke({
    "messages": [{"role": "user", "content": "你好，请帮忙写一首诗"}],
    "llm_calls": []
})

# 打印摘要
print_call_summary(result)