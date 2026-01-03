# main.py
from llm_utils import default_llm
from langchain_core.messages import HumanMessage

# 简单调用
def simple_llm_call():
    print("=== 简单 LLM 调用 ===")
    response = default_llm.invoke("你好，请介绍一下你自己")
    print(f"响应: {response.content}")


# 在 LangGraph 中使用
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

class AgentState(TypedDict):
    messages: List[HumanMessage]

def create_simple_agent():
    """创建简单的 agent"""
    
    def llm_node(state: AgentState):
        """LLM 节点"""
        print(f"Agent 正在处理消息...")
        
        # 这里会触发 LLM 调用，自动记录日志
        response = default_llm.invoke(state["messages"])
        
        # 将响应添加到消息列表
        state["messages"].append(response)
        return {"messages": state["messages"]}
    
    # 构建图
    workflow = StateGraph(AgentState)
    workflow.add_node("llm_processor", llm_node)
    workflow.set_entry_point("llm_processor")
    workflow.add_edge("llm_processor", END)
    
    return workflow.compile()


if __name__ == "__main__":
    # 测试简单调用
    simple_llm_call()
    
    print("\n" + "="*50 + "\n")
    
    # 测试在 LangGraph 中使用
    print("=== 在 LangGraph 中使用 ===")
    app = create_simple_agent()
    
    result = app.invoke({
        "messages": [HumanMessage(content="你好，请帮忙写一首关于春天的诗")]
    })
    
    print(f"最终结果: {result['messages'][-1].content}")