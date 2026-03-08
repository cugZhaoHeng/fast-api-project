from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from typing import TypedDict, List, Annotated
from langchain_core.tools import tool
import operator

# 定义工具
@tool
def get_weather(city: str) -> str:
    """获取城市天气"""
    return f"{city}的天气是晴天，25°C"

@tool
def search_restaurant(city: str, cuisine: str) -> str:
    """搜索餐厅"""
    return f"{city}的{cuisine}餐厅：XXX餐厅"

# 定义状态
class AgentState(TypedDict):
    messages: Annotated[List, operator.add]  # 自动累积消息
    current_city: str
    final_answer: str

# 创建不同的专家节点
def weather_expert(state: AgentState):
    """天气专家节点"""
    from langchain_openai import ChatOpenAI
    
    llm = ChatOpenAI(
        model="qwen2.5-instruct",
        base_url="http://10.10.3.92:9998/v1",
        api_key="osfks"
    )
    
    # 获取用户最后一条消息
    last_message = state["messages"][-1]
    city = state.get("current_city", "北京")
    
    # 创建专门针对天气的提示词
    prompt = f"""你是一个天气专家，请回答关于{city}天气的问题。
    用户问题：{last_message.content}
    请提供详细、专业的天气信息。"""
    
    response = llm.invoke(prompt)
    
    # 更新状态
    return {
        "messages": [response],
        "current_city": city,
        "final_answer": f"天气信息：{response.content}"
    }

def restaurant_expert(state: AgentState):
    """餐厅专家节点"""
    from langchain_openai import ChatOpenAI
    
    llm = ChatOpenAI(
        model="qwen2.5-instruct",
        base_url="http://10.10.3.92:9998/v1",
        api_key="osfks"
    )
    
    # 绑定工具
    llm_with_tools = llm.bind_tools([search_restaurant])
    
    # 获取天气专家的结果
    weather_info = state.get("final_answer", "")
    city = state.get("current_city", "北京")
    
    prompt = f"""你是一个餐厅推荐专家。
    基于以下信息：
    1. 城市：{city}
    2. 天气情况：{weather_info}
    
    请推荐适合当前天气的餐厅。"""
    
    response = llm_with_tools.invoke(prompt)
    
    return {
        "messages": [response],
        "final_answer": f"{weather_info}\n餐厅推荐：{response.content}"
    }

# 创建路由函数
def route_messages(state: AgentState) -> str:
    """根据消息内容路由到不同的专家"""
    last_message = state["messages"][-1]
    
    if "天气" in last_message.content:
        return "weather_expert"
    elif any(keyword in last_message.content for keyword in ["餐厅", "吃饭", "美食"]):
        return "restaurant_expert"
    else:
        return "general_assistant"

def general_assistant(state: AgentState):
    """通用助手节点"""
    from langchain_openai import ChatOpenAI
    
    llm = ChatOpenAI(
        model="qwen2.5-instruct",
        base_url="http://10.10.3.92:9998/v1",
        api_key="osfks"
    )
    
    response = llm.invoke(state["messages"])
    
    return {"messages": [response]}

# 构建图
def create_agent_workflow():
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("weather_expert", weather_expert)
    workflow.add_node("restaurant_expert", restaurant_expert)
    workflow.add_node("general_assistant", general_assistant)
    
    # 设置入口点和条件路由
    workflow.set_entry_point("general_assistant")
    
    # 添加条件边
    workflow.add_conditional_edges(
        "general_assistant",
        route_messages,
        {
            "weather_expert": "weather_expert",
            "restaurant_expert": "restaurant_expert",
            "general_assistant": END  # 如果还是通用问题，结束
        }
    )
    
    # 添加普通边
    workflow.add_edge("weather_expert", "restaurant_expert")
    workflow.add_edge("restaurant_expert", END)
    
    return workflow.compile()

# 使用工作流
def langgraph_workflow_call():
    app = create_agent_workflow()
    
    # 执行复杂的工作流
    result = app.invoke({
        "messages": [HumanMessage(content="查一下北京的天气，然后推荐适合的餐厅")],
        "current_city": "北京",
        "final_answer": ""
    })
    
    print(f"最终结果: {result['final_answer']}")
    print(f"所有消息: {result['messages']}")
    
    return result

if __name__ == "__main__":
    langgraph_workflow_call()