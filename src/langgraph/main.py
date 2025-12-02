import json
import uuid
from typing import Annotated, TypedDict, Union, List, Literal

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# 使用 LangGraph 基础组件，不依赖不稳定的 prebuilt 函数
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode

# --- 1. 核弹级修复版 ChatOpenAI (保留，用于解决本地模型格式问题) ---
def ensure_dict(value: Union[str, dict]) -> dict:
    if isinstance(value, dict): return value
    try:
        return json.loads(value)
    except:
        return {}

class FixedChatOpenAI(ChatOpenAI):
    def _create_chat_result(self, response: dict, generation_info: dict = None) -> ChatResult:
        if not isinstance(response, dict) and hasattr(response, "model_dump"):
            response = response.model_dump()
        generations = []
        for res in response.get("choices", []):
            message_dict = res.get("message", {})
            content = message_dict.get("content", "")
            tool_calls = []
            if "tool_calls" in message_dict and message_dict["tool_calls"]:
                for raw_tc in message_dict["tool_calls"]:
                    tc_id = raw_tc.get("id") or f"call_{uuid.uuid4().hex[:8]}"
                    args_parsed = ensure_dict(raw_tc.get("function", {}).get("arguments", "{}"))
                    tool_calls.append({
                        "name": raw_tc.get("function", {}).get("name"),
                        "args": args_parsed,
                        "id": tc_id,
                        "type": "tool_call"
                    })
            msg = AIMessage(content=content if content else "", tool_calls=tool_calls, invalid_tool_calls=[])
            generations.append(ChatGeneration(message=msg, generation_info=dict(finish_reason=res.get("finish_reason"))))
        return ChatResult(generations=generations, llm_output=response)

# --- 2. 工具定义 ---
@tool
def get_weather(city: str):
    """查询城市天气"""
    return f"{city}正在下大暴雨，气温8度。"

@tool
def recommend_food(weather: str):
    """根据天气推荐食物"""
    if "雨" in weather:
        return "这种天气必须吃麻辣火锅！"
    return "吃点清淡的吧。"

# --- 3. 核心：自定义 Agent 工厂函数 (替代不稳定的 create_react_agent) ---
def create_custom_agent(llm, tools, system_prompt: str):
    """
    手动构建一个 ReAct 循环子图。
    这比依赖版本变来变去的 create_react_agent 更稳定。
    """
    # 定义 Agent 的状态
    class AgentState(TypedDict):
        messages: Annotated[List[BaseMessage], "add_messages"] # 这里的 reducer 取决于你的 langgraph 版本，通常 MessagesState 自带
    
    # 1. 定义调用模型的节点
    def call_model(state):
        messages = state["messages"]
        # 在这里手动注入 System Prompt，最稳健的方式
        if system_prompt:
            # 检查第一条是不是 SystemMessage，不是则插入，防止重复（简单起见直接拼装）
            # 这里的逻辑是：每次调用模型前，强制把 SystemMessage 放在最前面给模型看
            # 但不一定非要存入 State 历史中，只在 invoke 时拼接即可
            messages = [SystemMessage(content=system_prompt)] + messages
        
        # 绑定工具
        llm_with_tools = llm.bind_tools(tools)
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # 2. 定义工具执行节点
    tool_node = ToolNode(tools)

    # 3. 定义路由逻辑
    def should_continue(state):
        last_message = state["messages"][-1]
        # 如果有 tool_calls，继续去 tools 节点
        if last_message.tool_calls:
            return "tools"
        # 否则结束
        return END

    # 4. 构建子图
    workflow = StateGraph(MessagesState) # 使用通用的 MessagesState
    
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "agent")
    
    # 条件边：Agent -> (Tools 或 End)
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        ["tools", END]
    )
    
    # 循环边：Tools -> Agent
    workflow.add_edge("tools", "agent")

    return workflow.compile()

# --- 4. 初始化 LLM ---
fixed_llm = FixedChatOpenAI(
    model="qwen2.5-instruct",
    base_url="http://10.10.3.92:9998/v1",
    api_key="dummy",
    temperature=0.0,
)

# --- 5. 创建两个独立的 Agent ---
# 此时 create_custom_agent 返回的是一个 CompiledGraph，可以直接作为节点

weather_agent = create_custom_agent(
    fixed_llm, 
    [get_weather], 
    system_prompt="你是一个天气助手。查询天气后，请直接回复天气情况，不要说多余的话。"
)

food_agent = create_custom_agent(
    fixed_llm, 
    [recommend_food], 
    system_prompt="你是一个美食助手。请阅读上下文中的天气情况，调用工具推荐食物。"
)

# --- 6. 构建主工作流 ---

main_workflow = StateGraph(MessagesState)

# 将两个编译好的 Agent 子图作为节点加入
main_workflow.add_node("weather_expert", weather_agent)
main_workflow.add_node("food_expert", food_agent)

# 串联
main_workflow.add_edge(START, "weather_expert")
main_workflow.add_edge("weather_expert", "food_expert")
main_workflow.add_edge("food_expert", END)

app = main_workflow.compile()

# --- 7. 运行 ---
if __name__ == "__main__":
    print(">>> 开始执行工作流...")
    
    inputs = {"messages": [HumanMessage(content="查一下杭州的天气，然后告诉我该吃什么")]}
    
    try:
        # stream_mode="values" 会打印整个状态的所有消息
        for event in app.stream(inputs, stream_mode="values"):
            # 取出最后一条消息查看进度
            if "messages" in event and event["messages"]:
                message = event["messages"][-1]
                
                if isinstance(message, ToolMessage):
                    print(f"🔧 [工具结果]: {message.content}")
                elif isinstance(message, AIMessage):
                    if message.tool_calls:
                        print(f"🤖 [AI 呼叫工具]: {message.tool_calls[0]['name']}")
                    else:
                        print(f"🤖 [AI 回复]: {message.content}")
                        
    except Exception as e:
        print(f"❌ 出错: {e}")
        import traceback
        traceback.print_exc()