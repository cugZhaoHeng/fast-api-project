import json
import uuid
import re
import os
from typing import Annotated, TypedDict, Any, List, Union
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langsmith import Client
from langsmith.run_helpers import traceable
from http_utils import LoggingHTTPTransport, LoggingSyncHTTPTransport
from loguru import logger
import httpx

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_PROJECT"] = "AI_ToolCall_Repair_Demo"

client = Client()
http_client = LoggingHTTPTransport()

# --- 1. 辅助函数：暴力清洗 JSON ---
def ensure_dict(value: Union[str, dict]) -> dict:
    """
    不管传入什么，尽最大努力把它变成字典。
    """
    if isinstance(value, dict):
        return value
    
    if not isinstance(value, str):
        print(f"⚠️ [警告] 参数类型既不是 str 也不是 dict: {type(value)}")
        return {}

    # 尝试 1: 直接解析
    try:
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
        # 如果解析出来还是字符串（双重编码），再解一次
        if isinstance(parsed, str):
            try:
                parsed_again = json.loads(parsed)
                if isinstance(parsed_again, dict):
                    return parsed_again
            except:
                pass
    except json.JSONDecodeError:
        pass

    # 尝试 2: 简单的单引号替换（针对某些不规范的本地模型）
    try:
        # 这是一个很粗糙的修复，但在 demo 中通常有效
        fixed_value = value.replace("'", '"')
        parsed = json.loads(fixed_value)
        if isinstance(parsed, dict):
            return parsed
    except:
        pass

    print(f"❌ [严重] 无法将参数转换为字典: {value}")
    return {}

# --- 2. 核弹级修复版 ChatOpenAI ---
class FixedChatOpenAI(ChatOpenAI):
    def _create_chat_result(self, response: dict, generation_info: dict = None) -> ChatResult:
        # 兼容性处理：如果 response 是对象则转 dict
        if not isinstance(response, dict) and hasattr(response, "model_dump"):
            response = response.model_dump()

        generations = []
        
        for res in response.get("choices", []):
            message_dict = res.get("message", {})
            content = message_dict.get("content", "")
            
            # 手动构建 tool_calls
            tool_calls = []
            
            if "tool_calls" in message_dict and message_dict["tool_calls"]:
                for raw_tc in message_dict["tool_calls"]:
                    # 1. 确保 ID 存在
                    tc_id = raw_tc.get("id") or f"call_{uuid.uuid4().hex[:8]}"
                    
                    # 2. 获取原始数据
                    func_name = raw_tc.get("function", {}).get("name")
                    args_raw = raw_tc.get("function", {}).get("arguments", "{}")
                    
                    # 3. 🚨 核心修复：强制转字典
                    args_parsed = ensure_dict(args_raw)
                    
                    # 调试打印（如果在控制台看到这个，说明解析成功）
                    # print(f"🔍 [Debug] 解析参数: {args_raw} -> {type(args_parsed)}")

                    tool_calls.append({
                        "name": func_name,
                        "args": args_parsed, # 这里绝对是字典
                        "id": tc_id,
                        "type": "tool_call"
                    })

            # 4. 构造消息，显式传入 invalid_tool_calls=[] 避免触发验证逻辑
            msg = AIMessage(
                content=content if content else "",
                tool_calls=tool_calls,
                invalid_tool_calls=[], 
            )

            gen_info = dict(finish_reason=res.get("finish_reason"))
            generations.append(ChatGeneration(message=msg, generation_info=gen_info))

        return ChatResult(generations=generations, llm_output=response)

# --- 3. 工具定义 ---
@tool
def get_weather(city: str):
    """查询天气"""
    return f"{city}正在下暴雨，气温10度。"

@tool
def recommend_food(weather: str):
    """根据天气推荐食物"""
    if "雨" in weather:
        return "这种天气适合吃麻辣火锅。"
    return "吃沙拉吧。"

# --- 4. 配置 ---
def get_llm():
    logger.info("Loading FixedChatOpenAI model with HTTP logging...")
    
    # 1. 创建自定义的 HTTP Transport
    transport = LoggingHTTPTransport()
    
    # 2. 使用 transport 创建 AsyncClient（关键步骤！）
    async_http_client = httpx.AsyncClient(
        transport=transport,
        timeout=60.0,
        limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
    )
    
    # 创建 HTTP 客户端
    http_client = None
    transport = LoggingSyncHTTPTransport()
    
    # 创建同步 HTTP 客户端
    http_client = httpx.Client(
        transport=transport,
        timeout=httpx.Timeout(60.0),
        limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        follow_redirects=True,
    )
    
    # 3. 创建 LLM 实例，传入 async_http_client
    llm = FixedChatOpenAI(
        model="qwen2.5-instruct",
        api_key="osfks",
        base_url="http://10.10.3.92:9998/v1",
        temperature=0.1,
        max_tokens=32768,
        http_client=async_http_client,  # ← 关键：传入 AsyncClient，不是 Transport
        timeout=60.0,
        max_retries=2,
    )
    
    logger.info(f"模型加载成功: {llm.model_name}")
    return llm

# --- 5. 节点逻辑 (保持你的逻辑) ---
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# @traceable(run_type="chain")
def weather_expert_node(state: State):
    llm = get_llm()
    # 禁用并行调用以提高稳定性
    llm_with_tools = llm.bind_tools([get_weather], parallel_tool_calls=False)
    
    print("--- Weather Agent: 正在思考 ---")
    response = llm_with_tools.invoke(state["messages"])
    
    # 构造返回列表
    messages_to_return = [response]
    
    # 内部执行
    if response.tool_calls:
        print(f"--- Weather Agent: 内部执行工具 ({len(response.tool_calls)}) ---")
        for tool_call in response.tool_calls:
            if tool_call["name"] == "get_weather":
                try:
                    tool_result = get_weather.invoke(tool_call["args"])
                    tool_msg = ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call["id"],
                        name=tool_call["name"]
                    )
                    messages_to_return.append(tool_msg)
                except Exception as e:
                    print(f"⚠️ 工具执行出错: {e}")
    
    return {"messages": messages_to_return}

# @traceable(run_type="chain")
def food_expert_node(state: State):
    llm = get_llm()
    llm_with_tools = llm.bind_tools([recommend_food], parallel_tool_calls=False)
    
    # 构建 Prompt，明确指示
    prompt = SystemMessage(content="你是美食专家。请根据对话历史中的天气信息，调用工具推荐食物。")
    print("--- Food Agent: 正在思考 ---")
    
    # 这里把 prompt 放在最前面
    inputs = [prompt] + state["messages"]
    response = llm_with_tools.invoke(inputs)
    
    messages_to_return = [response]
    
    if response.tool_calls:
        print(f"--- Food Agent: 内部执行工具 ({len(response.tool_calls)}) ---")
        for tool_call in response.tool_calls:
            if tool_call["name"] == "recommend_food":
                try:
                    tool_result = recommend_food.invoke(tool_call["args"])
                    tool_msg = ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call["id"],
                        name=tool_call["name"]
                    )
                    messages_to_return.append(tool_msg)
                except Exception as e:
                    print(f"⚠️ 工具执行出错: {e}")
                
    return {"messages": messages_to_return}

# --- 6. 运行 ---
workflow = StateGraph(State)
workflow.add_node("weather_expert", weather_expert_node)
workflow.add_node("food_expert", food_expert_node)
workflow.add_edge(START, "weather_expert")
workflow.add_edge("weather_expert", "food_expert")
workflow.add_edge("food_expert", END)

app = workflow.compile()

if __name__ == "__main__":
    # --- 新增：生成流程图代码 ---
    try:
        # 获取图的 Mermaid 格式数据
        graph_png = app.get_graph().draw_mermaid_png()
        
        # 将二进制数据写入图片文件
        with open("agent_workflow.png", "wb") as f:
            f.write(graph_png)
        print("✅ 流程图已保存为 agent_workflow.png")
    except Exception as e:
        print(f"⚠️ 绘图失败 (可能需要安装依赖: pip install pygraphviz): {e}")
        # 如果生成图片失败，打印 Mermaid 文本，你可以复制到在线编辑器
        print("可以直接复制以下文本到 https://mermaid.live/ 查看:")
        print(app.get_graph().draw_mermaid())
    try:
        print(">>> 开始执行工作流...")
        result = app.invoke({"messages": [HumanMessage(content="查一下上海的天气，然后告诉我吃什么")]})
        
        print("\n====== 最终结果 ======")
        for msg in result["messages"]:
            if isinstance(msg, ToolMessage):
                print(f"🔧 [工具结果]: {msg.content}")
            elif isinstance(msg, AIMessage):
                if msg.tool_calls:
                    print(f"🤖 [AI ToolCall]: {msg.tool_calls[0]['name']} -> {msg.tool_calls[0]['args']}")
                else:
                    print(f"🤖 [AI Reply]: {msg.content}")
            else:
                print(f"👤 [User]: {msg.content}")
                
    except Exception as e:
        print(f"\n❌ 程序崩溃: {e}")
        import traceback
        traceback.print_exc()