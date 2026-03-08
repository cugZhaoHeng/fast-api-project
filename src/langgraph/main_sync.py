# main_sync.py
from llm_utils_sync import default_llm_sync
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
import time

def test_sync_llm():
    """测试同步 LLM 调用"""
    print("=== 测试同步 LLM 调用 ===")
    
    # 方式1：使用默认实例
    llm = default_llm_sync

    
    # 直接调用
    response = llm.invoke("你好，请介绍一下你自己")
    print(f"直接调用结果: {response.content}")
    
    # 使用 prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个友好的助手"),
        ("human", "{input}")
    ])
    
    chain = prompt | llm
    
    # 同步调用链
    result = chain.invoke({"input": "今天的天气怎么样？"})
    print(f"链式调用结果: {result.content}")
    
    # 批量调用
    # messages_list = [
    #     [HumanMessage(content="第一个问题")],
    #     [HumanMessage(content="第二个问题")],
    # ]
    # batch_results = llm.batch(messages_list)
    # for i, result in enumerate(batch_results):
    #     print(f"批量结果 {i+1}: {result.content}")

def test_with_tools():
    """测试带工具的同步调用"""
    print("\n=== 测试带工具的同步调用 ===")
    
    llm = default_llm_sync
    
    # 定义工具
    from langchain_core.tools import tool
    
    @tool
    def get_weather(city: str) -> str:
        """获取城市天气"""
        return f"{city}的天气是晴天，25°C"
    
    @tool
    def calculate(expression: str) -> str:
        """计算数学表达式"""
        try:
            return str(eval(expression))
        except:
            return "无法计算该表达式"
    
    # 绑定工具到 LLM
    llm_with_tools = llm.bind_tools([get_weather, calculate])
    
    # 调用带工具的 LLM
    response = llm_with_tools.invoke("北京和上海的天气如何？")
    print(f"带工具的响应: {response}")
    
    # 检查是否有工具调用
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f"检测到工具调用: {len(response.tool_calls)} 个")
        for i, tool_call in enumerate(response.tool_calls):
            print(f"  工具 {i+1}: {tool_call.get('name')}")
            print(f"  参数: {tool_call.get('args')}")

if __name__ == "__main__":
    # 设置日志级别
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 测试
    test_sync_llm()
    test_with_tools()