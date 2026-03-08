from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from loguru import logger

@tool
def get_weather(city: str) -> str:
    """获取城市天气"""
    return f"{city}的天气是晴天，25°C"

@tool  
def search_restaurant(city: str, cuisine: str) -> str:
    """搜索餐厅"""
    return f"{city}的{cuisine}餐厅：XXX餐厅"

# 方式1：简单调用
def langchain_simple_call():
    llm = ChatOpenAI(
        model="qwen2.5-instruct",
        base_url="http://10.10.3.92:9998/v1",
        api_key="osfks"
    )
    
    # 带工具的调用
    llm_with_tools = llm.bind_tools([get_weather, search_restaurant])
    response = llm_with_tools.invoke("北京天气怎么样？")
    logger.info(f"LangChain Simple Call Response: {response.content}")
    if response.tool_calls:
        for tool_call in response.tool_calls:
            print(f"需要调用工具: {tool_call['name']}")
            print(f"参数: {tool_call['args']}")
    
    return response

# 方式2：创建链
def langchain_chain_call():
    llm = ChatOpenAI(
        model="qwen2.5-instruct",
        base_url="http://10.10.3.92:9998/v1",
        api_key="osfks"
    )
    
    # 创建链：prompt -> llm -> parser
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个有用的助手"),
        ("user", "{input}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    # 执行链
    result = chain.invoke({"input": "今天的天气怎么样？"})
    return result

# 特点：
# - 统一的LLM接口
# - 可以绑定工具
# - 可以创建处理链（多个步骤）
# - 有基本的状态管理（通过消息列表）
# - 支持输出解析
if __name__ == "__main__":
    print("=== LangChain 简单调用 ===")
    langchain_simple_call()