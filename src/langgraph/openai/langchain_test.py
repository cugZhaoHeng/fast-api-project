from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from loguru import logger

# llm = ChatOpenAI(model="qwen2.5-instruct",
#                  temperature=0.1,
#                  max_tokens=32768,
#                  api_key="dummy",
#                  base_url="http://10.10.3.92:9998/v1")
# response = llm.invoke([HumanMessage(content="你好，请介绍一下你自己"), SystemMessage(content="你是一个有帮助的助手， 请你从专业的角度来介绍你自己。")])
# logger.info(f"Response: {response.content}")

@tool
def get_weather(city: str) -> str:
    """获取城市天气"""
    return f"{city}的天气是晴天，25°C"

@tool  
def search_restaurant(city: str, cuisine: str) -> str:
    """搜索餐厅"""
    return f"{city}的{cuisine}餐厅：XXX餐厅"

def langchain_simple_call():
    llm: ChatOpenAI = ChatOpenAI(model="qwen2.5-instruct",
                                 base_url="http://10.10.3.92:9998/v1", api_key="dummy", temperature=0)
    response = llm.invoke([HumanMessage(content="武汉市今天的天气怎么样？")])
    logger.info(f"LangChain Simple Call Response: {response.content}")

def langchain_tool_call():
    llm = ChatOpenAI(
        model="qwen2.5-instruct",
        base_url="http://10.10.3.92:9998/v1",
        api_key="osfks"
    )
    # 这里在执行了bind_tools之后，一定要使用返回的 llm 实例进行调用
    llm = llm.bind_tools([get_weather, search_restaurant])
    response = llm.invoke([HumanMessage(content="武汉市今天的天气怎么样？")])
    logger.info(f"LangChain Tool Call Response: {response}")
    logger.info(f"LangChain Tool Call Response: {response.content}")

if __name__ == "__main__":
    # langchain_simple_call()
    langchain_tool_call()

