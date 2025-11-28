from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model

from utils.json_util import to_json


@tool( description="加法工具，计算 a + b")
def add(a: int, b: int) -> int:
    return a + b

model = init_chat_model(
        model='qwen2.5-instruct',
        model_provider='openai',
        base_url="http://10.10.3.92:9998/v1",
        api_key="dummy_key",
        temperature=0.0,
        timeout=None,
    )


agent = create_react_agent(model, [add])  # 会在需要时自动执行 add

result = agent.invoke({"messages": [{"role": "user", "content": "先用工具算 2+3，再用工具算 5+3"}]})
print(to_json(result))
# messages = result["messages"]
# print(len(messages))
# print(messages[0])
# print(messages[1])
# print(messages[2])
# print(messages[3])
# print(messages[4])