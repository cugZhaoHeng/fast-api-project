import openai
from loguru import logger



def use_openai_sdk():
    client = openai.OpenAI(api_key="dummy", base_url="http://10.10.3.92:9998/v1")
    response = client.chat.completions.create(model="qwen2.5-instruct",
                                       messages=[{"role": "user", "content": "你好，请介绍一下你自己"}])
    logger.info(f"Response: {response.choices[0].message.content}")

# 对于普通的OpenAI调用，在已经知道了url和token的情况下，甚至不需要使用OpenAI，直接使用 requests 库也可以
import requests

def direct_openai_call():
    url = "http://10.10.3.92:9998/v1/chat/completions"
    headers = {
        "Authorization": "dummy",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "qwen2.5-instruct",
        "messages": [
            {"role": "user", "content": "你好，请介绍一下你自己"}
        ],
        "temperature": 0.1,
        "max_tokens": 32768
    }
    response = requests.post(url=url, headers=headers, json=payload)
    result = response.json()
    logger.info(f"Direct call response: {result['choices'][0]['message']['content']}")

if __name__ == "__main__":
    direct_openai_call()