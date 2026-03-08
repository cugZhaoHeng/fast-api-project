import requests
import json

def fetch_answer(url, headers, payload):
    """
    发送流式请求，只保留最近两个数据块，遇到结束标志时返回前一个块的 data 字段。
    过程中打印每个非结束块 data 字段的长度。
    """
    # 累积答案的变量
    full_answer = ""

    try:
        # 必须开启 stream=True 才能实时接收流式数据
        response = requests.post(url, headers=headers, data=payload, stream=True)
        print(response.text)

        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data:'):
                    json_str = line_str[5:].strip()
                    if json_str:
                        try:
                            data = json.loads(json_str)
                            # 判断是否为结束块（code=0 且 data=True）
                            if data.get("code") == 0 and data.get("data") is True:
                                print(f"结束标志到达，完整答案长度: {len(full_answer)}")
                                return full_answer
                            else:
                                # 非结束块，提取 answer 并拼接
                                answer_part = data.get("data", {}).get("answer", "")
                                if answer_part:
                                    full_answer += answer_part
                                    # 可选：打印当前累积长度
                                    print(f"当前累积答案长度: {len(full_answer)}")
                        except json.JSONDecodeError as e:
                            print(f"JSON 解析失败: {json_str}, 错误: {e}")
    except requests.exceptions.RequestException as e:
        print(f"请求异常: {e}")
        return None

# 使用示例
if __name__ == "__main__":
    url = "http://10.10.3.93:9383/v1/conversation/ask"

    # 构建请求体（可根据需要参数化）
    payload = json.dumps({
        "kb_ids": ["3470f09415ff11f1a2b401b5bb47c022"],
        "question": "气体高压物性参数表，天然气在不同压力、温度下的物性参数，一般是一个多行多列的表格，列有压力，体积系数，粘度",
        "tenantId": None,
        "search_id": "55fbae44160811f1a2b401b5bb47c022"
    })

    headers = {
        'Accept': '*/*',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'Authorization': 'IjNiYzI2NGZhMTYwODExZjFhMmI0MDFiNWJiNDdjMDIyIg.aaU6JA.9I5IS6PjubZUqNj2DB1GHEKp0u8',
        'Connection': 'keep-alive',
        'Content-Type': 'application/json',
        'Origin': 'http://10.10.3.93:9383',
        'Referer': 'http://10.10.3.93:9383/next-search/55fbae44160811f1a2b401b5bb47c022?page=1',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36',
        'Cookie': 'session=.eJxtyzsOgCAMANC7dHZoCxbCZYjFEl1BJuPd_cyOb3gn5NGt5X2FBIzRSQ1GgkJUaWH1SDqr-lCQGSbItVnfIB1t2KOvOS0swb0t_rbrBtNKHFA.aaU6JA.vs8rlbi1v6MaQ_srXwhNBJOiNxw'
    }

    result_data = fetch_answer(url, headers, payload)
    if result_data:
        print("\n最终返回的 data 字段内容：")
        print(json.dumps(result_data, indent=2, ensure_ascii=False))
    else:
        print("未能获取到有效数据")