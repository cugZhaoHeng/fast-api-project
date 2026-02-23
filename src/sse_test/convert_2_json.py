import requests
import json

def fetch_answer(url, headers, payload):
    """
    发送流式请求，只保留最近两个数据块，遇到结束标志时返回前一个块的 data 字段。
    过程中打印每个非结束块 data 字段的长度。
    """
    prev = None      # 上一个非结束块
    current = None   # 当前非结束块
    try:
        response = requests.post(url, headers=headers, json=payload, stream=True)
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
                                print("遇到结束标志，准备返回前一个块的 data 字段")
                                # 返回前一个块的 data 字段
                                return prev.get("data") if prev else None
                            else:
                                # 非结束块，更新状态
                                prev = data
                        except json.JSONDecodeError as e:
                            print(f"JSON 解析失败: {json_str}, 错误: {e}")
    except requests.exceptions.RequestException as e:
        print(f"请求异常: {e}")
        return None

    # 如果循环结束仍未遇到结束标志（异常情况），返回最后一个非结束块的 data
    return current.get("data") if current else None

# 使用示例
if __name__ == "__main__":
    url = "http://192.168.111.139/v1/conversation/ask"
    headers = {
        "Authorization": "ImI2M2JhNGU0MDgwNjExZjE5NmMxYWU0MWUyYmRmMThjIg.aY26og.jJ8agOMBpuxU6VWVNCLIVA9XL04",
        "Connection": "keep-alive",
        "Content-Type": "application/json",
    }
    payload = {
        "kb_ids": ["cc45077707be11f1a22226047aa073c4"],
        "question": "岩石压缩性",
        "tenantId": None,
        "search_id": "8591ee1c070511f19b27f268ba6b439b"
    }

    result_data = fetch_answer(url, headers, payload)
    if result_data:
        print("\n最终返回的 data 字段内容：")
        print(json.dumps(result_data, indent=2, ensure_ascii=False))
    else:
        print("未能获取到有效数据")