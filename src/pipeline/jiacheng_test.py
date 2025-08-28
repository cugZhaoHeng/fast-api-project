# client_protobuf.py
import json

import requests
import sensor_history_pb2
from google.protobuf.json_format import MessageToDict
from logger import create_logger

logger = create_logger(__name__)
# API 地址
url = "http://192.168.111.247:9091/baidu/pipeline/api/sensor/get_sensor_data_by_day_buffer"

# 请求头
headers = {
    "Content-Type": "application/json"
}

# 请求体（JSON 格式）
payload = {
    "beginTime": 0,
    "day": "2025-08-27",
    "endTime": 0
}

# 发送 POST 请求
response = requests.post(url, json=payload, headers=headers)

# 检查状态码
if response.status_code == 200:
    # 假设返回的是 Protobuf 二进制数据（application/x-protobuf）
    print(f"Response raw length: {len(response.content)} bytes")

    # 创建 Protobuf 消息对象
    data = sensor_history_pb2.SensorData()

    try:
        # 解析二进制数据
        data.ParseFromString(response.content)
        print("✅ Protobuf parsing succeeded.")

        # 转为字典（类似 JSON）
        data_dict = MessageToDict(data, preserving_proto_field_name=True)
        logger.info(f"{json.dumps(data_dict, ensure_ascii=False)}")

    except Exception as e:
        print("❌ Failed to parse Protobuf:", e)
        print("Maybe the server returned JSON instead of Protobuf?")
        # 可以尝试打印原始响应看看
        try:
            print("Raw JSON response:", response.json())
        except:
            print("Raw text:", response.text)

else:
    print("❌ HTTP Error:", response.status_code, response.text)