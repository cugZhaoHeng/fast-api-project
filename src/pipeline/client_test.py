# client_test.py
import json

import requests
import user_pb2
from logger import create_logger


logger = create_logger(__name__)

def message_to_dict_with_defaults(msg):
    result = {}
    # 获取所有字段（包括未显式设置的）
    for field in msg.DESCRIPTOR.fields:
        key = field.name
        if field.label == field.LABEL_REPEATED:
            value = [message_to_dict_with_defaults(item) for item in getattr(msg, key)]
        elif field.type == field.TYPE_MESSAGE:
            value = message_to_dict_with_defaults(getattr(msg, key))
        else:
            # 关键：即使字段是默认值，也输出
            value = getattr(msg, key)
        result[key] = value
    return result

# 发起 GET 请求
response = requests.get("http://127.0.0.1:6112/get_user")

if response.status_code == 200:
    # 解析返回的 Protobuf 二进制数据
    user = user_pb2.User()
    user.ParseFromString(response.content)

    print("User Info:")
    print(f"ID: {user.id}")
    print(f"Name: {user.name}")
    print(f"Email: {user.email}")
else:
    print(f"Error: {response.status_code}")

# client.py
import requests
import sensor_data_pb2
import sensor_history_pb2
from google.protobuf.json_format import MessageToDict, MessageToJson

response = requests.get("http://127.0.0.1:6112/sensor_data")

if response.status_code == 200:
    data = sensor_data_pb2.SensorDataResponse()
    # data = sensor_history_pb2.SensorData()
    data.ParseFromString(response.content)

    print(f"data 类型：{type(data)}")

    # 方式 1: 转为 Python 字典（推荐，便于操作）
    data_dict = message_to_dict_with_defaults(data)
    print("\n=== 转为 Python dict ===")
    print(data_dict)

    # 方式 2: 转为 JSON 字符串（便于打印或传输）
    data_json = MessageToJson(data, preserving_proto_field_name=True, indent=2)
    print("\n=== 转为 JSON 字符串 ===")
    print(data_json)
else:
    print("Error:", response.status_code)
