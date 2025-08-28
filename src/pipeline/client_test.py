# client_test.py
from utils.logger import create_logger

logger = create_logger(__name__)

# client.py
import requests

import sensor_history_pb2
from google.protobuf.json_format import MessageToDict, MessageToJson

response = requests.get("http://127.0.0.1:6112/get_full_sensor_data")

if response.status_code == 200:
    data = sensor_history_pb2.SensorData()
    # data = sensor_history_pb2.SensorData()
    data.ParseFromString(response.content)

    logger.info(f"data 类型：{type(data)}")

    # 方式 1: 转为 Python 字典（推荐，便于操作）
    data_dict = MessageToDict(data)
    logger.info("\n=== 转为 Python dict ===")
    logger.info(data_dict)

    # 方式 2: 转为 JSON 字符串（便于打印或传输）
    data_json = MessageToJson(data, preserving_proto_field_name=True, indent=2)
    print("\n=== 转为 JSON 字符串 ===")
    print(data_json)
else:
    print("Error:", response.status_code)
