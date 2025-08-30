import requests
import sensor_history_pb2
from google.protobuf.json_format import MessageToJson
from utils.logger import create_logger

def get_sensor_history(day: str = "2025-08-27", beginTime: int = 0, endTime: int = 0) -> sensor_history_pb2.SensorHistory:
    '''
    该函数的作用，是从张嘉诚的接口 get_sensor_data_by_day_buffer 中，获取历史数据
    '''
    logger = create_logger(__name__)
    # API 地址
    url = "http://192.168.111.247:9091/baidu/pipeline/api/sensor/get_sensor_data_by_day_buffer"

    # 请求头
    headers = {
        "Content-Type": "application/json; charset=utf-8"
    }

    body = {
        "day": day,
        "beginTime": beginTime,
        "endTime": endTime
    }
    # 请求体（JSON 格式）

    # 发送 POST 请求
    logger.info(f"开始访问URL： {url}")
    response = requests.post(url, json=body, headers=headers)
    logger.info(response)
    # 检查状态码
    if response.status_code == 200:
        # 假设返回的是 Protobuf 二进制数据（application/x-protobuf）
        logger.info(f"Response raw length: {len(response.content)} bytes")
        # 创建 Protobuf 消息对象
        data = sensor_history_pb2.SensorData()
        try:
            # 解析二进制数据
            data.ParseFromString(response.content)
            logger.info("Protobuf parsing succeeded.")

            # 转为JSON（类似 JSON）
            data_dict = MessageToJson(data, preserving_proto_field_name=True)
            # logger.info(f"{data_dict}")
            return data_dict

        except Exception as e:
            logger.info("Failed to parse Protobuf:", e)
            logger.info("Maybe the server returned JSON instead of Protobuf?")
            # 可以尝试打印原始响应看看
            try:
                logger.info("Raw JSON response:", response.json())
            except:
                logger.info("Raw text:", response.text)

    else:
        logger.info("HTTP Error:", response.status_code, response.text)