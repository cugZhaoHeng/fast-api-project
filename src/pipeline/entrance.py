import json
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, Response

import sensor_history_pb2
import main_api
from utils.logger import create_logger

logger = create_logger(__name__)

class CustomError(Exception):
    def __init__(self, status, message):
        self.status = status
        self.message = message

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Hello from FastAPI!", "time": get_current()}

import json
from fastapi import FastAPI, Response
import logging
import sensor_history_pb2  # 确保已生成

app = FastAPI()
logger = logging.getLogger(__name__)

@app.get("/get_full_sensor_data", response_class=Response)
def get_full_sensor_data():
    # 从本地文件读取 JSON
    try:
        with open('data/response.json', 'r', encoding='utf-8') as f:
            a = json.load(f)
            logger.info("成功读取 JSON 数据", )
    except FileNotFoundError:
        logger.error("文件未找到: data/response.json")
        return Response(content="", status_code=404, media_type="application/json")
    except json.JSONDecodeError as e:
        logger.error("JSON 解析错误: %s", e)
        return Response(content="", status_code=500, media_type="application/json")
    except Exception as e:
        logger.error("读取文件时发生未知错误: %s", e)
        return Response(content="", status_code=500, media_type="application/json")

    # 创建 Protobuf 响应对象
    response = sensor_history_pb2.SensorData()

    # 设置 sensors
    if 'sensors' in a:
        response.sensors.extend(a['sensors'])
    else:
        logger.warning("JSON 中缺少 'sensors' 字段")

    # 添加 datas
    if 'datas' in a:
        for record in a['datas']:  # 遍历 datas 列表中的每个记录
            data_record = response.datas.add()  # 创建一个新的 DataRecord
            data_record.time = record.get('time', 0)  # 获取时间戳

            # 添加 data 点
            for point in record.get('data', []):  # 遍历 data 列表
                data_point = data_record.data.add()  # 创建一个新的 DataPoint
                data_point.index = point.get('index', 0)
                data_point.val = float(point.get('val', 0.0))  # 确保是 float
    else:
        logger.warning("JSON 中缺少 'datas' 字段")

    # 序列化为二进制
    try:
        serialized_data = response.SerializeToString()
    except Exception as e:
        logger.error("Protobuf 序列化失败: %s", e)
        return Response(content="", status_code=500, media_type="application/json")

    # 返回 Protobuf 数据
    return Response(
        content=serialized_data,
        media_type="application/x-protobuf"
    )
# 这行一定要放在上面，至少要在通用拦截之前
app.include_router(main_api.router)



# 设置允许的源、方法和头部信息
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 允许的源列表
    allow_credentials=True,  # 是否允许携带凭证（如 cookie）
    allow_methods=["*"],  # 允许的方法列表，'*' 表示所有方法
    allow_headers=["*"],  # 允许的头部列表，'*' 表示所有头部
)


@app.exception_handler(CustomError)
async def custom_error_handler(request, exc):
    return JSONResponse(
        status_code=exc.status,
        content=vars(exc)
    )





def get_current():
    """
    获取当前时间
    @return: yyyy-MM-dd HH:mm:ss 时间字符串
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=6112)
