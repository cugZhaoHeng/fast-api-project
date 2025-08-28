from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, Response

import user_pb2
import sensor_data_pb2

import main_api


class CustomError(Exception):
    def __init__(self, status, message):
        self.status = status
        self.message = message

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Hello from FastAPI!", "time": get_current()}

@app.get("/get_user")
def get_user():
    # 构造一个用户对象
    user = user_pb2.User()
    user.id = 1001
    user.name = "Alice"
    user.email = "alice@example.com"

    # 序列化为二进制
    serialized_data = user.SerializeToString()

    # 返回 Protobuf 二进制数据
    return Response(
        content=serialized_data,
        media_type="application/x-protobuf"
    )

@app.get("/sensor_data", response_class=Response)
def get_sensor_data():
    # 创建响应对象
    response = sensor_data_pb2.SensorDataResponse()

    # 设置 sensors
    response.sensors.extend(["-1", "0", "AJQT01SP001PT903A"])

    # 添加 datas
    for time_val, values in [
        (1756260842, [(0, 8.0), (1, 10.0)]),
        (1756260837, [(0, 3.0), (1, 5.0)])
    ]:
        data_point = response.datas.add()
        data_point.time = time_val
        for idx, val in values:
            sv = data_point.data.add()
            sv.index = idx
            sv.val = val

    # 序列化为二进制
    serialized_data = response.SerializeToString()

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
