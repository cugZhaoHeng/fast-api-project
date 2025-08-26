from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

import main_api


class CustomError(Exception):
    def __init__(self, status, message):
        self.status = status
        self.message = message

app = FastAPI()
# 这行一定要放在上面，至少要在通用拦截之前
app.include_router(main_api.router)

app.get("/")
def home():
    return {"message": "Hello from FastAPI!", "time": get_current()}

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
