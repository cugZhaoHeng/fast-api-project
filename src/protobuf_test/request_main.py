# server.py
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
import person_pb2
import json

app = FastAPI()

@app.post("/api/person")
async def receive_protobuf(request: Request):
    try:
        # 1. 读取二进制数据
        body = await request.body()
        
        # 2. 解析 Protobuf
        person = person_pb2.Person()
        person.ParseFromString(body)
        
        # 3. 转换为字典
        result = {
            "name": person.name,
            "age": person.age
        }
        
        print(f"接收到数据: {result}")
        
        # 4. 返回结果
        return JSONResponse(content={
            "status": "success",
            "data": result,
            "message": "Protobuf 解析成功"
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": f"解析失败: {str(e)}"
            }
        )

@app.get("/")
async def root():
    return {"message": "Protobuf API 服务已启动"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)