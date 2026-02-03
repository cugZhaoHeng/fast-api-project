import uvicorn
from fastapi import FastAPI, Request, Response
from google.protobuf.json_format import MessageToDict, ParseDict
import pipeline_model_pb2  # 你的 proto 编译文件

app = FastAPI(
    title="Pipeline Model API",
    description="Protobuf 闭环测试：接收Proto -> 转Dict处理 -> 转回Proto返回",
    version="1.0.0"
)

@app.post(
    "/api/model/process",
    summary="处理并返回 Protobuf 数据",
    description="接收二进制流，解析为字典进行业务处理后，重新封装为二进制流返回。",
    # Swagger 文档配置：声明返回类型为 protobuf
    responses={
        200: {
            "description": "成功处理，返回 Protobuf 二进制流",
            "content": {
                "application/x-protobuf": {
                    "schema": {"type": "string", "format": "binary"}
                }
            }
        }
    }
)
async def process_model_param(request: Request):
    try:
        # Step 1: 接收原始二进制数据
        req_bytes = await request.body()
        
        # Step 2: 二进制 -> Protobuf 对象
        req_proto = pipeline_model_pb2.ModelFullParam()
        req_proto.ParseFromString(req_bytes)
        
        # Step 3: Protobuf 对象 -> Python 字典 (data_dict)
        # preserving_proto_field_name=True: 保持 snake_case (如 input_key)
        # including_default_value_fields=True: 即使字段是默认值(0或空)也保留在字典里
        data_dict = MessageToDict(
            req_proto, 
            preserving_proto_field_name=True
        )
        
        print(f"【Server】接收到的字典数据 (req_type): {data_dict.get('req_type')}")

        # ==========================================
        # Step 4: 在这里做你的业务逻辑 (直接操作字典)
        # ==========================================
        
        # 演示：修改 req_type 证明数据被处理过
        if 'req_type' in data_dict:
            data_dict['req_type'] = int(data_dict['req_type']) + 100
        
        # 演示：向 input_key 列表添加一个新 key
        if 'input_key' in data_dict:
            data_dict['input_key'].append("processed_by_server")

        # ==========================================

        # Step 5: Python 字典 -> 新的 Protobuf 对象
        resp_proto = pipeline_model_pb2.ModelFullParam()
        # ParseDict 会自动处理 ListValue、Struct 等复杂类型的转换
        ParseDict(data_dict, resp_proto)

        # Step 6: Protobuf 对象 -> 二进制流
        resp_bytes = resp_proto.SerializeToString()

        # Step 7: 返回二进制流，并设置 Content-Type
        return Response(content=resp_bytes, media_type="application/x-protobuf")

    except Exception as e:
        import traceback
        traceback.print_exc()
        return Response(content=f"Server Error: {str(e)}".encode(), status_code=500)

if __name__ == "__main__":
    # 启动端口 8082
    uvicorn.run(app, host="0.0.0.0", port=8082)