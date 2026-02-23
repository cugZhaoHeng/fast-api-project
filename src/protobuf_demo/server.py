import uvicorn
from fastapi import FastAPI, Request, Response
from google.protobuf.json_format import MessageToDict, ParseDict
import pipeline_model_pb2  # 确保这个文件是你最新生成的

app = FastAPI(
    title="Pipeline Model API",
    description="Protobuf 全驼峰命名版本 (All CamelCase)",
    version="3.0.0"
)

@app.post(
    "/api/model/process",
    summary="接收并处理全驼峰命名的 Protobuf",
    responses={
        200: {
            "description": "成功，返回二进制流",
            "content": {"application/x-protobuf": {}}
        }
    }
)
async def process_model_param(request: Request):
    try:
        # 1. 接收二进制
        req_bytes = await request.body()
        
        # 2. 反序列化
        model_param = pipeline_model_pb2.ModelFullParam()
        model_param.ParseFromString(req_bytes)
        
        # --- 打印验证 ---
        # 注意：现在所有字段属性访问都推荐用驼峰（取决于你的编译环境，如果报错改为snake_case）
        # 但既然你的proto定义是驼峰，我们先尝试直接访问
        # 重点：上一版叫 input_value，现在叫 inputValue
        print(f"【Server】收到请求 reqType: {getattr(model_param, 'reqType', 'Unknown')}")
        print(f"【Server】inputValue 包含 Key 数量: {len(model_param.inputValue)}")
        
        # 3. 转为字典 (核心步骤)
        # preserving_proto_field_name=True: 
        # 强制使用 .proto 文件中定义的名称 (即 reqType, inputValue)
        data_dict = MessageToDict(
            model_param, 
            preserving_proto_field_name=True
        )
        
        # 4. 业务逻辑处理
        # 演示：修改 reqType
        if 'reqType' in data_dict:
            data_dict['reqType'] = int(data_dict['reqType']) + 200
            
        # 演示：修改 inputValue (重点变化)
        # 以前是 data_dict['input_value']，现在必须是 data_dict['inputValue']
        if 'inputValue' in data_dict:
            # 这是一个 map<string, ListValue>，对应 Python 字典
            for time_key, list_val in data_dict['inputValue'].items():
                # list_val 是一个列表，我们给每个值加 1.0 用于测试
                new_vals = [v + 1.0 for v in list_val]
                data_dict['inputValue'][time_key] = new_vals
                
        print(f"【Server】处理完成，准备返回。reqType 改为: {data_dict.get('reqType')}")

        # 5. 重新封装回 Protobuf
        resp_proto = pipeline_model_pb2.ModelFullParam()
        ParseDict(data_dict, resp_proto)

        # 6. 返回二进制
        return Response(
            content=resp_proto.SerializeToString(), 
            media_type="application/x-protobuf"
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return Response(content=f"Server Error: {str(e)}".encode(), status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8082)