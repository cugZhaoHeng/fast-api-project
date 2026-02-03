import requests
import pipeline_model_pb2
from google.protobuf.json_format import ParseDict

def run_client():
    url = "http://localhost:8082/api/model/process"

    # 1. 构造初始数据
    request_data = {
        "req_type": 1,  # 发送 1
        "period_forecast": 24,
        "input_key": ["temp", "pressure"]
    }
    
    # 2. 封装发送数据
    req_proto = pipeline_model_pb2.ModelFullParam()
    ParseDict(request_data, req_proto)
    req_bytes = req_proto.SerializeToString()
    
    print(f"【Client】发送 req_type: {req_proto.req_type}")

    # 3. 发送请求
    headers = {"Content-Type": "application/x-protobuf"}
    resp = requests.post(url, data=req_bytes, headers=headers)

    if resp.status_code == 200:
        # 4. 解析返回的数据
        resp_proto = pipeline_model_pb2.ModelFullParam()
        resp_proto.ParseFromString(resp.content)
        
        print("-" * 30)
        print("【Client】收到服务器返回的数据:")
        print(f"  - req_type (预期是101): {resp_proto.req_type}")
        print(f"  - input_key: {resp_proto.input_key}")
        print("-" * 30)
    else:
        print("请求失败:", resp.text)

if __name__ == "__main__":
    run_client()