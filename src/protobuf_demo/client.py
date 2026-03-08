import requests
from google.protobuf.json_format import ParseDict, MessageToDict
import model_full_param_proto_pb2  # 请确保已生成该文件

def send_model_request():
    # 1. 构造符合 .proto 定义的字典（所有字段均为 camelCase）
    mock_data = {
        "reqType": 2,
        "periodForecast": 24,
        "modelUpdateTime": "2026-02-04 16:00:00",

        "inputKey": ["pressure", "flow"],

        # map<string, FloatList> → 字典，值为 float 列表（自动映射到 FloatList.values）
        "inputValue": {
            "2026/02/04 15:00:00": [10.5, 20.3],
            "2026/02/04 16:00:00": [11.0, 21.1]
        },

        # ModelInputProto
        "controlPoint": {
            "headers": ["TIME", "PRESSURE", "FLOW"],
            # data: repeated StringList → List[List[str]]
            "data": [
                ["2026/02/04 15:00:00", "10.5", "20.3"],
                ["2026/02/04 16:00:00", "11.0", "21.1"]
            ]
        },

        # repeated StationPredictProto
        "stationPredictProtos": [
            {
                "stationId": "ST_001",
                "inputPress": 5.6,
                "outputPress": 4.8,
                "time": 1707052800  # Unix timestamp (秒)
            },
            {
                "stationId": "ST_002",
                "inputPress": 6.0,
                "outputPress": 5.0,
                "time": 1707056400
            }
        ],

        # repeated ModelOptimizeParamProto
        "modelOptimizeParamProtos": [
            {
                "requestSource": 2,
                "stationId": "ST_001",
                "name": "Pressure Adjustment",
                "triggerTime": 1707052800,
                "type": 1,
                "value": {"numberValue": 5.8},          # google.protobuf.Value
                "threshold": {"numberValue": 5.5},
                "triggerReason": "Manual trigger from frontend",
                "firstStationId": "ST_001",
                "secondStationId": "ST_002"
            }
        ],

        # map<string, ListValue>
        "modelUpdateParam": {
            "pipelineStore": [
                {"fields": {"pipelineId": {"stringValue": "PIPE_01"}, "value": {"stringValue": "200"}}}
            ],
            "compressorStatus": [
                {"stringValue": "ON"},
                {"stringValue": "OFF"}
            ]
        }
    }

    # 2. 转换为 Protobuf 消息对象
    request_msg = model_full_param_proto_pb2.ModelFullParamProto()
    try:
        ParseDict(mock_data, request_msg)
        print("✅ 成功构建 Protobuf 消息")
    except Exception as e:
        print(f"❌ 构建失败: {e}")
        return

    # 3. 序列化并发送
    url = "http://localhost:6111/predict/process_json_new/"
    payload = request_msg.SerializeToString()

    try:
        response = requests.post(
            url,
            data=payload,
            headers={"Content-Type": "application/x-protobuf"}
        )

        if response.status_code == 200:
            # 4. 解析响应（假设返回同类型消息）
            resp_msg = model_full_param_proto_pb2.ModelFullParamProto()
            resp_msg.ParseFromString(response.content)
            resp_dict = MessageToDict(resp_msg, preserving_proto_field_name=True)

            print("\n✅ 收到服务器响应:")
            print(f"Response as dict keys: {list(resp_dict.keys())}")
            print(f"  reqType: {resp_dict.get('reqType')}")
            print(f"  inputValue 示例: {resp_dict.get('inputValue', {})}")
        else:
            print(f"❌ 服务器错误 ({response.status_code}): {response.text}")

    except Exception as e:
        print(f"❌ 请求异常: {e}")

if __name__ == "__main__":
    send_model_request()