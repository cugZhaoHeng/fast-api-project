# client.py
import requests
import person_pb2
import json

def create_protobuf_data():
    """创建 Protobuf 数据"""
    person = person_pb2.Person()
    person.name = "张三"
    person.age = 20
    
    # 序列化为二进制
    return person.SerializeToString()

def send_request():
    """发送请求到服务器"""
    url = "http://127.0.0.1:8000/api/person"
    
    # 1. 创建 Protobuf 数据
    protobuf_data = create_protobuf_data()
    
    # 2. 设置请求头，表明发送的是 protobuf 数据
    headers = {
        "Content-Type": "application/x-protobuf",
        "Accept": "application/json"
    }
    
    # 3. 发送请求
    try:
        response = requests.post(
            url, 
            data=protobuf_data, 
            headers=headers
        )
        
        # 4. 打印结果
        print(f"状态码: {response.status_code}")
        print(f"响应内容: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")
        
    except Exception as e:
        print(f"请求失败: {e}")

def test_with_curl_command():
    """生成 curl 命令，方便测试"""
    person = person_pb2.Person()
    person.name = "张三"
    person.age = 20
    data = person.SerializeToString()
    
    # 保存为文件，方便 curl 使用
    with open("person.bin", "wb") as f:
        f.write(data)
    
    print("\n使用 curl 测试的命令:")
    print('curl -X POST http://127.0.0.1:8000/api/person \\')
    print('  -H "Content-Type: application/x-protobuf" \\')
    print('  --data-binary @person.bin')
    print('\n或者在 PowerShell 中使用:')
    print('curl.exe -X POST http://127.0.0.1:8000/api/person ')
    print('  -H "Content-Type: application/x-protobuf" ')
    print('  --data-binary @person.bin')

if __name__ == "__main__":
    print("1. 使用 Python requests 发送请求:")
    send_request()
    
    print("\n" + "="*50 + "\n")
    
    print("2. 使用 curl 命令测试:")
    test_with_curl_command()