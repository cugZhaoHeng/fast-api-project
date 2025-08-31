# client_receive_protobuf.py
import requests
import struct
import output_pb2

# ================== 配置 ==================
API_URL = "http://114.55.113.162:6112/predict/process_json/"
JSON_DATA = {
    "headers": ["col1", "col2"],
    "data": [["input_row1_val1", "input_row1_val2"]]
}

# ================== 流式解析函数（增强版）==================
def parse_protobuf_stream(response):
    buffer = b""
    message_count = 0

    for chunk in response.iter_content(chunk_size=8192):
        if not chunk:
            continue

        buffer += chunk

        while len(buffer) >= 4:
            size = struct.unpack('>I', buffer[:4])[0]
            total_size = 4 + size

            if len(buffer) >= total_size:
                pb_data = buffer[4:total_size]
                buffer = buffer[total_size:]
                msg = output_pb2.PredictionOutput()

                try:
                    msg.ParseFromString(pb_data)
                    message_count += 1

                    print(f"✅ 消息 {message_count}")
                    print(f"   📄 文件名: {msg.filename}")
                    print(f"   🔢 字节大小: {len(pb_data)} bytes")
                    print(f"   🏷️  表头: {list(msg.headers)}")

                    # 打印前 2 行数据
                    print(f"   📄 数据（前 2 行）:")
                    for i, row in enumerate(msg.data):
                        if i >= 2:  # 只打印前两行
                            print(f"     ... 还有 {len(msg.data) - 2} 行")
                            break
                        print(f"     第 {i+1} 行: {list(row.values)}")

                    print("-" * 60)

                except Exception as e:
                    print(f"❌ 解析失败: {e}")
                    print(f"     原始数据长度: {len(pb_data)}")
            else:
                break

    if message_count == 0:
        print("⚠️ 未收到任何有效消息，请检查服务端是否返回了正确格式的 Protobuf 流。")


# ================== 主函数 ==================
def main():
    try:
        print(f"📤 正在向 {API_URL} 发送请求...")
        with requests.post(API_URL, json=JSON_DATA, stream=True) as resp:
            if resp.status_code != 200:
                print(f"❌ 请求失败: {resp.status_code} {resp.text}")
                return

            content_type = resp.headers.get("content-type", "")
            if not content_type.startswith("application/x-protobuf"):
                print(f"❌ 返回类型错误: {content_type}")
                print(f"💡 响应预览: {resp.text[:300]}")
                return

            print("📥 开始接收 Protobuf 流...")
            parse_protobuf_stream(resp)

    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到服务器，请确保 FastAPI 服务正在运行。")
    except Exception as e:
        print(f"❌ 发生错误: {e}")


if __name__ == "__main__":
    main()