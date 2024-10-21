import asyncio

import pyaudio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import json
import base64
import time
import threading
import websocket
from websocket import create_connection
from urllib.parse import quote
import hashlib
import hmac

app = FastAPI()

base_url = "ws://rtasr.xfyun.cn/v1/ws"
app_id = 'd987b3ac'
api_key = 'b2165ad471e11d32ed77c88aa5796575'
end_tag = "{\"end\": true}"

class Client():
    def __init__(self, websocket):
        self.websocket = websocket
        # 生成鉴权参数
        ts = str(int(time.time()))
        tmp = app_id + ts
        hl = hashlib.md5()
        hl.update(tmp.encode(encoding='utf-8'))
        h2 = hl.hexdigest()
        apikey = (bytes(api_key.encode('utf-8')))
        h2 = h2.encode('utf-8')
        my_sign = hmac.new(apikey, h2, hashlib.sha1).digest()
        signa = base64.b64encode(my_sign).decode('utf-8')
        self.ws = create_connection(base_url + "?appid=" + app_id + "&ts=" + ts + "&signa=" + quote(signa))
        self.trecv = threading.Thread(target=self.recv)
        self.trecv.start()

    def send(self):
        # 打开麦克风
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
        try:
            while True:
                chunk = stream.read(1024)
                self.ws.send(bytes(chunk))
                time.sleep(0.04)
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

        self.ws.send(bytes(end_tag.encode('utf-8')))
        print("send end tag success")

    def recv(self):
        try:
            while self.ws.connected:
                result = str(self.ws.recv())
                if len(result) == 0:
                    print("receive result end")
                    break
                result_dict = json.loads(result)

                if result_dict["action"] == "started":
                    print("handshake success, result: " + result)

                if result_dict["action"] == "result":
                    result1 = json.loads(result_dict['data'])
                    result2 = result1['cn']['st']['rt'][0]['ws']
                    str1 = []
                    for item in result2:
                        result3 = item['cw'][0]['w']
                        str1.append(result3)
                    s = ''.join(str1)
                    print(s)  # 输出最后得到的字符串
                    self.send_to_client(s)

                if result_dict["action"] == "error":
                    print("rtasr error: " + result)
                    self.ws.close()
                    return
        except websocket.WebSocketConnectionClosedException:
            print("receive result end")

    def send_to_client(self, text):
        # 将识别结果发送回客户端
        asyncio.run(self.websocket.send_json({"action": "result", "text": text}))

clients = []

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.append(websocket)
    client = Client(websocket)

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.receive":
                if "bytes" in message:
                    # 获取音频数据的大小
                    data_size = len(message["bytes"])
                    print(f"Received audio data of size: {data_size} bytes")

                    # 发送一个确认消息回客户端
                    await websocket.send_json({"action": "ack", "size": data_size})

                    # 将音频数据转发到讯飞云
                    client.send(message["bytes"])
                elif "text" in message:
                    # 处理文本消息
                    text_message = message["text"]
                    print(f"Received text message: {text_message}")

                    # 发送一个确认消息回客户端
                    await websocket.send_json({"action": "ack", "text": text_message})
            elif message["type"] == "websocket.disconnect":
                print("Client disconnected")
                break
    except WebSocketDisconnect:
        clients.remove(websocket)
        client.close()
        print("Client disconnected")

@app.get("/")
async def get():
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read(), status_code=200)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)