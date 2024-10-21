#-*- encoding:utf-8 -*-
#实现语音实时转为文字

import sys
import hashlib
from hashlib import md5
import hmac
import base64
import json
import time
import threading
from websocket import create_connection
import websocket
from urllib.parse import quote
import logging
import importlib
import pyaudio

logging.basicConfig()

base_url = "ws://rtasr.xfyun.cn/v1/ws"
app_id = 'd987b3ac'
api_key = 'b2165ad471e11d32ed77c88aa5796575'
end_tag = "{\"end\": true}"
class Client():
    def __init__(self):
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
        # self.ws = create_connection(base_url + "?appid=" + app_id + "&ts=" + ts + "&signa=" + quote(signa)+"&lang=en" )#对应参数
        self.ws = create_connection(base_url + "?appid=" + app_id + "&ts=" + ts + "&signa=" + quote(signa) )#对应参数
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

                # 解析结果
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

                if result_dict["action"] == "error":
                    print("rtasr error: " + result)
                    self.ws.close()
                    return
        except websocket.WebSocketConnectionClosedException:
            print("receive result end")

    def close(self):
        self.ws.close()
        print("connection closed")

if __name__ == '__main__':
    client = Client()
    client.send()