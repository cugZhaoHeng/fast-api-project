"""
天然气管网代理模型预测运行工具 - API 版本

本模块提供代理模型预测程序接口，支持：
1. 最大步数限制
2. 自动结果保存
3. 直接从 API 读取数据并进行处理
4. 生成代理模型所需的输入文件
5. 自动评估预测结果(MAPE、MSE、RMSE、MAE、R2)
6. 随机抽样节点评估（雷达图、柱状图）
7. 总体评估热力图
"""
import csv
import io
import os
import sys

from common_class import QueryBody

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))  # 添加上一级目录
import json
from io import BytesIO
import re
from typing import List

import requests
from fastapi import APIRouter, UploadFile, File, HTTPException
from starlette.responses import StreamingResponse, Response
from utils.logger import create_logger
import sensor_history_pb2
from pydantic import BaseModel
from starlette.responses import StreamingResponse
from logger import create_logger


logger = create_logger(__name__)
router = APIRouter(prefix="/predict", tags=["pipeline"])
xiaosong_data = []
def parse_json_url(url):
    """
    从URL获取并解析可能格式不正确的JSON数据

    参数:
        url (str): JSON数据的URL

    返回:
        list: 解析后的JSON对象列表
    """
    try:
        # 从URL获取数据
        response = requests.get(url, timeout=30)
        response.raise_for_status()  # 如果请求失败，抛出异常
        content = response.text.strip()

        # 首先尝试直接解析
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            # 尝试修复常见的JSON格式问题
            # 1. 移除末尾多余的逗号
            content = re.sub(r',\s*}', '}', content)
            content = re.sub(r',\s*]', ']', content)

            # 2. 处理不完整的数组和对象
            if content.count('[') > content.count(']'):
                content += ']' * (content.count('[') - content.count(']'))
            if content.count('{') > content.count('}'):
                content += '}' * (content.count('{') - content.count('}'))

            # 3. 尝试截取到最后一个完整的JSON对象
            try:
                # 找到最后一个完整的对象或数组
                lines = content.split('\n')
                for i in range(len(lines) - 1, -1, -1):
                    try:
                        partial_content = '\n'.join(lines[:i + 1])
                        # 确保JSON结构完整
                        if partial_content.strip().endswith(('}', ']')):
                            return json.loads(partial_content)
                    except json.JSONDecodeError:
                        continue

                # 如果还是失败，尝试使用正则表达式提取JSON对象
                json_objects = re.findall(r'{[^{}]*}', content)
                if json_objects:
                    return [json.loads(obj) for obj in json_objects]
            except Exception:
                pass

            # 最后的备选方案：返回空列表
            logger.info(f"无法解析URL {url} 的JSON数据，返回空列表")
            return []

    except Exception as e:
        logger.info(f"从URL获取数据失败: {e}")
        return []

@router.get("/get_flow_hour", summary="获取当前 Flow 数据")
async def get_current_flow_data():
    logger.info(f"有人访问了 get_flow_hour 接口")
    try:
        with open("./data/flow_hour.json", "r", encoding="utf-8") as f:
            flow_hour = json.load(f)
        return flow_hour
    except Exception as e:
        logger.error(e)

@router.get("/get_sensor", summary="获取当前 Sensor 数据")
async def get_current_flow_data():
    logger.info(f"有人访问了 get_sensor 接口")
    try:
        with open("./data/sensor.json", "r", encoding="utf-8") as f:
            sensor_data = json.load(f)
        return sensor_data
    except Exception as e:
        logger.error(e)

@router.post("/get_flow_history", summary="获取历史的 Flow 数据")
async def get_flow_history(queryBody: QueryBody):
    logger.info(f"queryBody: {queryBody}")
    try:
        with open("./data/flow_history.json", "r", encoding="utf-8") as f:
            flow_history = json.load(f)
        return flow_history
    except Exception as e:
        logger.info(f"数据有问题")

@router.post("/get_full_sensor_data",summary="获取历史 Sensor 数据", response_class=Response)
def get_full_sensor_data(queryBody: QueryBody):
    logger.info(f"queryBody: {queryBody}")
    # 从本地文件读取 JSON
    try:
        with open('data/sensor_history.json', 'r', encoding='utf-8') as f:
            a = json.load(f)
            logger.info("成功读取 JSON 数据", )
    except FileNotFoundError:
        logger.error("文件未找到: data/sensor_history.json")
        return Response(content="", status_code=404, media_type="application/json")
    except json.JSONDecodeError as e:
        logger.error("JSON 解析错误: %s", e)
        return Response(content="", status_code=500, media_type="application/json")
    except Exception as e:
        logger.error("读取文件时发生未知错误: %s", e)
        return Response(content="", status_code=500, media_type="application/json")

    # 创建 Protobuf 响应对象
    response = sensor_history_pb2.SensorData()

    # 设置 sensors
    if 'sensors' in a:
        response.sensors.extend(a['sensors'])
    else:
        logger.warning("JSON 中缺少 'sensors' 字段")

    # 添加 datas
    if 'datas' in a:
        for record in a['datas']:  # 遍历 datas 列表中的每个记录
            data_record = response.datas.add()  # 创建一个新的 DataRecord
            data_record.time = record.get('time', 0)  # 获取时间戳

            # 添加 data 点
            for point in record.get('data', []):  # 遍历 data 列表
                data_point = data_record.data.add()  # 创建一个新的 DataPoint
                data_point.index = point.get('index', 0)
                data_point.val = float(point.get('val', 0.0))  # 确保是 float
    else:
        logger.warning("JSON 中缺少 'datas' 字段")

    # 序列化为二进制
    try:
        serialized_data = response.SerializeToString()
    except Exception as e:
        logger.error("Protobuf 序列化失败: %s", e)
        return Response(content="", status_code=500, media_type="application/json")

    # 返回 Protobuf 数据
    return Response(
        content=serialized_data,
        media_type="application/x-protobuf"
    )

@router.post("/process_csv/", summary="数据预测接口")
async def process_csv_endpoint(file: UploadFile = File(...)):
    """
    接收上传的 CSV 文件，打印内容，然后返回本地的 processed_files.zip 文件。
    """
    # 1. 验证文件类型（可选）
    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="仅接收CSV文件")

    try:
        # 2. 读取并打印上传文件内容
        contents = await file.read()
        try:
            csv_text = contents.decode('utf-8')
        except UnicodeDecodeError:
            # 如果不是 UTF-8，尝试其他编码（如 gbk），或直接打印二进制信息
            try:
                csv_text = contents.decode('gbk')
            except:
                csv_text = "<无法解码的二进制内容>"

        print(f"接收到的文件名: {file.filename}")
        print(f"文件大小: {len(contents)} 字节")
        print(f"文件内容预览:\n{csv_text[:1000]}...")  # 只打印前1000字符避免刷屏

        await file.close()

        # 3. 读取本地的 ZIP 文件（与当前 Python 文件同目录）
        zip_filename = "data/processed_files.zip"
        # 获取当前文件所在目录
        current_dir = os.path.dirname(__file__)
        zip_path = os.path.join(current_dir, zip_filename)

        if not os.path.exists(zip_path):
            raise HTTPException(status_code=500, detail=f"本地 ZIP 文件不存在: {zip_path}")

        # 读取 ZIP 文件到内存
        with open(zip_path, 'rb') as f:
            zip_data = f.read()

        # 创建 BytesIO 流
        zip_buffer = BytesIO(zip_data)

        # 设置响应头
        headers = {
            "Content-Disposition": f"attachment; filename={zip_filename}"
        }

        # 返回文件流
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers=headers
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理请求时出错: {str(e)}")
import zipfile
import output_pb2
import struct

# 当前文件目录 + 本地 ZIP 文件路径
current_dir = os.path.dirname(__file__)
ZIP_FILE_PATH = os.path.join(current_dir, "processed_files.zip")

# ========== 请求模型 ==========
class ProcessCSVRequest(BaseModel):
    headers: List[str]
    data: List[List[str]]

@router.post("/process_json/")
async def process_csv_endpoint(request: ProcessCSVRequest):
    """
    测试接口：
    - 接收 JSON 数据（不处理）
    - 读取本地 processed_files.zip
    - 解压并解析每个 CSV 文件
    - 流式返回 Protobuf 消息（带长度前缀）
    """
    # ========== 可选：打印收到的 JSON（仅测试用）==========
    print(f"Received JSON - Headers: {request.headers}")
    print(f"Data rows: {len(request.data)}")

    # ========== 检查本地 ZIP 文件是否存在 ==========
    if not os.path.exists(ZIP_FILE_PATH):
        raise HTTPException(status_code=500, detail=f"本地 ZIP 文件不存在: {ZIP_FILE_PATH}")

    # ========== 流式生成器：读 ZIP → 解 CSV → 输出 Protobuf ==========
    def protobuf_stream():
        with zipfile.ZipFile(ZIP_FILE_PATH, 'r') as z:
            for fname in z.namelist():
                # 跳过目录或非 CSV
                if fname.endswith('/') or not fname.lower().endswith('.csv'):
                    continue

                try:
                    # 读取并解码 CSV 内容
                    with z.open(fname) as f:
                        content = f.read().decode('utf-8')

                    csv_file = io.StringIO(content)
                    reader = csv.reader(csv_file)
                    all_rows = list(reader)

                    if not all_rows:
                        print(f"跳过空文件: {fname}")
                        continue

                    headers = all_rows[0]
                    data_rows = all_rows[1:]

                    # 构造 Protobuf 消息
                    pb_msg = output_pb2.PredictionOutput()
                    pb_msg.filename = os.path.basename(fname)
                    pb_msg.headers.extend(headers)

                    for row in data_rows:
                        pb_row = pb_msg.data.add()
                        pb_row.values.extend(row)

                    # ✅ 使用 SerializeToString() 序列化
                    serialized = pb_msg.SerializeToString()  # 返回 bytes

                    # 加上 4 字节长度前缀（大端）
                    prefix = struct.pack('>I', len(serialized))
                    yield prefix + serialized

                    print(f"已发送: {fname}, 数据行数: {len(data_rows)}")

                except Exception as e:
                    print(f"处理文件失败: {fname}, 错误: {e}")
                    # 可选：发送错误消息
                    error_msg = output_pb2.PredictionOutput()
                    error_msg.filename = fname
                    error_msg.headers.append("error")
                    err_row = error_msg.data.add()
                    err_row.values.append(f"Parse error: {str(e)}")
                    serialized = error_msg.SerializeToString()
                    prefix = struct.pack('>I', len(serialized))
                    yield prefix + serialized

    # ========== 返回流（Protobuf 格式）==========
    return StreamingResponse(
        content=protobuf_stream(),
        media_type="application/x-protobuf"
    )

   