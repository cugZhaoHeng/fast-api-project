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
import os
import json
import zipfile
from io import BytesIO, StringIO

import pandas as pd
import torch
import numpy as np
from datetime import datetime, timedelta
import re
import matplotlib.pyplot as plt
import random
import requests
from fastapi import APIRouter, UploadFile, File, HTTPException
from matplotlib.colors import LinearSegmentedColormap
from starlette.responses import StreamingResponse
from utils.logger import create_logger


logger = create_logger(__name__)
router = APIRouter(prefix="/predict", tags=["origin_dataset"])
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

@router.get("/get_flow_hour")
async def get_current_flow_data():
    logger.info(f"有人访问了get_flow_hour接口")
    return parse_json_url("http://114.55.113.162:8084/shifan-data/flow_hour.json")

@router.get("/get_sensor")
async def get_current_flow_data():
    return parse_json_url("http://114.55.113.162:8084/shifan-data/sensor.json")

@router.post("/process_csv/")
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
        zip_filename = "processed_files.zip"
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

   