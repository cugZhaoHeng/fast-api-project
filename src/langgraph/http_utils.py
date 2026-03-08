import httpx
import json
from typing import Any, List, Optional
from loguru import logger
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from langchain_core.outputs import Generation, ChatGeneration, LLMResult
from langchain_core.language_models.chat_models import BaseChatModel
from datetime import datetime
import uuid

class LoggingHTTPTransport(httpx.AsyncHTTPTransport):
    """自定义 HTTP Transport 用于记录请求详情"""
    
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        import time
        start_time = time.time()
        
        # 记录请求信息
        request_info = {
            "timestamp": datetime.now().isoformat(),
            "method": request.method,
            "url": str(request.url),
            "headers": dict(request.headers),
            "body": None
        }
        
        # 过滤敏感信息
        if 'authorization' in request_info['headers']:
            request_info['headers']['authorization'] = 'Bearer ***'
        if 'api-key' in request_info['headers']:
            request_info['headers']['api-key'] = '***'
        
        # 解析请求体
        if request.content:
            try:
                body_content = request.content.decode('utf-8')
                request_info["body"] = json.loads(body_content)
            except:
                request_info["body"] = str(request.content)[:1000]
        
        logger.info(f"🚀 LLM REQUEST:\n{json.dumps(request_info, indent=2, ensure_ascii=False)}")
        
        # 执行请求
        response = await super().handle_async_request(request)
        
        # 记录响应信息
        duration_ms = (time.time() - start_time) * 1000
        response_info = {
            "timestamp": datetime.now().isoformat(),
            "duration_ms": round(duration_ms, 2),
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "body": None
        }
        
        # 解析响应体
        try:
            response_content = response.content.decode('utf-8')
            response_info["body"] = json.loads(response_content)
        except:
            response_info["body"] = str(response.content)[:2000]
        
        logger.info(f"📨 LLM RESPONSE ({duration_ms:.0f}ms):\n{json.dumps(response_info, indent=2, ensure_ascii=False)}")
            
        return response



class LoggingSyncHTTPTransport(httpx.HTTPTransport):
    """同步版本的自定义 HTTP Transport 用于记录请求详情"""
    
    def handle_request(self, request: httpx.Request) -> httpx.Response:
        import time
        start_time = time.time()
        request_id = str(uuid.uuid4())[:8]
        
        # 记录请求信息
        request_info = {
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "headers": dict(request.headers),
            "body": None
        }
        
        # 过滤敏感信息
        self._sanitize_headers(request_info['headers'])
        
        # 解析请求体
        if request.content:
            try:
                body_content = request.content.decode('utf-8')
                request_info["body"] = json.loads(body_content)
            except:
                request_info["body"] = str(request.content)[:1000]
        
        logger.info(f"[{request_id}] 🚀 LLM REQUEST:\n{json.dumps(request_info, indent=2, ensure_ascii=False)}")
        
        # 执行请求
        response = super().handle_request(request)
        
        # 记录响应信息
        duration_ms = (time.time() - start_time) * 1000
        response_info = {
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "duration_ms": round(duration_ms, 2),
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "body": None
        }
        
        # 解析响应体
        try:
            response_content = response.content.decode('utf-8')
            response_info["body"] = json.loads(response_content)
        except:
            response_info["body"] = str(response.content)[:2000]
        
        logger.info(f"[{request_id}] 📨 LLM RESPONSE ({duration_ms:.0f}ms):\n{json.dumps(response_info, indent=2, ensure_ascii=False)}")
            
        return response
    
    def _sanitize_headers(self, headers: dict) -> None:
        """过滤敏感信息"""
        sensitive_keys = ['authorization', 'api-key', 'x-api-key', 'api_key']
        for key in sensitive_keys:
            if key in headers:
                headers[key] = '***'
            # 检查大小写变体
            key_lower = key.lower()
            for header_key in list(headers.keys()):
                if header_key.lower() == key_lower:
                    headers[header_key] = '***'
