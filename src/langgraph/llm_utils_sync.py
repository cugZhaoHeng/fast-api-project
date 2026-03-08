# llm_utils_sync.py
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
        logger.info(f"response:{response.content}")
        
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


class FixedChatOpenAISync(ChatOpenAI):
    """同步版本的 FixedChatOpenAI"""
    
    def _generate(
        self,
        messages: List[Any],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> LLMResult:
        # 记录调用开始
        logger.debug(f"开始 LLM 调用，消息数量: {len(messages)}")
        
        # 调用父类生成原始结果
        result: LLMResult = super()._generate(messages, stop=stop, **kwargs)
        
        # 记录原始响应摘要
        if result.generations and len(result.generations) > 0:
            first_gen = result.generations[0]
            if hasattr(first_gen, 'message') and hasattr(first_gen.message, 'content'):
                content_preview = first_gen.message.content[:100] + "..." if len(first_gen.message.content) > 100 else first_gen.message.content
                logger.debug(f"LLM 响应内容预览: {content_preview}")
        
        # 修复 tool_calls 中的 arguments 字符串
        for generation in result.generations:
            if not isinstance(generation, ChatGeneration):
                continue

            message = generation.message
            if not isinstance(message, AIMessage):
                continue

            # 如果没有 tool_calls，跳过
            if not getattr(message, "tool_calls", None):
                continue

            fixed_tool_calls = []
            for tc in message.tool_calls:
                fixed_tc = dict(tc)  # 浅拷贝
                args = tc.get("args")
                if isinstance(args, str):
                    try:
                        fixed_tc["args"] = json.loads(args)
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.warning(f"解析 tool_call arguments 失败: {e}")
                        fixed_tc["args"] = {}
                fixed_tool_calls.append(fixed_tc)

            # 创建新的 AIMessage
            generation.message = AIMessage(
                content=message.content,
                tool_calls=fixed_tool_calls,
                additional_kwargs=message.additional_kwargs,
                response_metadata=message.response_metadata,
                id=message.id,
            )

        return result
    
    def invoke(self, *args, **kwargs):
        """同步调用方法，添加额外日志"""
        logger.debug("调用 LLM.invoke() 方法")
        return super().invoke(*args, **kwargs)
    
    def batch(self, *args, **kwargs):
        """同步批量调用方法，添加额外日志"""
        logger.debug("调用 LLM.batch() 方法")
        return super().batch(*args, **kwargs)


def load_chat_model_openai_sync(
    enable_logging: bool = True,
    model: str = "qwen2.5-instruct",
    base_url: str = "http://10.10.3.92:9998/v1",
    api_key: str = "osfks",
    **kwargs
) -> BaseChatModel:
    """加载同步版本的聊天模型"""
    logger.info(f"加载同步版 FixedChatOpenAI 模型: {model}")
    
    # 创建 HTTP 客户端
    http_client = None
    if enable_logging:
        # 创建带日志记录的同步 transport
        transport = LoggingSyncHTTPTransport()
        
        # 创建同步 HTTP 客户端
        http_client = httpx.Client(
            transport=transport,
            timeout=httpx.Timeout(60.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            follow_redirects=True,
        )
        logger.info("已启用 HTTP 请求/响应日志记录")
    else:
        logger.info("HTTP 日志记录已禁用")
    
    # 合并默认参数和自定义参数
    default_kwargs = {
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
        "temperature": 0.1,
        "max_tokens": 32768,
        "timeout": 60.0,
        "max_retries": 2,
    }
    
    # 更新默认参数
    default_kwargs.update(kwargs)
    
    # 如果启用了日志记录，添加 http_client
    if http_client:
        default_kwargs["http_client"] = http_client
    
    # 创建 LLM 实例
    llm = FixedChatOpenAISync(**default_kwargs)
    
    logger.info(f"同步模型加载成功: {llm.model_name}")
    return llm


# 工具函数
def print_request_summary(request_info: dict) -> None:
    """打印请求摘要"""
    summary = {
        "method": request_info.get("method"),
        "url": request_info.get("url"),
        "body_preview": {}
    }
    
    # 提取 body 中的关键信息
    body = request_info.get("body")
    if body and isinstance(body, dict):
        if "messages" in body:
            msg_count = len(body["messages"])
            summary["body_preview"]["message_count"] = msg_count
            if msg_count > 0:
                last_msg = body["messages"][-1]
                if isinstance(last_msg, dict) and "content" in last_msg:
                    content = last_msg["content"]
                    preview = content[:50] + "..." if len(content) > 50 else content
                    summary["body_preview"]["last_message"] = preview
        if "model" in body:
            summary["body_preview"]["model"] = body["model"]
    
    logger.debug(f"请求摘要: {json.dumps(summary, ensure_ascii=False)}")


# 创建默认的同步 LLM 实例
default_llm_sync = load_chat_model_openai_sync()