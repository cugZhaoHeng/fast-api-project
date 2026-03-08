# llm_utils_with_callbacks.py
import json
from typing import Any, Dict, List, Optional
from loguru import logger
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import LLMResult, ChatGeneration, Generation
from langchain_core.language_models.chat_models import BaseChatModel
from datetime import datetime
import uuid


class DetailedLLMCallback(BaseCallbackHandler):
    """详细的 LLM 回调处理器"""
    
    def __init__(self, request_id: str = None):
        super().__init__()
        self.request_id = request_id or str(uuid.uuid4())[:8]
        self.start_time = None
        
    def on_llm_start(
        self, 
        serialized: Dict[str, Any], 
        prompts: List[str], 
        **kwargs: Any
    ) -> Any:
        """当 LLM 开始运行时调用"""
        self.start_time = datetime.now()
        
        # 提取请求信息
        invocation_params = serialized.get("kwargs", {})
        
        request_info = {
            "request_id": self.request_id,
            "timestamp": self.start_time.isoformat(),
            "model": invocation_params.get("model", "unknown"),
            "base_url": invocation_params.get("base_url", "unknown"),
            "temperature": invocation_params.get("temperature"),
            "max_tokens": invocation_params.get("max_tokens"),
            "prompts": prompts,
            "invocation_params": {k: v for k, v in invocation_params.items() 
                                 if k not in ['api_key', 'http_client']},  # 过滤敏感信息
            "other_kwargs": kwargs
        }
        
        logger.info(f"[{self.request_id}] 🚀 LLM 调用开始")
        logger.debug(f"请求详情: {json.dumps(request_info, indent=2, ensure_ascii=False)}")
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """当 LLM 结束运行时调用"""
        if not self.start_time:
            return
            
        duration = (datetime.now() - self.start_time).total_seconds() * 1000
        
        # 构建响应信息
        response_info = {
            "request_id": self.request_id,
            "timestamp": datetime.now().isoformat(),
            "duration_ms": round(duration, 2),
            "generations": []
        }
        
        # 解析生成结果
        for i, generation_list in enumerate(response.generations):
            for j, generation in enumerate(generation_list):
                gen_info = {
                    "index": f"{i}-{j}",
                    "text": generation.text[:200] + "..." if len(generation.text) > 200 else generation.text,
                    "message_type": type(generation.message).__name__ if hasattr(generation, 'message') else "unknown"
                }
                
                if hasattr(generation, 'message'):
                    msg = generation.message
                    if hasattr(msg, 'content'):
                        gen_info["content"] = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                    
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        gen_info["tool_calls_count"] = len(msg.tool_calls)
                        gen_info["tool_calls"] = [
                            {"name": tc.get("name"), "args": tc.get("args")}
                            for tc in msg.tool_calls[:3]  # 只记录前3个
                        ]
                
                response_info["generations"].append(gen_info)
        
        logger.info(f"[{self.request_id}] 📨 LLM 调用完成 ({duration:.0f}ms)")
        logger.debug(f"响应详情: {json.dumps(response_info, indent=2, ensure_ascii=False)}")
    
    def on_llm_error(self, error: BaseException, **kwargs: Any) -> Any:
        """当 LLM 出错时调用"""
        logger.error(f"[{self.request_id}] ❌ LLM 调用出错: {error}")


class FixedChatOpenAI(ChatOpenAI):
    """修复 Qwen 等模型在 OpenAI 兼容模式下 tool_calls.arguments 为字符串的问题"""
    
    def _generate(
        self,
        messages: List[Any],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> LLMResult:
        # 在调用前记录消息详情（可选）
        if logger.isEnabledFor(10):  # DEBUG 级别
            logger.debug(f"准备调用 LLM，消息数量: {len(messages)}")
            for i, msg in enumerate(messages):
                if hasattr(msg, 'content'):
                    content_preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                    logger.debug(f"消息 {i}: {type(msg).__name__}: {content_preview}")
        
        # 调用父类生成原始结果
        result: LLMResult = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        
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
                fixed_tc = dict(tc)
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


def load_chat_model_with_callbacks(**kwargs) -> BaseChatModel:
    """加载带回调的聊天模型"""
    logger.info("加载 FixedChatOpenAI 模型...")
    
    # 创建 LLM 实例
    llm = FixedChatOpenAI(
        model=kwargs.get("model", "qwen2.5-instruct"),
        api_key=kwargs.get("api_key", "osfks"),
        base_url=kwargs.get("base_url", "http://10.10.3.92:9998/v1"),
        temperature=kwargs.get("temperature", 0.1),
        max_tokens=kwargs.get("max_tokens", 32768),
        timeout=kwargs.get("timeout", 60.0),
        max_retries=kwargs.get("max_retries", 2),
    )
    
    logger.info(f"模型加载成功: {llm.model_name}")
    return llm


# 使用装饰器包装 LLM 调用
def with_llm_logging(func):
    """为 LLM 调用添加日志的装饰器"""
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        request_id = str(uuid.uuid4())[:8]
        start_time = datetime.now()
        
        # 提取消息参数
        messages = None
        if args and len(args) > 0:
            messages = args[0] if isinstance(args[0], list) else None
        if not messages and 'messages' in kwargs:
            messages = kwargs['messages']
        
        # 记录请求
        if messages:
            logger.info(f"[{request_id}] 🚀 LLM 调用开始")
            for i, msg in enumerate(messages):
                if hasattr(msg, 'content'):
                    content_preview = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
                    logger.debug(f"[{request_id}] 消息 {i}: {content_preview}")
        
        try:
            # 执行原始函数
            result = func(*args, **kwargs)
            
            # 记录响应
            duration = (datetime.now() - start_time).total_seconds() * 1000
            
            if hasattr(result, 'content'):
                content_preview = result.content[:100] + "..." if len(result.content) > 100 else result.content
                logger.info(f"[{request_id}] 📨 LLM 调用完成 ({duration:.0f}ms): {content_preview}")
            else:
                logger.info(f"[{request_id}] 📨 LLM 调用完成 ({duration:.0f}ms)")
            
            return result
        except Exception as e:
            logger.error(f"[{request_id}] ❌ LLM 调用失败: {e}")
            raise
    
    return wrapper