# logging_llm_wrapper.py
import json
from typing import Any, List, Optional, Union, Iterator
from loguru import logger
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from datetime import datetime
import uuid


class LoggingLLMWrapper(BaseChatModel):
    """包装器类，为任何 LLM 添加详细的日志记录"""
    
    def __init__(self, llm: BaseChatModel, log_level: str = "INFO"):
        super().__init__()
        self.llm = llm
        self.log_level = log_level
        self._should_log = logger.isEnabledFor(
            getattr(logger.level(self.log_level), "no", 20)
        )
    
    @property
    def _llm_type(self) -> str:
        return f"logging_wrapper_{self.llm._llm_type}"
    
    def _log_request(self, messages: List[BaseMessage], request_id: str, **kwargs):
        """记录请求日志"""
        if not self._should_log:
            return
            
        request_info = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "message_count": len(messages),
            "messages_summary": []
        }
        
        for i, msg in enumerate(messages):
            msg_info = {"index": i, "type": type(msg).__name__}
            if hasattr(msg, 'content'):
                content = msg.content
                msg_info["content_preview"] = content[:100] + "..." if len(content) > 100 else content
            if hasattr(msg, 'additional_kwargs') and msg.additional_kwargs:
                msg_info["has_additional_kwargs"] = True
            request_info["messages_summary"].append(msg_info)
        
        logger.info(f"[{request_id}] 🚀 LLM 调用开始，{len(messages)} 条消息")
        logger.debug(f"[{request_id}] 请求详情: {json.dumps(request_info, indent=2, ensure_ascii=False)}")
    
    def _log_response(self, result: ChatResult, request_id: str, duration_ms: float):
        """记录响应日志"""
        if not self._should_log:
            return
            
        response_info = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "duration_ms": round(duration_ms, 2),
            "generation_count": len(result.generations),
            "generations": []
        }
        
        for i, generation in enumerate(result.generations):
            gen_info = {"index": i}
            if hasattr(generation, 'message'):
                msg = generation.message
                gen_info["message_type"] = type(msg).__name__
                
                if hasattr(msg, 'content'):
                    content = msg.content
                    gen_info["content_preview"] = content[:200] + "..." if len(content) > 200 else content
                
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    gen_info["tool_calls_count"] = len(msg.tool_calls)
                    gen_info["tool_calls"] = [
                        {"name": tc.get("name")} for tc in msg.tool_calls[:2]
                    ]
            
            response_info["generations"].append(gen_info)
        
        logger.info(f"[{request_id}] 📨 LLM 调用完成 ({duration_ms:.0f}ms)")
        logger.debug(f"[{request_id}] 响应详情: {json.dumps(response_info, indent=2, ensure_ascii=False)}")
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        import time
        request_id = str(uuid.uuid4())[:8]
        
        # 记录请求
        self._log_request(messages, request_id, **kwargs)
        
        # 调用底层 LLM
        start_time = time.time()
        try:
            result = self.llm._generate(messages, stop=stop, **kwargs)
            duration_ms = (time.time() - start_time) * 1000
            
            # 记录响应
            self._log_response(result, request_id, duration_ms)
            
            return result
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"[{request_id}] ❌ LLM 调用失败 ({duration_ms:.0f}ms): {e}")
            raise
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        import asyncio
        import time
        request_id = str(uuid.uuid4())[:8]
        
        # 记录请求
        self._log_request(messages, request_id, **kwargs)
        
        # 调用底层 LLM
        start_time = time.time()
        try:
            result = await self.llm._agenerate(messages, stop=stop, **kwargs)
            duration_ms = (time.time() - start_time) * 1000
            
            # 记录响应
            self._log_response(result, request_id, duration_ms)
            
            return result
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"[{request_id}] ❌ LLM 异步调用失败 ({duration_ms:.0f}ms): {e}")
            raise
    
    def stream(self, *args, **kwargs) -> Iterator[Any]:
        """流式响应（简单包装，记录开始和结束）"""
        request_id = str(uuid.uuid4())[:8]
        logger.info(f"[{request_id}] 🌊 LLM 流式调用开始")
        
        try:
            for chunk in self.llm.stream(*args, **kwargs):
                yield chunk
            logger.info(f"[{request_id}] 🌊 LLM 流式调用完成")
        except Exception as e:
            logger.error(f"[{request_id}] ❌ LLM 流式调用失败: {e}")
            raise
    
    async def astream(self, *args, **kwargs) -> Any:
        """异步流式响应"""
        request_id = str(uuid.uuid4())[:8]
        logger.info(f"[{request_id}] 🌊 LLM 异步流式调用开始")
        
        try:
            async for chunk in self.llm.astream(*args, **kwargs):
                yield chunk
            logger.info(f"[{request_id}] 🌊 LLM 异步流式调用完成")
        except Exception as e:
            logger.error(f"[{request_id}] ❌ LLM 异步流式调用失败: {e}")
            raise
    
    # 代理其他方法
    def bind_tools(self, tools, **kwargs):
        return self.llm.bind_tools(tools, **kwargs)
    
    def with_structured_output(self, schema, **kwargs):
        return self.llm.with_structured_output(schema, **kwargs)
    
    @property
    def model_name(self):
        return self.llm.model_name
    
    def get_num_tokens(self, text: str) -> int:
        return self.llm.get_num_tokens(text)