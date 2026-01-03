# llm_utils.py
import json
from typing import Any, List, Optional
from loguru import logger
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import LLMResult, ChatGeneration, Generation
from langchain_core.language_models.chat_models import BaseChatModel
from datetime import datetime
import uuid
import time


class FixedChatOpenAI(ChatOpenAI):
    """修复 Qwen 等模型在 OpenAI 兼容模式下 tool_calls.arguments 为字符串的问题"""
    
    def _generate(
        self,
        messages: List[Any],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> LLMResult:
        # 为这次调用生成唯一ID
        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        
        # 记录请求开始
        logger.info(f"[{request_id}] 🚀 LLM 调用开始")
        
        # 安全地获取模型信息
        try:
            model_name = self.model_name if hasattr(self, 'model_name') else "unknown"
        except:
            model_name = "unknown"
            
        # 尝试获取温度和其他参数
        temperature = getattr(self, 'temperature', 0.1)
        max_tokens = getattr(self, 'max_tokens', 32768)
        
        # 记录请求详情
        request_info = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "message_count": len(messages),
            "messages": []
        }
        
        for i, msg in enumerate(messages):
            msg_info = {
                "index": i,
                "type": type(msg).__name__,
            }
            
            # 尝试获取角色信息
            if isinstance(msg, HumanMessage):
                msg_info["role"] = "user"
            elif isinstance(msg, AIMessage):
                msg_info["role"] = "assistant"
            elif isinstance(msg, SystemMessage):
                msg_info["role"] = "system"
            else:
                msg_info["role"] = getattr(msg, 'type', 'unknown').replace('_', '') if hasattr(msg, 'type') else 'unknown'
            
            # 获取内容
            if hasattr(msg, 'content'):
                content = msg.content
                if content:
                    msg_info["content_length"] = len(str(content))
                    # 只记录前100个字符，避免日志过长
                    content_str = str(content)
                    msg_info["content"] = content_str[:100] + "..." if len(content_str) > 100 else content_str
                else:
                    msg_info["content"] = None
            
            # 检查是否有工具调用
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                msg_info["tool_calls"] = len(msg.tool_calls)
                
            request_info["messages"].append(msg_info)
        
        logger.debug(f"[{request_id}] 请求详情:\n{json.dumps(request_info, indent=2, ensure_ascii=False)}")
        
        try:
            # 调用父类生成原始结果
            result: LLMResult = super()._generate(messages, stop=stop, **kwargs)
            duration_ms = (time.time() - start_time) * 1000
            
            # 记录响应
            self._log_response(request_id, result, duration_ms)
            
            # 修复 tool_calls 中的 arguments 字符串
            result = self._fix_tool_calls(result)
            
            return result
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"[{request_id}] ❌ LLM 调用失败 ({duration_ms:.0f}ms): {e}")
            raise
    
    def _log_response(self, request_id: str, result: LLMResult, duration_ms: float):
        """记录响应信息"""
        response_info = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "duration_ms": round(duration_ms, 2),
            "generation_count": len(result.generations),
            "generations": []
        }
        
        # 处理 generations 的结构
        for i, generation_item in enumerate(result.generations):
            # generation_item 可能是 ChatGeneration 或列表
            if isinstance(generation_item, list):
                # 如果是列表，遍历列表中的每个生成
                for j, generation in enumerate(generation_item):
                    self._add_generation_info(response_info, generation, f"{i}-{j}")
            else:
                # 如果不是列表，直接处理
                self._add_generation_info(response_info, generation_item, str(i))
        
        logger.info(f"[{request_id}] 📨 LLM 调用完成 ({duration_ms:.0f}ms)")
        logger.debug(f"[{request_id}] 响应详情:\n{json.dumps(response_info, indent=2, ensure_ascii=False)}")
        
        # 打印简化的响应内容
        self._log_response_preview(request_id, result)
    
    def _add_generation_info(self, response_info: dict, generation: Generation, index: str):
        """添加生成信息到响应日志"""
        gen_info = {
            "index": index,
        }
        
        # 记录文本内容
        if hasattr(generation, 'text'):
            text = generation.text
            gen_info["text_preview"] = text[:200] + "..." if len(text) > 200 else text
        
        if hasattr(generation, 'message'):
            msg = generation.message
            gen_info["message_type"] = type(msg).__name__
            
            # 记录消息内容
            if hasattr(msg, 'content'):
                content = msg.content
                if content:
                    gen_info["content_length"] = len(str(content))
                    content_str = str(content)
                    gen_info["content_preview"] = content_str[:200] + "..." if len(content_str) > 200 else content_str
                else:
                    gen_info["content"] = None
            
            # 记录工具调用
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                gen_info["tool_calls_count"] = len(msg.tool_calls)
                gen_info["tool_calls"] = [
                    {"name": tc.get("name"), "args": tc.get("args")}
                    for tc in msg.tool_calls[:3]  # 只记录前3个
                ]
        
        response_info["generations"].append(gen_info)
    
    def _log_response_preview(self, request_id: str, result: LLMResult):
        """记录响应预览"""
        if not result.generations:
            return
            
        # 获取第一个生成
        first_generation = None
        for gen_item in result.generations:
            if isinstance(gen_item, list) and gen_item:
                first_generation = gen_item[0]
                break
            elif isinstance(gen_item, Generation):
                first_generation = gen_item
                break
        
        if first_generation and hasattr(first_generation, 'message') and hasattr(first_generation.message, 'content'):
            content = first_generation.message.content
            if content:
                content_str = str(content)
                preview = content_str[:150] + "..." if len(content_str) > 150 else content_str
                logger.info(f"[{request_id}] 💬 响应内容: {preview}")
            else:
                logger.info(f"[{request_id}] 💬 响应内容为空")
    
    def _fix_tool_calls(self, result: LLMResult) -> LLMResult:
        """修复 tool_calls 中的 arguments 字符串"""
        for generation_item in result.generations:
            # 处理不同的结构
            if isinstance(generation_item, list):
                for generation in generation_item:
                    self._fix_single_generation(generation)
            else:
                self._fix_single_generation(generation_item)
        
        return result
    
    def _fix_single_generation(self, generation: Generation):
        """修复单个生成中的 tool_calls"""
        if not isinstance(generation, ChatGeneration):
            return

        message = generation.message
        if not isinstance(message, AIMessage):
            return

        # 如果没有 tool_calls，跳过
        if not getattr(message, "tool_calls", None):
            return

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


def load_chat_model_openai() -> BaseChatModel:
    """加载聊天模型"""
    logger.info("正在加载 FixedChatOpenAI 模型...")
    
    llm = FixedChatOpenAI(
        model="qwen2.5-instruct",
        api_key="osfks",
        base_url="http://10.10.3.92:9998/v1",
        temperature=0.1,
        max_tokens=32768,
        timeout=60.0,
        max_retries=2,
    )
    
    logger.info(f"模型加载成功")
    return llm


# 创建默认的 LLM 实例
default_llm: BaseChatModel = load_chat_model_openai()


# 测试代码
if __name__ == "__main__":
    
    # 简单调用测试
    print("=== 简单 LLM 调用测试 ===")
    try:
        response = default_llm.invoke("你好，请介绍一下你自己")
        print(f"\n响应内容: {response.content}")
    except Exception as e:
        print(f"调用失败: {e}")