# langgraph_with_logging.py
from typing import TypedDict, List, Annotated
from langchain_core.messages import BaseMessage
from loguru import logger
import json
from datetime import datetime
import uuid


class AgentState(TypedDict):
    """增强的 Agent 状态，包含调用日志"""
    messages: List[BaseMessage]
    llm_calls: Annotated[List[dict], "LLM 调用日志"]
    current_request_id: str


def create_logging_llm_node(llm, node_name: str = "llm_node"):
    """创建带日志记录的 LLM 节点"""
    
    def llm_node(state: AgentState):
        import time
        
        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        
        # 获取当前消息
        current_messages = state["messages"]
        
        # 记录请求
        request_info = {
            "request_id": request_id,
            "node_name": node_name,
            "timestamp": datetime.now().isoformat(),
            "message_count": len(current_messages),
            "last_message": current_messages[-1].content[:100] if current_messages else ""
        }
        
        logger.info(f"[{request_id}] 🚀 {node_name} 开始处理")
        logger.debug(f"请求详情: {json.dumps(request_info, indent=2, ensure_ascii=False)}")
        
        # 调用 LLM
        try:
            response = llm.invoke(current_messages)
            duration_ms = (time.time() - start_time) * 1000
            
            # 记录响应
            response_info = {
                "request_id": request_id,
                "duration_ms": round(duration_ms, 2),
                "response_content": response.content[:200] + "..." if len(response.content) > 200 else response.content,
                "has_tool_calls": bool(getattr(response, 'tool_calls', None))
            }
            
            logger.info(f"[{request_id}] 📨 {node_name} 处理完成 ({duration_ms:.0f}ms)")
            logger.debug(f"响应详情: {json.dumps(response_info, indent=2, ensure_ascii=False)}")
            
            # 更新状态
            new_messages = current_messages + [response]
            
            # 记录调用日志
            call_log = {
                "request_id": request_id,
                "node": node_name,
                "timestamp": datetime.now().isoformat(),
                "duration_ms": duration_ms,
                "input_count": len(current_messages),
                "output": response.content[:500] if response.content else "",
                "success": True
            }
            
            llm_calls = state.get("llm_calls", [])
            llm_calls.append(call_log)
            
            return {
                "messages": new_messages,
                "llm_calls": llm_calls,
                "current_request_id": request_id
            }
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"[{request_id}] ❌ {node_name} 处理失败 ({duration_ms:.0f}ms): {e}")
            
            # 记录错误日志
            error_log = {
                "request_id": request_id,
                "node": node_name,
                "timestamp": datetime.now().isoformat(),
                "duration_ms": duration_ms,
                "error": str(e),
                "success": False
            }
            
            llm_calls = state.get("llm_calls", [])
            llm_calls.append(error_log)
            
            # 重新抛出异常或返回错误状态
            raise
    
    return llm_node


def print_call_summary(state: AgentState):
    """打印调用摘要"""
    llm_calls = state.get("llm_calls", [])
    
    if not llm_calls:
        logger.info("没有 LLM 调用记录")
        return
    
    logger.info("=" * 60)
    logger.info("LLM 调用摘要:")
    logger.info("=" * 60)
    
    successful_calls = [call for call in llm_calls if call.get("success", False)]
    failed_calls = [call for call in llm_calls if not call.get("success", False)]
    
    logger.info(f"总调用次数: {len(llm_calls)}")
    logger.info(f"成功: {len(successful_calls)}")
    logger.info(f"失败: {len(failed_calls)}")
    
    if successful_calls:
        total_duration = sum(call.get("duration_ms", 0) for call in successful_calls)
        avg_duration = total_duration / len(successful_calls) if successful_calls else 0
        logger.info(f"平均耗时: {avg_duration:.0f}ms")
        logger.info(f"总耗时: {total_duration:.0f}ms")
    
    # 详细日志
    for i, call in enumerate(llm_calls):
        status = "✅" if call.get("success") else "❌"
        logger.info(f"{status} 调用 {i+1}: {call.get('node', 'unknown')} - {call.get('duration_ms', 0):.0f}ms")