"""
LangGraph StateGraph definition for the chatbot workflow.

Builds and exposes a compiled ``app`` that can be called with
``await app.ainvoke(state_dict)``.
"""

import logging
from typing import Literal

from langgraph.graph import StateGraph, END

from app.graph.state import AgentState
from app.graph.nodes import (
    classify_intent_node,
    retrieve_rag_node,
    retrieve_conversation_memory_node,
    build_prompt_node,
    call_llm_node,
    store_conversation_memory_node,
    video_command_node,
    receive_video_image_node,
    receive_video_prompt_node,
    analyze_document_node,
)

logger = logging.getLogger(__name__)


# =================================================================== #
#  Router function – reads ``intent`` from state                       #
# =================================================================== #

def intent_router(state: AgentState) -> Literal[
    "video_command",
    "receive_video_image",
    "receive_video_prompt",
    "retrieve_course",
    "analyze_document",
    "general_chat",
]:
    """
    Map the classified intent to the next node name.

    Returns the string key of the node to invoke next.
    """
    intent = state.get("intent")

    mapping = {
        "video_command": "video_command",
        "receive_video_image": "receive_video_image",
        "receive_video_prompt": "receive_video_prompt",
        "retrieve_course": "retrieve_course",
        "analyze_document": "analyze_document",
        "general_chat": "general_chat",
    }
    return mapping.get(intent, "general_chat")


# =================================================================== #
#  Graph construction                                                  #
# =================================================================== #

def build_graph() -> StateGraph:
    """Create and compile the LangGraph StateGraph."""
    workflow = StateGraph(AgentState)

    # ------------------------------------------------------------------ #
    #  Register nodes                                                    #
    # ------------------------------------------------------------------ #
    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("video_command", video_command_node)
    workflow.add_node("receive_video_image", receive_video_image_node)
    workflow.add_node("receive_video_prompt", receive_video_prompt_node)
    workflow.add_node("retrieve_rag", retrieve_rag_node)
    workflow.add_node("retrieve_memory", retrieve_conversation_memory_node)
    workflow.add_node("build_prompt", build_prompt_node)
    workflow.add_node("call_llm", call_llm_node)
    workflow.add_node("store_memory", store_conversation_memory_node)
    workflow.add_node("analyze_document", analyze_document_node)

    # ------------------------------------------------------------------ #
    #  Set entry point                                                   #
    # ------------------------------------------------------------------ #
    workflow.set_entry_point("classify_intent")

    # ------------------------------------------------------------------ #
    #  Conditional edge from classify_intent                              #
    # ------------------------------------------------------------------ #
    workflow.add_conditional_edges(
        "classify_intent",
        intent_router,
        {
            "video_command": "video_command",
            "receive_video_image": "receive_video_image",
            "receive_video_prompt": "receive_video_prompt",
            "retrieve_course": "retrieve_rag",
            "analyze_document": "analyze_document",
            "general_chat": "retrieve_rag",
        },
    )

    # ------------------------------------------------------------------ #
    #  Fixed edges                                                       #
    # ------------------------------------------------------------------ #

    # Video command and image prompt branches → end after single node
    # 由于用户每一次与AI对话的交互（发送图片，图片解析，选择prompt,生成视频每一个都属于不同的对话轮次-一次Q&A属于一轮），因此其每一个节点独立且都指向END
    workflow.add_edge("video_command", END)
    workflow.add_edge("receive_video_image", END)
    workflow.add_edge("receive_video_prompt", END)

    # Course pipeline: retrieve_rag → retrieve_memory
    # retrieve rag 用于处理课程相关的检索
    # retrieve rag 执行了完整的课程相关的混合检索流程
    # retrieve rag 返回的是rag_context
    workflow.add_edge("retrieve_rag", "retrieve_memory")

    # Memory retrieval feeds into prompt builder (for both general & course paths)
    # 其通过从以往的conversation_memory集合（Milvus）中检索出与当前用户输入相关的历史对话写入conversation_memory_context，来为后续的prompt构建提供上下文信息
    workflow.add_edge("retrieve_memory", "build_prompt")

    # Conversation pipeline: build_prompt → call_llm → store_memory
    # build prompt 用于构建最终的prompt，返回的是final_prompt
    # call llm 用于调用llm生成最终的response，返回的是llm_response
    # store_memory 把 (user_message + final_response) embed 后写入 conversation_memory 集合，不走 conversation_memory_context，也不走 RAG 混合索引。
    workflow.add_edge("build_prompt", "call_llm")
    workflow.add_edge("call_llm", "store_memory")
    workflow.add_edge("store_memory", END)

    # Document analysis → memory retrieval → LLM (same pipeline as general chat)
    # analyze document 用于分析用户上传的文档，文档与摘要会被写入rag_context(与retrieve_rag的rag_context是同一个共享变量，但是不会冲突，因为它们不可能在同一轮对话同时执行)
    # Document analysis → RAG (PDF is already in Milvus after await ingest_text)
    workflow.add_edge("analyze_document", "retrieve_rag")

    # ------------------------------------------------------------------ #
    #  Compile                                                           #
    # ------------------------------------------------------------------ #
    app = workflow.compile()
    logger.info("LangGraph workflow compiled successfully.")
    return app


# Module-level compiled graph – import and use everywhere.
app = build_graph()
