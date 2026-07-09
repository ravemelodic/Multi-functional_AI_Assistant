"""
LangGraph StateGraph definition for the chatbot workflow.

Builds and exposes a compiled ``app`` that can be called with
``await app.ainvoke(state_dict)``.
"""

import logging
from typing import Literal

from langgraph.graph import StateGraph, END

from graph.state import AgentState
from graph.nodes import (
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
            "general_chat": "retrieve_memory",
        },
    )

    # ------------------------------------------------------------------ #
    #  Fixed edges                                                       #
    # ------------------------------------------------------------------ #

    # Video command and image prompt branches → end after single node
    workflow.add_edge("video_command", END)
    workflow.add_edge("receive_video_image", END)
    workflow.add_edge("receive_video_prompt", END)

    # Course pipeline: retrieve_rag → retrieve_memory
    workflow.add_edge("retrieve_rag", "retrieve_memory")

    # Memory retrieval feeds into prompt builder (for both general & course paths)
    workflow.add_edge("retrieve_memory", "build_prompt")

    # Conversation pipeline: build_prompt → call_llm → store_memory
    workflow.add_edge("build_prompt", "call_llm")
    workflow.add_edge("call_llm", "store_memory")
    workflow.add_edge("store_memory", END)

    # Document analysis → end
    workflow.add_edge("analyze_document", END)

    # ------------------------------------------------------------------ #
    #  Compile                                                           #
    # ------------------------------------------------------------------ #
    app = workflow.compile()
    logger.info("LangGraph workflow compiled successfully.")
    return app


# Module-level compiled graph – import and use everywhere.
app = build_graph()
