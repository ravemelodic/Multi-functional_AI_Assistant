"""
AgentState TypedDict for LangGraph.

Every node reads from and writes to this state, which flows through the
entire graph.  Fields are typed so LangGraph can track changes.
"""

from typing import TypedDict, Optional, Any


class AgentState(TypedDict):
    """
    The full state object that travels through the LangGraph workflow.

    Fields
    ------
    user_id : int
        Telegram user id.

    user_message : str
        Raw text from the user.

    intent : str | None
        Classified intent: "retrieve_course" | "rag_pdf" | "generate_video"
        | "analyze_image" | "general_chat" | None.

    course_code : str | None
        Extracted course code (e.g. "COMP7940") if present.

    rag_context : str
        Relevant chunks retrieved from Milvus vector store.

    rag_empty : bool
        True when RAG returned no results above the similarity threshold.

    conversation_memory_context : str
        Relevant past conversation turns retrieved from the memory vector store.

    augmented_prompt : str
        The final prompt sent to the LLM (rag_context + conversation_memory_context + user msg).

    final_response : str | None
        The LLM response sent back to the user.

    error : str | None
        Error message if any node failed.

    waiting_for_video_image : bool
        True when bot has asked for an image for video generation.

    waiting_for_video_prompt : bool
        True after image is received, waiting for prompt selection.

    video_image_base64 : str | None
        Base64-encoded image data for video generation.

    suggested_prompts : list[str]
        AI-suggested animation prompt options.

    video_task_id : str | None
        Celery task ID for the running video generation.

    celery_result : dict | None
        Result dict from a Celery task.
    """
    user_id: int
    user_message: str

    intent: Optional[str]
    course_code: Optional[str]
    rag_context: str
    rag_empty: bool
    conversation_memory_context: str
    augmented_prompt: str
    final_response: Optional[str]
    error: Optional[str]

    # Video workflow state
    waiting_for_video_image: bool
    waiting_for_video_prompt: bool
    video_image_base64: Optional[str]
    suggested_prompts: list[str]
    video_task_id: Optional[str]

    # Celery result
    celery_result: Optional[dict]

    # Queue fields — used when messages are processed asynchronously via Redis
    chat_id: Optional[int]           # Telegram chat ID, for sending the response
    reply_message_id: Optional[int]  # "Thinking..." message ID to edit

    # Raw Telegram objects (for nodes that need to send messages directly)
    _raw_update: Any
    _raw_context: Any
