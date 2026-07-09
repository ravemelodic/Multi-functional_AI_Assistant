"""
LangGraph node functions for the chatbot workflow.

Each function accepts the current AgentState dict (with a leading underscore
for the raw Telegram objects) and returns a dict of fields to update.
"""

import re
import io
import base64
import os
import logging
import asyncio
from typing import Any

from app.graph.state import AgentState
from app.configs.settings import settings
from app.llm import ChatGPT

# ------------------------------------------------------------------ #
#  Module-level globals (initialised once when the graph is built)    #
# ------------------------------------------------------------------ #
gpt_client: ChatGPT | None = None
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Initialisation helpers (called once at startup)                    #
# ------------------------------------------------------------------ #
async def init_global_resources():
    """Create the LLM client (called once at startup)."""
    global gpt_client
    gpt_client = ChatGPT(settings)
    logger.info("LLM client initialised.")


async def close_global_resources():
    """Tear down client (called once at shutdown)."""
    global gpt_client
    if gpt_client:
        await gpt_client.close()
        gpt_client = None


# =================================================================== #
#  INTENT CLASSIFICATION                                              #
# =================================================================== #

def classify_intent_node(state: AgentState) -> dict[str, Any]:
    """
    Examine the user_message and user_data flags to determine the intent.

    Returns a dict with at least ``intent`` set.  The router edge in the
    graph reads this field to decide which node to execute next.
    """
    ctx = state.get("_raw_context")
    user_data = getattr(ctx, "user_data", {}) if ctx else {}

    # 1. Check multi-step video workflow flags first ------------------
    if user_data.get("waiting_for_video_prompt"):
        return {
            "intent": "receive_video_prompt",
            "waiting_for_video_prompt": True,
        }

    if user_data.get("waiting_for_video_image"):
        return {
            "intent": "receive_video_image",
            "waiting_for_video_image": True,
        }

    msg = state["user_message"]

    # 2. /video command ----------------------------------------------
    if msg.strip().startswith("/video"):
        return {"intent": "video_command"}

    # 3. Course code detected (e.g. COMP7940) ------------------------
    course_match = re.search(r"[a-zA-Z]{4}\d{4}", msg)
    if course_match:
        return {
            "intent": "retrieve_course",
            "course_code": course_match.group(),
        }

    # 4. Attachments (documents / photos) are handled by Telegram
    #    handler BEFORE invoking the graph – not classified here.
    #    But if a text message arrives without any special pattern,
    #    it is a general chat.
    return {"intent": "general_chat"}


# =================================================================== #
#  COURSE RETRIEVAL (PostgreSQL exact match)                          #
# =================================================================== #


# =================================================================== #
#  CONVERSATION MEMORY RETRIEVAL (Milvus vector store)                #
# =================================================================== #

async def retrieve_conversation_memory_node(state: AgentState) -> dict[str, Any]:
    """
    Retrieve semantically relevant past conversation turns from the
    memory vector store, scoped to the current user.

    Returns the formatted memories as ``conversation_memory_context``
    (or empty string if nothing found / Milvus unavailable).
    """
    user_msg = state.get("user_message", "")
    user_id = state.get("user_id", 0)

    from rag.retriever import retrieve_conversation_memory

    try:
        docs_scores = await retrieve_conversation_memory(
            query=user_msg,
            user_id=user_id,
            k=5,
            score_threshold=0.45,
        )
    except Exception as exc:
        logger.warning("Conversation memory retrieval failed: %s", exc)
        return {"conversation_memory_context": ""}

    if not docs_scores:
        return {"conversation_memory_context": ""}

    # Format the top memories into a readable block
    lines = ["Relevant conversation history (from oldest to most recent):"]
    # Reverse so older memories appear first (chronological feel)
    docs_scores.reverse()
    for doc, score in docs_scores:
        lines.append(f"--- (relevance: {score:.2f})")
        lines.append(doc.page_content)

    mem_text = "\n".join(lines)
    logger.info(
        "Retrieved %d conversation memories for user %d",
        len(docs_scores), user_id,
    )
    return {"conversation_memory_context": mem_text}


# =================================================================== #
#  RAG RETRIEVAL (Milvus vector store)                                #
# =================================================================== #

async def retrieve_rag_node(state: AgentState) -> dict[str, Any]:
    """
    Retrieve semantically similar course chunks from Milvus using
    **Hybrid Search** (BM25 sparse + dense vector + RRF fusion).

    Falls back to pure dense vector search if the BM25 index is
    unavailable.  If no results pass the similarity threshold,
    ``rag_context`` is set to empty and ``rag_empty`` is flagged.
    """
    from rag.retriever import retrieve_hybrid_with_scores, retrieve_with_scores

    query = state.get("course_code") or state["user_message"]

    try:
        # Prefer hybrid search; falls back to pure dense internally
        docs_scores = await retrieve_hybrid_with_scores(
            query,
            k=5,
            score_threshold=0.5,
            dense_weight=settings.HYBRID_DENSE_WEIGHT,
            sparse_weight=settings.HYBRID_SPARSE_WEIGHT,
        )
    except Exception as exc:
        logger.warning("RAG retrieval failed (Milvus may not be running): %s", exc)
        return {"rag_context": "", "rag_empty": True}

    if not docs_scores:
        return {"rag_context": "", "rag_empty": True}

    rag_text = "\n\n".join(d.page_content for d, _ in docs_scores)

    # Log scores for debugging
    scores = [round(s, 3) for _, s in docs_scores]
    logger.info("RAG matched %d chunks (scores=%s)", len(docs_scores), scores)

    return {"rag_context": rag_text, "rag_empty": False}


# =================================================================== #
#  PROMPT ASSEMBLY                                                    #
# =================================================================== #

def build_prompt_node(state: AgentState) -> dict[str, Any]:
    """Assemble the final prompt sent to the LLM."""
    msg = state["user_message"]
    rag_ctx = state.get("rag_context", "")
    rag_empty = state.get("rag_empty", False)
    mem_ctx = state.get("conversation_memory_context", "")

    parts = []
    if rag_ctx:
        parts.append(f"Additional course information:\n{rag_ctx}")
    elif rag_empty:
        parts.append(
            "Note: No relevant course information was found. "
            "If the student is asking about a specific course you don't have "
            "information about, tell them honestly that you don't have data on "
            "that course. Do NOT make up course details or assign information "
            "from a different course to the one they asked about."
        )
    if mem_ctx:
        parts.append(
            f"Your past conversations with this student:\n{mem_ctx}\n\n"
            "(Note: The above is the student's past conversation history. "
            "Use it to maintain continuity — refer back to previously discussed "
            "topics when relevant. Do NOT mention that you are reading from a "
            "database or memory store.)"
        )

    parts.append(f"Student Question: {msg}")
    return {"augmented_prompt": "\n\n".join(parts)}


# =================================================================== #
#  LLM CALL                                                           #
# =================================================================== #

async def call_llm_node(state: AgentState) -> dict[str, Any]:
    """Send the augmented prompt to the ChatGPT API."""
    global gpt_client
    if gpt_client is None:
        return {"final_response": "Error: ChatGPT client not initialised."}

    prompt = state.get("augmented_prompt") or state["user_message"]
    try:
        resp = await gpt_client.submit(prompt)
        return {"final_response": resp}
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        return {"final_response": f"Sorry, I encountered an error: {exc}"}


# =================================================================== #
#  DATABASE LOGGING                                                   #
# =================================================================== #


# =================================================================== #
#  STORE CONVERSATION MEMORY (Milvus vector store)                    #
# =================================================================== #

async def store_conversation_memory_node(state: AgentState) -> dict[str, Any]:
    """
    Store the current conversation turn (user message + bot response)
    into the Milvus memory vector collection for future retrieval.

    This gives the bot persistent memory across sessions — the next time
    the user asks a related question, it will find this conversation turn.
    """
    user_id = state.get("user_id", 0)
    user_msg = state.get("user_message", "")
    bot_resp = state.get("final_response", "")

    # Skip empty responses or non-conversational turns
    if not user_msg or not bot_resp:
        return {}

    from rag.retriever import store_conversation_memory

    try:
        ok = await store_conversation_memory(
            user_id=user_id,
            user_message=user_msg,
            bot_response=bot_resp,
        )
        if ok:
            logger.debug("Stored conversation memory for user %d", user_id)
        else:
            logger.debug("Skipped memory storage for user %d (Milvus unavailable)", user_id)
    except Exception as exc:
        logger.warning("Failed to store conversation memory: %s", exc)

    return {}


# =================================================================== #
#  VIDEO COMMAND HANDLER                                              #
# =================================================================== #

def video_command_node(state: AgentState) -> dict[str, Any]:
    """
    Handle /video command – sets flags so the next image message is
    captured for video generation.  The actual reply is sent via the
    Telegram handler (outside the graph) because the node only returns
    state updates.
    """
    return {
        "intent": "generate_video",
        "waiting_for_video_image": True,
        "waiting_for_video_prompt": False,
        "final_response": (
            "Video Mode\n\n"
            "Step 1: Please send me an image (photo or document)\n"
            "Step 2: I will analyse it and suggest animation prompts\n\n"
            "Send your image now!"
        ),
    }


async def receive_video_image_node(state: AgentState) -> dict[str, Any]:
    """
    Called when the user sends an image while ``waiting_for_video_image`` is set.
    Downloads the image, submits an ``analyze_image`` Celery task, and waits
    for the result.
    """
    update = state.get("_raw_update")
    ctx = state.get("_raw_context")

    if not update or not ctx:
        return {"error": "No Telegram update/context in state"}

    # -- grab the file ---------------------------------------------------
    if update.message.photo:
        photo = update.message.photo[-1]
        file = await photo.get_file()
    elif update.message.document:
        file = await update.message.document.get_file()
    else:
        return {"final_response": "Please send an image file."}

    # -- download & base64-encode ---------------------------------------
    try:
        image_bytes = io.BytesIO()
        await file.download_to_memory(image_bytes)
        image_bytes.seek(0)
        b64 = base64.b64encode(image_bytes.read()).decode("utf-8")

        fpath = file.file_path or ""
        mime = "image/jpeg"
        if fpath.lower().endswith(".png"):
            mime = "image/png"
        elif fpath.lower().endswith(".gif"):
            mime = "image/gif"
        elif fpath.lower().endswith(".webp"):
            mime = "image/webp"

        image_data_uri = f"data:{mime};base64,{b64}"
    except Exception as exc:
        logger.error("Failed to download/encode image: %s", exc)
        return {
            "error": str(exc),
            "final_response": "Failed to process the image. Please try again.",
        }

    # -- submit Celery image-analysis task -------------------------------
    from tasks import analyze_image_task

    try:
        task = analyze_image_task.apply_async(
            args=[image_data_uri, state["user_id"]],
            queue="ocr",
        )
        result: dict = task.get(timeout=30)
    except Exception as exc:
        logger.error("Image analysis task failed: %s", exc)
        # Graceful fallback – allow manual prompt entry
        return {
            "video_image_base64": image_data_uri,
            "waiting_for_video_image": False,
            "waiting_for_video_prompt": True,
            "final_response": (
                "Image received!\n\n"
                " AI analysis is currently unavailable. "
                "Please describe how you want the video to be animated.\n\n"
                "Examples:\n"
                "- 'smooth zoom in effect'\n"
                "- 'gentle camera pan from left to right'\n\n"
                "Or send 'default' for smooth natural animation."
            ),
        }

    if result.get("success"):
        analysis = result["analysis"]
        suggested = result.get("suggested_prompts", [])

        # Build numbered prompt list
        prompt_lines = "\n".join(
            f"{i+1}. {p}" for i, p in enumerate(suggested)
        )
        reply = (
            f"Image received!\n\n"
            f" AI Analysis:\n{analysis}\n\n"
            "----------------------------------------\n"
            " Quick select: Send 1, 2, or 3 to choose a suggested prompt\n"
            " Or type your own custom prompt\n"
            " Or send 'default' for smooth natural animation"
        )
        return {
            "video_image_base64": image_data_uri,
            "suggested_prompts": suggested,
            "waiting_for_video_image": False,
            "waiting_for_video_prompt": True,
            "final_response": reply,
        }
    else:
        return {
            "video_image_base64": image_data_uri,
            "waiting_for_video_image": False,
            "waiting_for_video_prompt": True,
            "final_response": (
                "Image received!\n\n"
                " AI analysis unavailable. "
                "Please describe how you want the video to be animated.\n\n"
                "Or send 'default' for smooth natural animation."
            ),
        }


async def receive_video_prompt_node(state: AgentState) -> dict[str, Any]:
    """
    Called when ``waiting_for_video_prompt`` is True.
    The user has sent a prompt choice (1, 2, 3, custom, or 'default').
    Submits a Celery ``generate_video`` task and starts a background monitor.
    """
    user_input = state["user_message"]
    image_b64 = state.get("video_image_base64")
    if not image_b64:
        return {"final_response": "Error: Image data lost. Please start over with /video"}

    # Resolve the prompt ---------------------------------------------------
    suggested = state.get("suggested_prompts", [])
    if user_input in ("1", "2", "3") and suggested:
        idx = int(user_input) - 1
        if 0 <= idx < len(suggested):
            user_prompt = suggested[idx]
        else:
            user_prompt = user_input
    elif user_input.lower() == "default":
        user_prompt = "smooth natural animation"
    else:
        user_prompt = user_input

    output_video = f"/comp7940-lab/temp/output_video_{state['user_id']}.mp4"

    # Submit Celery task ---------------------------------------------------
    from tasks import generate_video_task

    try:
        task = generate_video_task.apply_async(
            args=[image_b64, user_prompt, state["user_id"], output_video],
            queue="video",
        )
    except Exception as exc:
        logger.error("Failed to submit video task: %s", exc)
        return {"final_response": f"Error submitting video task: {exc}"}

    # Start background monitor (fire-and-forget) ---------------------------
    update = state.get("_raw_update")
    ctx = state.get("_raw_context")
    if update and ctx:
        asyncio.create_task(
            _monitor_video_task(update, ctx, task, output_video, user_prompt)
        )

    return {
        "video_task_id": task.id,
        "waiting_for_video_prompt": False,
        "final_response": (
            f"Video generation started!\nPrompt: {user_prompt}\n\n"
            "Your video is being processed in the background.\n"
            "This usually takes 2-10 minutes.\n"
            "I will send you the video when it is ready!\n\n"
            "You can continue chatting with me while waiting."
        ),
    }


async def _monitor_video_task(update, ctx, task, output_video, user_prompt):
    """Background monitor – polls Celery task status and sends updates."""
    user_id = update.effective_user.id
    last_status = None

    try:
        while not task.ready():
            info = task.info or {}
            status = info.get("status") if isinstance(info, dict) else None
            position = info.get("position", 0) if isinstance(info, dict) else 0

            if status and status != last_status and status in ("InQueue", "InProgress"):
                last_status = status
                emoji = {"InQueue": "Pending", "InProgress": "Processing"}.get(status, "Processing")
                msg_text = (
                    f"{emoji} Your video is queued (Position: {position})"
                    if status == "InQueue"
                    else f"{emoji} Your video is being processed..."
                )
                try:
                    await ctx.bot.send_message(chat_id=user_id, text=msg_text)
                except Exception:
                    pass

            await asyncio.sleep(10)

        result = task.get()
        if result["success"] and os.path.exists(output_video):
            await ctx.bot.send_message(chat_id=user_id, text="Video generated! Uploading...")
            with open(output_video, "rb") as f:
                await ctx.bot.send_video(
                    chat_id=user_id,
                    video=f,
                    caption=f"Your generated video is ready!\nPrompt: {user_prompt}",
                    supports_streaming=True,
                )

            os.remove(output_video)
        else:
            err = result.get("error", "Unknown error")
            await ctx.bot.send_message(
                chat_id=user_id,
                text=f"Video generation failed: {err}\n\nPlease try again later.",
            )
    except Exception as exc:
        logger.error("Video monitoring error: %s", exc)
        await ctx.bot.send_message(
            chat_id=user_id,
            text=f"An error occurred: {exc}\n\nPlease try again later.",
        )


# =================================================================== #
#  DOCUMENT ANALYSIS                                                   #
# =================================================================== #

async def analyze_document_node(state: AgentState) -> dict[str, Any]:
    """Process a PDF upload via Celery OCR worker."""
    update = state.get("_raw_update")
    ctx = state.get("_raw_context")
    if not update or not ctx:
        return {"error": "No Telegram update/context in state"}

    file = update.message.document
    if not file:
        return {"final_response": "Please send a PDF file."}

    file_name = file.file_name or "document"
    if not file_name.lower().endswith(".pdf"):
        return {
            "final_response": (
                "Document Analysis\n\n"
                "Currently only PDF files are supported.\n"
                "For images, please use the /video command for image-to-video conversion."
            )
        }

    temp_path = f"/comp7940-lab/temp/doc_{state['user_id']}_{file.file_unique_id}.pdf"
    try:
        telegram_file = await file.get_file()
        await telegram_file.download_to_drive(temp_path)

        from tasks import analyze_document_task

        task = analyze_document_task.apply_async(
            args=[temp_path, "pdf", state["user_id"]],
            queue="ocr",
        )
        result = task.get(timeout=180)

        if result["success"]:
            summary = result["summary"]

            return {
                "final_response": (
                    f"Document Analysis Result\n\n"
                    f"File: {file_name}\n\n{summary}"
                )
            }
        else:
            return {
                "final_response": (
                    f"Document analysis failed\n\n"
                    f"Error: {result.get('error', 'Unknown error')}\n\n"
                    f"Please try again or contact support."
                )
            }
    except Exception as exc:
        logger.error("Document analysis error: %s", exc)
        return {"final_response": f"An error occurred during document analysis:\n{exc}"}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
