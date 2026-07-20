"""
Telegram Bot Agent – refactored to use LangGraph orchestration.

Each incoming Telegram update is pre-processed by the appropriate handler,
which builds an initial ``AgentState`` dict and forwards it to the compiled
LangGraph graph via ``app.ainvoke()``.

Includes production hardening:
- Per-user sliding-window rate limiter
- Input message length validation
- Global concurrency throttle
"""

import io
import base64
import json
import logging
import os
import asyncio
import time
from collections import defaultdict
from collections import deque
import redis.asyncio

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    MessageHandler,
    ContextTypes,
    filters,
    CommandHandler,
)

from app.graph.state import AgentState
from app.graph.workflow import app as langgraph_app
from app.graph.nodes import init_global_resources, close_global_resources
from app.configs.settings import settings

# ------------------------------------------------------------------ #
#  Logging                                                           #
# ------------------------------------------------------------------ #
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=getattr(logging, settings.BOT_LOG_LEVEL.upper(), logging.INFO),
    handlers=[
        logging.FileHandler(f"{settings.LOG_DIR}/bot.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Log-safe helper – truncate sensitive user data at INFO level       #
# ------------------------------------------------------------------ #

def _safe_msg(msg: str, max_len: int = 50) -> str:
    """
    Return a truncated version of the user message for INFO-level logs.
    At DEBUG level the full message is logged.
    At INFO level (and above), only the first ``max_len`` characters are kept
    — enough to understand intent, not enough to leak full content.
    """
    if logger.isEnabledFor(logging.DEBUG):
        return msg
    if len(msg) <= max_len:
        return msg
    return msg[:max_len] + "..."

# =================================================================== #
#  Rate Limiter (per-user sliding window, in-memory)                  #
# =================================================================== #

class RateLimiter:
    """
    Per-user sliding-window rate limiter.

    Tracks the timestamps of recent messages per user and rejects
    requests that exceed ``max_requests`` within ``window_seconds``.
    """

    def __init__(self, max_requests: int = 20, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._buckets: dict[int, deque] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def check(self, user_id: int) -> tuple[bool, int]:
        """
        Check if a request from ``user_id`` is allowed.

        Returns
        -------
        (allowed: bool, retry_after_seconds: int)
        """
        now = time.time()
        async with self._lock:
            bucket = self._buckets[user_id]
            # Prune old entries
            cutoff = now - self.window_seconds
            while bucket and bucket[0] < cutoff:
                bucket.popleft()

            if len(bucket) >= self.max_requests:
                # Calculate when the oldest entry expires
                retry_after = int(bucket[0] + self.window_seconds - now) + 1
                return False, retry_after

            bucket.append(now)
            return True, 0

    @property
    def active_users(self) -> int:
        """Return the number of users with recent activity."""
        return len(self._buckets)


# Module-level rate limiter instance
rate_limiter = RateLimiter(max_requests=30, window_seconds=60)

# Redis queue for conversation processing
QUEUE_KEY = "conversation_queue"
_redis_client: redis.asyncio.Redis | None = None
_queue_consumer_started = False
_queue_semaphore = asyncio.Semaphore(20)  # max 20 concurrent queue consumers

# =================================================================== #
#  Input validation                                                   #
# =================================================================== #

MAX_MESSAGE_LENGTH = 4096  # characters – reject excessively long messages


def validate_message(text: str) -> tuple[bool, str]:
    """
    Validate an incoming user message.

    Returns (is_valid, error_message).
    """
    if not text or not text.strip():
        return False, "Empty message received."

    if len(text) > MAX_MESSAGE_LENGTH:
        return (
            False,
            f"Message too long ({len(text)} characters). "
            f"Please keep messages under {MAX_MESSAGE_LENGTH} characters.",
        )

    # Check for obviously abusive patterns (excessive repetition)
    # e.g. "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa..."
    if len(text) > 200:
        # If more than 70% of chars are the same 3 characters → likely spam
        from collections import Counter
        top3 = Counter(text.lower()).most_common(3)
        top3_total = sum(count for _, count in top3)
        if top3_total / len(text) > 0.70:
            return False, "Your message appears to be spam. Please send a normal message."

    return True, ""


# =================================================================== #
#  Global concurrency throttle                                        #
# =================================================================== #

_graph_semaphore = asyncio.Semaphore(50)  # max 50 concurrent graph invocations


# =================================================================== #
#  TEXT MESSAGE HANDLER  (main entry into the graph)                  #
# =================================================================== #
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handles plain text messages (and commands that are not pre-routed).

    Includes rate limiting, input validation, and concurrency throttling.
    """
    user_id = update.effective_user.id
    user_msg = update.message.text.strip()
    logger.info("Text from user %d: %s", user_id, _safe_msg(user_msg))

    # -- Rate limiting --------------------------------------------------
    allowed, retry_after = await rate_limiter.check(user_id)
    if not allowed:
        logger.warning("Rate limit hit for user %d (retry after %ds)", user_id, retry_after)
        await update.message.reply_text(
            f"⏱️ You're sending messages too fast. "
            f"Please wait {retry_after} seconds before sending another message."
        )
        return

    # -- Input validation -----------------------------------------------
    is_valid, error_msg = validate_message(user_msg)
    if not is_valid:
        logger.warning("Invalid message from user %d: %s", user_id, error_msg)
        await update.message.reply_text(error_msg)
        return

    # -- Pre-detect intent from user_data flags -------------------------
    intent: str | None = None
    if context.user_data.get("waiting_for_video_prompt"):
        intent = "receive_video_prompt"

    # -- Build initial AgentState ---------------------------------------
    initial_state: AgentState = {
        "user_id": user_id,
        "user_message": user_msg,
        "intent": intent,
        "course_code": None,
        "db_context": "",
        "rag_context": "",
        "rag_empty": False,
        "conversation_memory_context": "",
        "augmented_prompt": "",
        "final_response": None,
        "error": None,
        "waiting_for_video_image": context.user_data.get("waiting_for_video_image", False),
        "waiting_for_video_prompt": context.user_data.get("waiting_for_video_prompt", False),
        "video_image_base64": context.user_data.get("video_image_base64"),
        "suggested_prompts": context.user_data.get("suggested_prompts", []),
        "video_task_id": None,
        "celery_result": None,
        "_raw_update": update,
        "_raw_context": context,
    }

    loading_msg = await update.message.reply_text("Thinking...")

    try:
        # Run graph with timeout and concurrency throttle
        async with _graph_semaphore:
            result = await asyncio.wait_for(
                langgraph_app.ainvoke(initial_state),
                timeout=30.0,  # 30s max per conversation
            )
    except asyncio.TimeoutError:
        logger.warning("Graph invocation timed out for user %d", user_id)
        await loading_msg.edit_text(
            "Sorry, the request timed out. The AI service may be "
            "experiencing high traffic. Please try again in a moment."
        )
        return
    except Exception as exc:
        logger.exception("Graph invocation failed for user %d", user_id)
        await loading_msg.edit_text(f"Sorry, something went wrong: {exc}")
        return

    # -- Sync user_data flags from result -------------------------------
    _sync_user_data(context, result)

    # -- Send the final response ----------------------------------------
    response = result.get("final_response")
    if response:
        # Truncate response if it exceeds Telegram's limit (4096 chars)
        if len(response) > 4096:
            response = response[:4093] + "..."
        try:
            await loading_msg.edit_text(response)
        except Exception as exc:
            logger.error("Failed to send response to user %d: %s", user_id, exc)
    else:
        await loading_msg.edit_text("I received your message but have nothing to say yet.")


# =================================================================== #
#  /video COMMAND HANDLER                                             #
# =================================================================== #
async def handle_video_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Initiate the image-to-video workflow."""
    await update.message.reply_text(
        "Image to Video Mode\n\n"
        "Step 1: Please send me an image (photo or document)\n"
        "Step 2: I will analyse it and suggest animation prompts\n\n"
        "Send your image now!"
    )
    context.user_data["waiting_for_video_image"] = True
    context.user_data["waiting_for_video_prompt"] = False


# =================================================================== #
#  PHOTO / DOCUMENT HANDLER                                           #
# =================================================================== #
async def handle_attachment(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handles image and document uploads.

    Routes:
    - Video mode (waiting_for_video_image) → receive_video_image graph node
    - PDF document → receive_video_image  → receive_video_prompt  → ... graph node
    - Standalone photo → informational message asking to use /video
    """
    # -- Video mode: user sent an image to animate ----------------------
    if context.user_data.get("waiting_for_video_image"):
        await _route_video_image(update, context)
        return

    # -- Document (PDF) analysis -----------------------------------------
    if update.message.document:
        doc = update.message.document
        file_name = doc.file_name or "document"

        if not file_name.lower().endswith(".pdf"):
            await update.message.reply_text(
                "Document Analysis\n\n"
                "Currently only PDF files are supported.\n"
                "For images, please use the /video command for image-to-video conversion."
            )
            return

        await _route_document_analysis(update, context, doc, file_name)
        return

    # -- Standalone photo (not in video mode) ---------------------------
    await update.message.reply_text(
        "Image received!\n\n"
        "To convert this image to video, please use the /video command first."
    )


async def _route_video_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """User is in video-image-collection mode: invoke the graph."""
    initial_state: AgentState = {
        "user_id": update.effective_user.id,
        "user_message": "[Image upload for video]",
        "intent": "receive_video_image",
        "course_code": None,
        "db_context": "",
        "rag_context": "",
        "rag_empty": False,
        "augmented_prompt": "",
        "final_response": None,
        "error": None,
        "waiting_for_video_image": True,
        "waiting_for_video_prompt": False,
        "video_image_base64": None,
        "suggested_prompts": [],
        "video_task_id": None,
        "celery_result": None,
        "_raw_update": update,
        "_raw_context": context,
    }

    try:
        result = await langgraph_app.ainvoke(initial_state)
    except Exception as exc:
        logger.exception("Graph invocation failed for video image")
        await update.message.reply_text(f"Error processing image: {exc}")
        return

    _sync_user_data(context, result)

    resp = result.get("final_response")
    if resp:
        await update.message.reply_text(resp)


async def _route_document_analysis(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    doc,
    file_name: str,
):
    """User uploaded a PDF: invoke the graph for analysis."""
    status_msg = await update.message.reply_text(
        "Document Analysis\n\n"
        "Downloading PDF...\n"
        "This may take a moment..."
    )

    # Capture user's caption (e.g. "只分析第三章") as the query
    user_caption = (update.message.caption or "").strip()
    user_query = (
        user_caption
        if user_caption
        else f"请分析这份文档 {file_name}，提取关键信息、截止日期和要求"
    )

    initial_state: AgentState = {
        "user_id": update.effective_user.id,
        "user_message": f"[Document: {file_name}] {user_query}",
        "intent": "analyze_document",
        "course_code": None,
        "db_context": "",
        "rag_context": "",
        "rag_empty": False,
        "augmented_prompt": "",
        "final_response": None,
        "error": None,
        "waiting_for_video_image": False,
        "waiting_for_video_prompt": False,
        "video_image_base64": None,
        "suggested_prompts": [],
        "video_task_id": None,
        "celery_result": None,
        "_raw_update": update,
        "_raw_context": context,
    }

    try:
        result = await langgraph_app.ainvoke(initial_state)
    except Exception as exc:
        logger.exception("Graph invocation failed for document analysis")
        await status_msg.edit_text(f"Error analysing document: {exc}")
        return

    _sync_user_data(context, result)

    resp = result.get("final_response")
    if resp:
        await status_msg.edit_text(resp)


# =================================================================== #
#  UTILITY: sync LangGraph result back into Telegram user_data         #
# =================================================================== #

# =================================================================== #
#  SYNC PATH — runs the graph directly (video workflow)               #
# =================================================================== #

async def _run_sync(
    state: AgentState, update: Update, context: ContextTypes.DEFAULT_TYPE, user_id: int,
) -> None:
    """Execute the graph synchronously (for video workflow that needs raw Telegram objects)."""
    loading_msg = await update.message.reply_text("Thinking...")
    try:
        async with _graph_semaphore:
            result = await asyncio.wait_for(
                langgraph_app.ainvoke(state),
                timeout=30.0,
            )
    except asyncio.TimeoutError:
        logger.warning("Graph invocation timed out for user %d", user_id)
        await loading_msg.edit_text(
            "Sorry, the request timed out. The AI service may be "
            "experiencing high traffic. Please try again in a moment."
        )
        return
    except Exception as exc:
        logger.exception("Graph invocation failed for user %d", user_id)
        await loading_msg.edit_text(f"Sorry, something went wrong: {exc}")
        return

    _sync_user_data(context, result)
    response = result.get("final_response")
    if response:
        if len(response) > 4096:
            response = response[:4093] + "..."
        try:
            await loading_msg.edit_text(response)
        except Exception:
            pass
    else:
        await loading_msg.edit_text("I received your message but have nothing to say yet.")


# =================================================================== #
#  QUEUE PATH — enqueue for async processing (削峰填谷)                #
# =================================================================== #

async def _enqueue_message(
    state: AgentState, update: Update, user_id: int,
) -> None:
    """Push the message to Redis queue and reply immediately with a placeholder."""
    loading_msg = await update.message.reply_text("Thinking...")

    # Build a JSON-serialisable copy (strip raw Telegram objects)
    queue_state = {}
    for k, v in state.items():
        if k in ("_raw_update", "_raw_context"):
            continue
        # Convert non-serialisable types to str
        try:
            json.dumps(v)
            queue_state[k] = v
        except (TypeError, ValueError):
            queue_state[k] = str(v)

    queue_state["chat_id"] = update.effective_chat.id
    queue_state["reply_message_id"] = loading_msg.message_id

    try:
        if _redis_client is not None:
            await _redis_client.rpush(QUEUE_KEY, json.dumps(queue_state, ensure_ascii=False))
            logger.debug("Enqueued message for user %d", user_id)
        else:
            logger.warning("Redis unavailable, falling back to sync for user %d", user_id)
            await _run_sync(state, update, None, user_id)  # type: ignore
    except Exception as exc:
        logger.warning("Redis enqueue failed for user %d, fallback to sync: %s", user_id, exc)
        await _run_sync(state, update, None, user_id)  # type: ignore


# =================================================================== #
#  QUEUE CONSUMER — background worker that processes queued messages   #
# =================================================================== #

async def conversation_queue_consumer(bot_instance) -> None:
    """
    Background asyncio task: consumes messages from Redis queue,
    runs LangGraph, and sends the response back to the user.
    """
    global _queue_consumer_started
    if _queue_consumer_started:
        return
    _queue_consumer_started = True

    logger.info("Queue consumer started (channel=%s)", QUEUE_KEY)

    while True:
        try:
            if _redis_client is None:
                await asyncio.sleep(1)
                continue

            _, data = await _redis_client.blpop(QUEUE_KEY, timeout=30)
            if not data:
                continue

            state = json.loads(data)
            user_id = state.get("user_id", "?")

            async with _queue_semaphore:
                try:
                    result = await asyncio.wait_for(
                        langgraph_app.ainvoke(state),
                        timeout=30.0,
                    )
                except asyncio.TimeoutError:
                    logger.warning("Queue task timed out for user %s", user_id)
                    try:
                        await bot_instance.edit_message_text(
                            chat_id=state["chat_id"],
                            message_id=state["reply_message_id"],
                            text="Sorry, the request timed out. Please try again.",
                        )
                    except Exception:
                        pass
                    continue
                except Exception as exc:
                    logger.exception("Queue ainvoke failed for user %s", user_id)
                    try:
                        await bot_instance.edit_message_text(
                            chat_id=state["chat_id"],
                            message_id=state["reply_message_id"],
                            text=f"Sorry, something went wrong: {exc}",
                        )
                    except Exception:
                        pass
                    continue

            # Send response back to user
            response = result.get("final_response", "")
            if not response:
                response = "I received your message but have nothing to say yet."
            if len(response) > 4096:
                response = response[:4093] + "..."

            try:
                await bot_instance.edit_message_text(
                    chat_id=state["chat_id"],
                    message_id=state["reply_message_id"],
                    text=response,
                )
            except Exception as exc:
                logger.warning("Failed to send queue response for user %s: %s", user_id, exc)

        except Exception as exc:
            logger.error("Queue consumer error: %s", exc)
            await asyncio.sleep(1)


# =================================================================== #
#  Redis lifecycle helpers                                              #
# =================================================================== #

async def _init_queue():
    """Create the Redis async connection (with optional password)."""
    global _redis_client
    try:
        _redis_client = redis.asyncio.Redis(
            host=settings.REDIS_HOST,
            port=int(settings.REDIS_PORT),
            password=settings.REDIS_PASSWORD or None,
            decode_responses=True,
        )
        await _redis_client.ping()
        logger.info("Redis queue connection established (%s:%s)", settings.REDIS_HOST, settings.REDIS_PORT)
    except Exception as exc:
        logger.warning("Redis queue unavailable, messages will be processed synchronously: %s", exc)
        _redis_client = None


async def _close_queue():
    """Close the Redis connection."""
    global _redis_client
    if _redis_client is not None:
        try:
            await _redis_client.close()
        except Exception:
            pass
        _redis_client = None


def _sync_user_data(context: ContextTypes.DEFAULT_TYPE, result: dict):
    """Copy state fields that represent conversational flags back to user_data."""
    flags = [
        "waiting_for_video_image",
        "waiting_for_video_prompt",
        "suggested_prompts",
    ]
    for key in flags:
        if key in result:
            context.user_data[key] = result[key]

    # video_image_base64 is stored but must be carried in state only
    if "video_image_base64" in result:
        context.user_data["video_image_base64"] = result["video_image_base64"]


# =================================================================== #
#  APPLICATION INITIALISATION                                          #
# =================================================================== #
async def init_app():
    """Create and return the configured Telegram Application."""
    logger.info("Initialising Telegram Bot Agent (LangGraph) ...")

    # Initialise DB pool + LLM client (used by graph nodes)
    await init_global_resources()

    # Build the PTB application
    app = ApplicationBuilder().token(settings.TELEGRAM_ACCESS_TOKEN).build()

    # Register handlers
    app.add_handler(CommandHandler("video", handle_video_command))
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text)
    )
    app.add_handler(
        MessageHandler(
            filters.PHOTO | filters.Document.PDF | filters.Document.IMAGE,
            handle_attachment,
        )
    )

    logger.info("Bot agent initialised successfully.")
    return app


# =================================================================== #
#  ENTRY POINT                                                        #
# =================================================================== #
if __name__ == "__main__":
    async def main():
        application = await init_app()

        # Initialise Redis queue
        await _init_queue()

        # Start queue consumer as background task
        bot_instance = application.bot
        asyncio.create_task(conversation_queue_consumer(bot_instance))

        async with application:
            await application.start()
            await application.updater.start_polling()

            try:
                await asyncio.Event().wait()  # run forever
            except (KeyboardInterrupt, SystemExit):
                logger.info("Shutting down ...")
            finally:
                await _close_queue()
                await application.updater.stop()
                await application.stop()
                await close_global_resources()

    asyncio.run(main())
