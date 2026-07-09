"""
ChatGPT client – async/sync REST wrapper for Azure OpenAI.

Supports both the legacy ``configparser.ConfigParser`` initialisation
(used by Celery workers in ``tasks.py``) and the new Pydantic ``Settings``
object (used by the LangGraph nodes).

Includes retry with exponential backoff, circuit-breaker state tracking,
and configurable timeouts for production resilience.
"""

import asyncio
import httpx
import configparser
import logging
from typing import Union

logger = logging.getLogger(__name__)

# Allow duck-typing: anything with .CHATGPT_API_KEY / .CHATGPT_BASE_URL / …
# is accepted.  We import the concrete type only for type hints.
try:
    from app.configs.settings import Settings as _SettingsType
except ImportError:
    _SettingsType = None  # not available when run as __main__


class ChatGPT:
    """
    Async + sync HTTP client for the HKBU Azure OpenAI chat-completion API.

    Parameters
    ----------
    config : Union[configparser.SectionProxy-like, Settings, configparser.ConfigParser]
        Provide a ``configparser.ConfigParser`` (as used by Celery workers) or
        a ``configs.settings.Settings`` instance (as used by the LangGraph bot).
    """

    # Circuit-breaker defaults
    _circuit_state = "CLOSED"       # CLOSED / HALF_OPEN / OPEN
    _circuit_failures = 0
    _circuit_threshold = 5          # trips after 5 consecutive failures
    _circuit_reset_after = 30.0     # seconds before trying again
    _last_failure_time = 0.0

    def __init__(self, config):
        # Normalise config access – supports both ConfigParser and Settings
        if hasattr(config, "CHATGPT_API_KEY"):
            # ---- Pydantic Settings ----
            api_key = config.CHATGPT_API_KEY
            base_url = config.CHATGPT_BASE_URL
            model = config.CHATGPT_MODEL
            api_ver = config.CHATGPT_API_VER
        else:
            # ---- legacy ConfigParser ----
            sec = config["CHATGPT"] if hasattr(config, "__getitem__") else config
            api_key = sec["API_KEY"]
            base_url = sec["BASE_URL"]
            model = sec["MODEL"]
            api_ver = sec["API_VER"]

        self.url = f"{base_url}/deployments/{model}/chat/completions?api-version={api_ver}"
        self.headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
            "api-key": api_key,
        }

        # System prompts
        self.system_message = (
            "You are a multifunctional assistant for HKBU students. "
            "Your capabilities include:\n"
            "1. Course & Assignment Helper: When the user mentions a course code (e.g., COMP7940), "
            "you will receive database information (course time, location, assignment deadlines, "
            "requirements). Always prioritise that data. If no database info is available, answer "
            "generally without inventing specific course details.\n"
            "2. Image-to-Video Generation: The bot supports converting an image into an animated "
            "video. Guide the user to use the /video command: send /video, then upload an image, "
            "then provide an animation prompt (or type 'default' for smooth natural animation). "
            "The video will be generated in the background and sent when ready.\n"
            "3. Document Analysis: The bot can quickly extract text from PDFs (direct text "
            "extraction, no OCR, fast) and provide concise summaries. For images with embedded "
            "text, basic extraction is supported. This feature is ready and efficient.\n"
            "Keep responses clear, concise, and student-friendly. If a user asks something outside "
            "these capabilities, answer politely or suggest using the available features."
        )
        self.image_analysis_message = (
            "You are an expert at analysing images and suggesting creative video animation prompts. "
            "Describe what you see in the image briefly, then suggest 3 specific video animation "
            "prompts that would work well with this image. Focus on camera movements, transitions, "
            "and effects."
        )

        # HTTP clients with timeouts and connection limits
        limits = httpx.Limits(
            max_connections=200,
            max_keepalive_connections=50,
        )
        timeout = httpx.Timeout(60.0, connect=10.0)  # overall 60s, connect 10s
        self.client: httpx.AsyncClient | None = httpx.AsyncClient(
            timeout=timeout, limits=limits
        )
        self.sync_client: httpx.Client | None = httpx.Client(
            timeout=timeout, limits=limits
        )

    # ---- circuit breaker helpers ---------------------------------------

    @classmethod
    def _check_circuit_breaker(cls) -> bool:
        """
        Return True if the call is allowed through.

        When the circuit is OPEN, check if enough time has passed to try
        again (HALF_OPEN).  The first HALF_OPEN call that fails trips back
        to OPEN; a success resets to CLOSED.
        """
        now = __import__("time").time()
        if cls._circuit_state == "OPEN":
            if now - cls._last_failure_time >= cls._circuit_reset_after:
                cls._circuit_state = "HALF_OPEN"
                logger.warning("LLM circuit breaker → HALF_OPEN (allowing trial call)")
                return True
            logger.warning("LLM circuit breaker OPEN – blocking call")
            return False
        return True

    @classmethod
    def _record_success(cls):
        """Reset the circuit breaker on success."""
        cls._circuit_state = "CLOSED"
        cls._circuit_failures = 0

    @classmethod
    def _record_failure(cls):
        """Trip or increment the circuit breaker on failure."""
        cls._circuit_failures += 1
        cls._last_failure_time = __import__("time").time()
        if cls._circuit_failures >= cls._circuit_threshold:
            cls._circuit_state = "OPEN"
            logger.error(
                "LLM circuit breaker → OPEN (%d consecutive failures, cooling for %.0fs)",
                cls._circuit_failures,
                cls._circuit_reset_after,
            )

    # ---- internal HTTP client helpers --------------------------------

    def _get_client(self):
        if self.client is None:
            limits = httpx.Limits(max_connections=200, max_keepalive_connections=50)
            timeout = httpx.Timeout(60.0, connect=10.0)
            self.client = httpx.AsyncClient(timeout=timeout, limits=limits)
        return self.client

    def _get_sync_client(self):
        if self.sync_client is None:
            limits = httpx.Limits(max_connections=200, max_keepalive_connections=50)
            timeout = httpx.Timeout(60.0, connect=10.0)
            self.sync_client = httpx.Client(timeout=timeout, limits=limits)
        return self.sync_client

    async def close(self):
        if self.client:
            await self.client.aclose()
            self.client = None
        if self.sync_client:
            self.sync_client.close()
            self.sync_client = None

    # ---- retry helper --------------------------------------------------

    async def _post_with_retry(
        self, client, payload: dict, max_retries: int = 3
    ) -> str:
        """
        POST with exponential backoff retry.

        Retries on:
        - HTTP 429 (rate limited)
        - HTTP 5xx (server errors)
        - Network / timeout errors

        Does NOT retry on HTTP 4xx (client errors other than 429) or 200
        (success but unexpected response shape).
        """
        import asyncio

        last_exc = None

        for attempt in range(1, max_retries + 1):
            # Circuit breaker check
            if not self._check_circuit_breaker():
                return (
                    "Service temporarily unavailable. The AI backend is"
                    " currently overloaded. Please try again in a moment."
                )

            try:
                resp = await client.post(
                    self.url, json=payload, headers=self.headers
                )
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                logger.warning(
                    "LLM connection failed (attempt %d/%d): %s",
                    attempt, max_retries, exc,
                )
                last_exc = exc
                self._record_failure()
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)  # 2s, 4s, 8s
                continue
            except httpx.TimeoutException as exc:
                logger.warning(
                    "LLM timed out (attempt %d/%d): %s",
                    attempt, max_retries, exc,
                )
                last_exc = exc
                self._record_failure()
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)
                continue
            except httpx.HTTPError as exc:
                logger.error("LLM HTTP error: %s", exc)
                self._record_failure()
                return f"Error communicating with AI service: {exc}"

            # Status-code handling
            if resp.status_code == 200:
                try:
                    text = resp.json()["choices"][0]["message"]["content"]
                    self._record_success()
                    return text
                except (KeyError, IndexError, ValueError) as exc:
                    logger.error("Unexpected LLM response format: %s", exc)
                    self._record_failure()
                    return f"Error: unexpected AI response format"

            if resp.status_code == 429:
                # Rate limited – always retry with longer backoff
                logger.warning(
                    "LLM rate limited (429) – attempt %d/%d",
                    attempt, max_retries,
                )
                last_exc = Exception(f"HTTP 429: {resp.text}")
                self._record_failure()
                if attempt < max_retries:
                    # Use Retry-After header if present
                    retry_after = resp.headers.get("Retry-After")
                    wait = float(retry_after) if retry_after else (2 ** attempt)
                    await asyncio.sleep(min(wait, 30))
                continue

            if 500 <= resp.status_code < 600:
                logger.warning(
                    "LLM server error %d (attempt %d/%d)",
                    resp.status_code, attempt, max_retries,
                )
                last_exc = Exception(f"HTTP {resp.status_code}: {resp.text}")
                self._record_failure()
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)
                continue

            # Other 4xx – don't retry
            self._record_failure()
            return f"Error: {resp.text}"

        # All retries exhausted
        logger.error("LLM call failed after %d retries", max_retries)
        return (
            "Sorry, the AI service is currently unavailable. "
            "Please try again later."
        )

    # ---- async text completion ---------------------------------------

    async def submit(self, user_message: str) -> str:
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": user_message},
        ]
        payload = {
            "messages": messages,
            "temperature": 1,
            "max_tokens": 150,
            "top_p": 1,
            "stream": False,
        }
        client = self._get_client()
        return await self._post_with_retry(client, payload)

    # ---- sync text completion (for Celery workers) -------------------

    def _post_sync_with_retry(self, client, payload: dict, max_retries: int = 3) -> str:
        """Synchronous version of _post_with_retry."""
        import time

        last_exc = None

        for attempt in range(1, max_retries + 1):
            if not self._check_circuit_breaker():
                return "Service temporarily unavailable. Please try again."

            try:
                resp = client.post(self.url, json=payload, headers=self.headers)
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                logger.warning("LLM sync connect failed (attempt %d/%d): %s", attempt, max_retries, exc)
                last_exc = exc
                self._record_failure()
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                continue
            except httpx.TimeoutException as exc:
                logger.warning("LLM sync timeout (attempt %d/%d): %s", attempt, max_retries, exc)
                last_exc = exc
                self._record_failure()
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                continue
            except httpx.HTTPError as exc:
                logger.error("LLM sync HTTP error: %s", exc)
                self._record_failure()
                return f"Error communicating with AI service: {exc}"

            if resp.status_code == 200:
                try:
                    text = resp.json()["choices"][0]["message"]["content"]
                    self._record_success()
                    return text
                except (KeyError, IndexError, ValueError) as exc:
                    logger.error("Unexpected LLM sync response format: %s", exc)
                    self._record_failure()
                    return "Error: unexpected AI response format"

            if resp.status_code == 429:
                logger.warning("LLM sync rate limited (429) – attempt %d/%d", attempt, max_retries)
                self._record_failure()
                if attempt < max_retries:
                    retry_after = resp.headers.get("Retry-After")
                    wait = float(retry_after) if retry_after else (2 ** attempt)
                    time.sleep(min(wait, 30))
                continue

            if 500 <= resp.status_code < 600:
                logger.warning("LLM sync server error %d (attempt %d/%d)", resp.status_code, attempt, max_retries)
                self._record_failure()
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                continue

            self._record_failure()
            return f"Error: {resp.text}"

        logger.error("LLM sync call failed after %d retries", max_retries)
        return "Sorry, the AI service is currently unavailable. Please try again later."

    def submit_sync(self, user_message: str, max_tokens: int = 150) -> str:
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": user_message},
        ]
        payload = {
            "messages": messages,
            "temperature": 1,
            "max_tokens": max_tokens,
            "top_p": 1,
            "stream": False,
        }
        client = self._get_sync_client()
        return self._post_sync_with_retry(client, payload)

    # ---- async image + text completion --------------------------------

    async def submit_with_image(
        self, user_message: str, image_base64: str, use_image_analysis_prompt: bool = False
    ) -> str:
        system_msg = self.image_analysis_message if use_image_analysis_prompt else self.system_message
        messages = [
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message},
                    {"type": "image_url", "image_url": {"url": image_base64, "detail": "low"}},
                ],
            },
        ]
        payload = {
            "messages": messages,
            "temperature": 1,
            "max_tokens": 300,
            "top_p": 1,
            "stream": False,
        }
        client = self._get_client()
        return await self._post_with_retry(client, payload)

    # ---- sync image + text completion (for Celery workers) ------------

    def submit_with_image_sync(
        self, user_message: str, image_base64: str, use_image_analysis_prompt: bool = False
    ) -> str:
        system_msg = self.image_analysis_message if use_image_analysis_prompt else self.system_message
        messages = [
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message},
                    {"type": "image_url", "image_url": {"url": image_base64, "detail": "low"}},
                ],
            },
        ]
        payload = {
            "messages": messages,
            "temperature": 1,
            "max_tokens": 300,
            "top_p": 1,
            "stream": False,
        }
        client = self._get_sync_client()
        return self._post_sync_with_retry(client, payload)


if __name__ == "__main__":
    # Standalone test
    import asyncio

    async def main():
        config = configparser.ConfigParser()
        config.read("config.ini")
        chat = ChatGPT(config)
        print("ChatGPT HKBU Client - Test Mode (Async)\n")
        try:
            while True:
                inp = input("Query: ").strip()
                if inp.lower() == "exit":
                    break
                if inp:
                    print(f"Response: {await chat.submit(inp)}\n")
        finally:
            await chat.close()

    asyncio.run(main())
