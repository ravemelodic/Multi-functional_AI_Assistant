"""
Milvus vector-store retriever using LangChain integration.

Provides:
- ``get_retriever()``        —  cached ``VectorStoreRetriever`` (top-k)
- ``retrieve_with_scores()`` —  cached vector store + relevance scores
                                 with configurable threshold filtering.

The collection stores chunked course documents (syllabi, descriptions,
assignment briefs) and retrieves them via cosine similarity on
OpenAI embeddings.

Prerequisites (set in config.ini or environment)
-------------------------------------------------
- MILVUS_HOST, MILVUS_PORT          # local Milvus (Docker)
- or MILVUS_URI, MILVUS_TOKEN       # Zilliz Cloud
- MILVUS_COLLECTION                 # collection name
- EMBEDDING_API_KEY, EMBEDDING_BASE_URL, EMBEDDING_MODEL
"""

import asyncio
import logging
import time
from functools import lru_cache
from typing import List, Tuple, Optional

from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from app.configs.settings import settings

logger = logging.getLogger(__name__)

# Conversation memory collection (stores user-bot chat history for persistent memory)
MEMORY_COLLECTION = "conversation_memory"

# ------------------------------------------------------------------ #
#  Hybrid search globals (BM25 sparse index cache)                    #
# ------------------------------------------------------------------ #

_HYBRID_CACHE = {
    "bm25": None,          # BM25Okapi instance
    "bm25_docs": None,     # list[Document] for BM25
    "bm25_ts": 0,          # last build timestamp
}
_BM25_CACHE_TTL = 600  # rebuild BM25 index every 10 min


# ------------------------------------------------------------------ #
#  TTL-based cache helper (avoids stale connections after restarts)   #
# ------------------------------------------------------------------ #

_VECTOR_STORE_CACHE = {"ts": 0, "instance": None, "memory_instance": None}
_CACHE_TTL = 300  # seconds – re-create connection after 5 min


def _get_cached_vector_store(store_type: str = "course") -> Milvus:
    """
    Get or create a vector store with TTL-based caching.

    If the cached instance is older than ``_CACHE_TTL`` seconds, or if
    the underlying Milvus connection fails during use, the caller should
    explicitly invalidate with ``_invalidate_vector_store_cache()``.
    """
    global _VECTOR_STORE_CACHE
    key = "instance" if store_type == "course" else "memory_instance"
    now = time.time()

    cached = _VECTOR_STORE_CACHE.get(key)
    ts = _VECTOR_STORE_CACHE.get("ts", 0)

    if cached is not None and (now - ts) < _CACHE_TTL:
        return cached

    # Cache miss or expired → create fresh
    collection = settings.MILVUS_COLLECTION if store_type == "course" else MEMORY_COLLECTION
    embeddings = get_embeddings()
    connection_args = _build_connection_args()

    try:
        instance = Milvus(
            embedding_function=embeddings,
            collection_name=collection,
            connection_args=connection_args,
            auto_id=True,
        )
    except Exception as exc:
        logger.error(
            "Failed to connect to Milvus collection '%s': %s",
            collection, exc,
        )
        # On failure, keep the old cached instance if it exists (fallback)
        if cached is not None:
            logger.warning("Falling back to existing Milvus connection (stale may still work)")
            return cached
        raise

    _VECTOR_STORE_CACHE[key] = instance
    _VECTOR_STORE_CACHE["ts"] = now
    logger.info("Created fresh Milvus connection for '%s' (TTL=%ds)", collection, _CACHE_TTL)
    return instance


def _invalidate_cache(store_type: str = None):
    """
    Invalidate cached vector store(s).

    Call when a Milvus operation fails so the next request reconnects.
    """
    global _VECTOR_STORE_CACHE
    if store_type == "course":
        _VECTOR_STORE_CACHE["instance"] = None
    elif store_type == "memory":
        _VECTOR_STORE_CACHE["memory_instance"] = None
    else:
        _VECTOR_STORE_CACHE["instance"] = None
        _VECTOR_STORE_CACHE["memory_instance"] = None
    logger.info("Invalidated Milvus vector store cache (%s)", store_type or "all")


# ------------------------------------------------------------------ #
#  Internal helpers                                                   #
# ------------------------------------------------------------------ #

def get_embeddings():
    """
    Create an embedding model instance from settings.

    Supports two providers:
    - ``"openai"`` (default) — uses ``OpenAIEmbeddings``, calls ``{BASE_URL}/embeddings``
    - ``"azure"`` — uses ``AzureOpenAIEmbeddings``, calls Azure-format endpoint

    The API key falls back to ``CHATGPT_API_KEY`` if ``EMBEDDING_API_KEY`` is not set.
    """
    if not settings.EMBEDDING_API_KEY:
        api_key = settings.WAN_AI_API_KEY or settings.CHATGPT_API_KEY
        base_url = settings.EMBEDDING_BASE_URL or settings.WAN_AI_BASE_URL or settings.CHATGPT_BASE_URL
    else:
        api_key = settings.EMBEDDING_API_KEY
        base_url = settings.EMBEDDING_BASE_URL

    if settings.EMBEDDING_PROVIDER == "azure":
        logger.info("Using AzureOpenAIEmbeddings (model=%s, endpoint=%s)", settings.EMBEDDING_MODEL, base_url)
        return AzureOpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            azure_endpoint=base_url,
            api_key=api_key,
            api_version=settings.EMBEDDING_API_VER,
        )

    logger.info("Using OpenAIEmbeddings (model=%s, base=%s)", settings.EMBEDDING_MODEL, base_url)
    return OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        openai_api_key=api_key,
        openai_api_base=base_url,
    )


# Keep backward compatibility — _build_embeddings is used internally
_build_embeddings = get_embeddings


def _build_connection_args() -> dict:
    """
    Build Milvus connection parameters.

    Prefer MILVUS_URI (Zilliz Cloud) if set, otherwise host+port (local).
    """
    if settings.MILVUS_URI:
        args: dict = {"uri": settings.MILVUS_URI}
        if settings.MILVUS_TOKEN:
            args["token"] = settings.MILVUS_TOKEN
        return args

    # Local Milvus (Docker)
    return {
        "host": settings.MILVUS_HOST,
        "port": settings.MILVUS_PORT,
    }


# (_get_vector_store replaced by TTL-cached _get_cached_vector_store above)


# ------------------------------------------------------------------ #
#  Public API                                                         #
# ------------------------------------------------------------------ #

@lru_cache(maxsize=1)
def get_retriever() -> VectorStoreRetriever:
    """
    Return a cached Milvus-backed retriever (singleton, top-3).
    """
    vector_store = _get_cached_vector_store("course")
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )
    logger.info("Milvus retriever ready (k=3).")
    return retriever


async def retrieve_with_scores(
    query: str,
    k: int = 5,
    score_threshold: float = 0.5,
) -> List[Tuple[Document, float]]:
    """
    Retrieve documents with relevance scores, filtered by threshold.

    Uses ``similarity_search_with_relevance_scores`` which normalises the
    distance metric into a ``[0, 1]`` relevance score (higher = more
    relevant).  Results below ``score_threshold`` are discarded.

    Parameters
    ----------
    query : str
        Search query text.
    k : int
        Number of raw candidates to fetch from Milvus (default 5).
    score_threshold : float
        Minimum relevance score to keep a result (default 0.5).

    Returns
    -------
    list of (Document, float)
        Matching documents with their relevance scores, sorted by score
        descending.
    """
    vector_store = _get_cached_vector_store("course")

    try:
        raw: List[Tuple[Document, float]] = await asyncio.to_thread(
            lambda: vector_store.similarity_search_with_relevance_scores(query, k=k),
        )
    except Exception as exc:
        logger.warning("Milvus search failed, invalidating cache: %s", exc)
        _invalidate_cache("course")
        return []

    filtered = [(doc, score) for doc, score in raw if score >= score_threshold]
    filtered.sort(key=lambda x: x[1], reverse=True)

    if filtered:
        logger.info(
            "retrieve_with_scores(%s…) – %d / %d passed threshold=%.2f",
            query[:40], len(filtered), len(raw), score_threshold,
        )
    else:
        logger.info(
            "retrieve_with_scores(%s…) – all %d results below threshold=%.2f",
            query[:40], len(raw), score_threshold,
        )

    return filtered


# =================================================================== #
#  Hybrid Search – BM25 sparse + Dense vector + RRF fusion             #
# =================================================================== #


def _build_bm25_index() -> tuple:
    """
    Build a BM25 sparse index from all documents in the Milvus collection.

    Uses pymilvus to dump stored text, then builds a ``BM25Okapi`` index
    cached in ``_HYBRID_CACHE`` (refreshed every ``_BM25_CACHE_TTL`` s).

    Returns
    -------
    (bm25_index, documents)
        ``bm25_index`` is a ``BM25Okapi`` instance (or None on failure).
        ``documents`` is a list of ``Document`` objects in the same order
        as the tokenised corpus used by the index.
    """
    global _HYBRID_CACHE
    now = time.time()

    # Return cached index if still fresh
    if _HYBRID_CACHE["bm25"] is not None and (now - _HYBRID_CACHE["bm25_ts"]) < _BM25_CACHE_TTL:
        return _HYBRID_CACHE["bm25"], _HYBRID_CACHE["bm25_docs"]

    try:
        from pymilvus import connections, Collection as MilvusCollection, utility, DataType
        from rank_bm25 import BM25Okapi
    except ImportError as exc:
        logger.warning("BM25/sparse dependencies not installed: %s", exc)
        return None, []

    try:
        conn_args = _build_connection_args()
        connections.connect(**conn_args)

        if not utility.has_collection(settings.MILVUS_COLLECTION):
            logger.warning("Collection '%s' not found – BM25 index unavailable", settings.MILVUS_COLLECTION)
            return None, []

        collection = MilvusCollection(settings.MILVUS_COLLECTION)
        collection.load()

        # Auto-detect the text field name from the schema
        text_field = "text"
        for field in collection.schema.fields:
            if field.dtype == DataType.VARCHAR and "text" in field.name.lower():
                text_field = field.name
                break

        # Query all stored documents
        results = collection.query(expr="", output_fields=["*"], limit=10000)

        docs: list[Document] = []
        raw_texts: list[str] = []
        for r in results:
            text = r.get(text_field, "")
            if text:
                meta = {k: v for k, v in r.items() if k not in (text_field, "pk", "vector")}
                docs.append(Document(page_content=text, metadata=meta))
                raw_texts.append(text)

        if not docs:
            logger.info("No documents found for BM25 index")
            return None, []

        # Build BM25Okapi with simple tokenisation
        tokenized_corpus = [t.lower().split() for t in raw_texts]
        bm25 = BM25Okapi(tokenized_corpus)

        # Update cache
        _HYBRID_CACHE["bm25"] = bm25
        _HYBRID_CACHE["bm25_docs"] = docs
        _HYBRID_CACHE["bm25_ts"] = now

        logger.info("BM25 sparse index built – %d documents, %d unique terms", len(docs), len(bm25.idf))
        return bm25, docs

    except Exception as exc:
        logger.warning("Failed to build BM25 index: %s", exc)
        return None, []


async def retrieve_hybrid_with_scores(
    query: str,
    k: int = 5,
    score_threshold: float = 0.5,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
    expr: str | None = None,
) -> List[Tuple[Document, float]]:
    """
    Hybrid retrieval combining BM25 sparse scores + dense vector cosine similarity.

    Fuses results with **weighted RRF** (Reciprocal Rank Fusion):

        score(d) = sparse_weight × 1/(k_rrf + rank_sparse(d))
                 + dense_weight  × 1/(k_rrf + rank_dense(d))

    Parameters
    ----------
    query : str
        User query text.
    k : int
        Number of final results to return.
    score_threshold : float
        Minimum normalised RRF score to retain a result.
    dense_weight, sparse_weight : float
        Relative importance of each system.  Must sum to 1.0.

    Returns
    -------
    list of (Document, float)
        Fused results sorted by RRF score descending.
        Falls back to ``retrieve_with_scores()`` (pure dense) if BM25 is unavailable.
    """
    K_RRF = 60  # RRF constant

    # ---- 1. Dense vector search -----------------------------------------
    vector_store = _get_cached_vector_store("course")
    try:
        raw_dense: List[Tuple[Document, float]] = await asyncio.to_thread(
            lambda: vector_store.similarity_search_with_relevance_scores(query, k=k * 3, expr=expr) if expr else vector_store.similarity_search_with_relevance_scores(query, k=k * 3),
        )
    except Exception as exc:
        logger.warning("Dense search failed in hybrid mode: %s", exc)
        _invalidate_cache("course")
        raw_dense = []

    # ---- 2. BM25 sparse search -------------------------------------------
    bm25, bm25_docs = _build_bm25_index()

    # Fall back to pure dense if BM25 is unavailable
    if bm25 is None or not bm25_docs:
        if raw_dense:
            return [(doc, score) for doc, score in raw_dense if score >= score_threshold][:k]
        return await retrieve_with_scores(query, k=k, score_threshold=score_threshold)

    tokenized_query = query.lower().split()
    sparse_scores = bm25.get_scores(tokenized_query)

    # Rank BM25 results (descending by BM25 score)
    sparse_ranks = sorted(
        range(len(sparse_scores)),
        key=lambda i: sparse_scores[i],
        reverse=True,
    )
    # Keep only documents with positive BM25 scores
    sparse_ranks = [i for i in sparse_ranks if sparse_scores[i] > 0][:k * 3]

    # ---- 3. RRF fusion ---------------------------------------------------
    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    # Dense contributions
    for rank, (doc, _) in enumerate(raw_dense):
        rrf_scores[doc.page_content] = dense_weight * (1.0 / (K_RRF + rank + 1))
        doc_map[doc.page_content] = doc

    # Sparse contributions
    for rank, idx in enumerate(sparse_ranks):
        doc = bm25_docs[idx]
        contrib = sparse_weight * (1.0 / (K_RRF + rank + 1))
        if doc.page_content in rrf_scores:
            rrf_scores[doc.page_content] += contrib
        else:
            rrf_scores[doc.page_content] = contrib
            doc_map[doc.page_content] = doc

    # Sort by RRF score descending
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    if not sorted_items:
        return []

    # Normalise scores to [0, 1] and apply threshold
    max_score = sorted_items[0][1]
    filtered = []
    for content, rrf in sorted_items:
        norm_score = rrf / max_score
        if norm_score >= score_threshold:
            filtered.append((doc_map[content], round(norm_score, 4)))

    result = filtered[:k]

    logger.info(
        "Hybrid retrieve(%.40s…) – sparse=%d dense=%d → fused=%d (threshold=%.2f)",
        query, len(sparse_ranks), len(raw_dense), len(result), score_threshold,
    )
    return result


# =================================================================== #
#  Conversation Memory (persistent chat history in Milvus)             #
# =================================================================== #


async def store_conversation_memory(
    user_id: int,
    user_message: str,
    bot_response: str,
) -> bool:
    """
    Store one conversation turn in the memory vector collection.

    Parameters
    ----------
    user_id : int
        Telegram user id (used as metadata for per-user filtering).
    user_message : str
        The user's original message.
    bot_response : str
        The bot's reply.

    Returns
    -------
    bool
        True on success, False on failure (logged, non-fatal).
    """
    try:
        vector_store = _get_cached_vector_store("memory")
    except Exception as exc:
        logger.warning("Cannot store conversation memory (Milvus unavailable): %s", exc)
        return False

    try:
        text = f"User: {user_message}\nBot: {bot_response}"
        doc = Document(
            page_content=text,
            metadata={
                "user_id": user_id,
                "source": "conversation_memory",
                "type": "chat",
            },
        )
        # add_documents auto-creates the collection if it does not exist
        await asyncio.to_thread(vector_store.add_documents, [doc])
        logger.debug("Stored conversation memory for user %d (length=%d)", user_id, len(text))
        return True
    except Exception as exc:
        logger.warning("Failed to store conversation memory, invalidating cache: %s", exc)
        _invalidate_cache("memory")
        return False


async def retrieve_conversation_memory(
    query: str,
    user_id: int,
    k: int = 5,
    score_threshold: float = 0.45,
) -> List[Tuple[Document, float]]:
    """
    Retrieve semantically relevant past conversations from the memory collection.

    Parameters
    ----------
    query : str
        The current user message (used for similarity search).
    user_id : int
        Only return memories for this user.
    k : int
        Number of raw candidates to fetch (default 5).
    score_threshold : float
        Minimum relevance score to keep a result (default 0.45).

    Returns
    -------
    list of (Document, float)
        Matching memory documents (each contains one past conversation turn).
        Empty list if the collection is unreachable or no matches found.
    """
    try:
        vector_store = _get_cached_vector_store("memory")
    except Exception as exc:
        logger.warning("Cannot retrieve conversation memory (Milvus unavailable): %s", exc)
        return []

    # Pre-filter via metadata — LangChain Milvus supports ``expr`` in
    # similarity_search.  We use a simple user_id expression so Milvus
    # only compares against the requesting user's memories.
    expr = f'user_id == {user_id}'

    try:
        raw: List[Tuple[Document, float]] = await asyncio.to_thread(
            lambda: vector_store.similarity_search_with_relevance_scores(
                query, k=k, expr=expr,
            ),
        )
    except Exception as exc:
        logger.warning("Memory retrieval query failed, invalidating cache: %s", exc)
        _invalidate_cache("memory")
        return []

    filtered = [(doc, score) for doc, score in raw if score >= score_threshold]
    filtered.sort(key=lambda x: x[1], reverse=True)

    if filtered:
        logger.info(
            "Memory retrieve(user=%d, query=%.40s) – %d / %d passed threshold=%.2f",
            user_id, query, len(filtered), len(raw), score_threshold,
        )
    else:
        logger.debug(
            "Memory retrieve(user=%d) – all %d results below threshold=%.2f",
            user_id, len(raw), score_threshold,
        )

    return filtered
