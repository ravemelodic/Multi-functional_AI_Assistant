"""
Ingest PDF / plain-text documents into Milvus for RAG retrieval.

Unlike course data (which is structured CSV/JSON), this handles
unstructured documents such as lecture notes, syllabi, and reference
materials uploaded by the user or the bot admin.

Usage
-----
    from rag.ingest_documents import ingest_text, ingest_pdf

    # From extracted text
    await ingest_text("... lecture content ...", {"source": "lecture_note"})

    # From a PDF file
    await ingest_pdf("/path/to/file.pdf", {"course_code": "COMP7940"})
"""

import logging
import os
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def get_text_splitter(chunk_size: int = 500, chunk_overlap: int = 50):
    """Default text splitter for document ingestion."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


async def ingest_text(
    text: str,
    metadata: Optional[dict] = None,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> int:
    """
    Ingest plain text into Milvus for RAG retrieval.

    Parameters
    ----------
    text : str
        The text content to ingest.
    metadata : dict, optional
        Metadata dict, e.g. ``{"source": "user_upload", "filename": "notes.pdf"}``.
    chunk_size : int
        Target chunk size in characters.
    chunk_overlap : int
        Overlap between consecutive chunks.

    Returns
    -------
    int
        Number of chunks stored.
    """
    from langchain_milvus import Milvus
    from rag.retriever import get_embeddings

    from configs.settings import settings

    if metadata is None:
        metadata = {}
    metadata.setdefault("source", "user_upload")

    splitter = get_text_splitter(chunk_size, chunk_overlap)
    doc = Document(page_content=text, metadata=metadata)
    chunks = splitter.split_documents([doc])

    embeddings = get_embeddings()

    if settings.MILVUS_URI:
        connection_args: dict = {"uri": settings.MILVUS_URI}
        if settings.MILVUS_TOKEN:
            connection_args["token"] = settings.MILVUS_TOKEN
    else:
        connection_args = {"host": settings.MILVUS_HOST, "port": settings.MILVUS_PORT}

    vector_store = Milvus.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=settings.MILVUS_COLLECTION,
        connection_args=connection_args,
        auto_id=True,
    )
    _ = vector_store.similarity_search("test", k=1)

    logger.info("Ingested %d chunks from '%s'.", len(chunks), metadata.get("source", "?"))
    return len(chunks)


async def ingest_pdf(
    pdf_path: str,
    metadata: Optional[dict] = None,
) -> int:
    """
    Extract text from a PDF file and ingest into Milvus.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file on disk.
    metadata : dict, optional
        Extra metadata (filename is auto-detected).

    Returns
    -------
    int
        Number of chunks stored.
    """
    try:
        import pymupdf  # PyMuPDF (modern name)
    except ImportError:
        import fitz as pymupdf  # fallback

    doc = pymupdf.open(pdf_path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()

    if metadata is None:
        metadata = {}
    metadata.setdefault("source", "pdf_upload")
    metadata.setdefault("filename", os.path.basename(pdf_path))

    return await ingest_text(text, metadata)
