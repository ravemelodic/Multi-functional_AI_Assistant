"""
Parse course/assignment data from CSV/JSON files into LangChain Documents
for Milvus ingestion.

Used by both the FastAPI admin interface and the Telegram bot's
/upload_courses command.

Examples
--------
    from rag.parse_courses import parse_csv_content, store_to_milvus

    with open("courses.csv") as f:
        docs = parse_csv_content(f.read(), source_type="courses")
    count = await store_to_milvus(docs)
    print(f"Stored {count} chunks to Milvus")
"""

import csv
import io
import json
import logging
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Parsers                                                            #
# ------------------------------------------------------------------ #

def parse_csv_content(content: str, source_type: str = "courses") -> List[Document]:
    """
    Parse CSV string content into LangChain Documents.

    Parameters
    ----------
    content : str
        Raw CSV text (with header row).
    source_type : str
        ``"courses"`` or ``"assignments"``.

    Returns
    -------
    List[Document]
        One Document per row, with metadata.
    """
    docs: List[Document] = []
    reader = csv.DictReader(io.StringIO(content))

    builder = _build_course_doc if source_type == "courses" else _build_assignment_doc

    for row in reader:
        doc = builder(row)
        if doc is not None:
            docs.append(doc)

    logger.info("Parsed %d %s from CSV.", len(docs), source_type)
    return docs


def parse_json_content(content: str, source_type: str = "courses") -> List[Document]:
    """
    Parse JSON string content into LangChain Documents.

    Accepts a JSON array of objects, or a single object.
    """
    data = json.loads(content)
    if not isinstance(data, list):
        data = [data]

    docs: List[Document] = []
    builder = _build_course_doc if source_type == "courses" else _build_assignment_doc

    for item in data:
        doc = builder(item)
        if doc is not None:
            docs.append(doc)

    logger.info("Parsed %d %s from JSON.", len(docs), source_type)
    return docs


# ------------------------------------------------------------------ #
#  Document builders                                                  #
# ------------------------------------------------------------------ #

def _build_course_doc(row: dict) -> Document | None:
    code = row.get("course_code", "").strip().upper()
    if not code:
        return None

    text = (
        f"Course Code: {code}\n"
        f"Course Name: {row.get('course_name', 'N/A')}\n"
        f"Class Time: {row.get('class_time', 'TBA')}\n"
        f"Location: {row.get('location', 'TBA')}\n"
        f"Description: {row.get('description', '')}"
    )
    return Document(
        page_content=text,
        metadata={
            "course_code": code,
            "source": "courses",
            "data_type": "course_info",
        },
    )


def _build_assignment_doc(row: dict) -> Document | None:
    code = row.get("course_code", "").strip().upper()
    if not code:
        return None

    text = (
        f"Course Code: {code}\n"
        f"Assignment: {row.get('title', 'N/A')}\n"
        f"Deadline: {row.get('deadline', 'N/A')}\n"
        f"Description: {row.get('description', '')}"
    )
    return Document(
        page_content=text,
        metadata={
            "course_code": code,
            "source": "assignments",
            "data_type": "assignment_info",
        },
    )


# ------------------------------------------------------------------ #
#  Chunking                                                           #
# ------------------------------------------------------------------ #

def chunk_documents(
    docs: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:
    """Split documents into smaller chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )
    return splitter.split_documents(docs)


# ------------------------------------------------------------------ #
#  Milvus storage                                                     #
# ------------------------------------------------------------------ #

async def store_to_milvus(docs: List[Document]) -> int:
    """
    Store parsed Documents to Milvus vector store.

    Returns the number of chunks actually stored.
    """
    from langchain_milvus import Milvus
    from rag.retriever import get_embeddings

    from configs.settings import settings

    if not docs:
        return 0

    embeddings = get_embeddings()

    if settings.MILVUS_URI:
        connection_args: dict = {"uri": settings.MILVUS_URI}
        if settings.MILVUS_TOKEN:
            connection_args["token"] = settings.MILVUS_TOKEN
    else:
        connection_args = {"host": settings.MILVUS_HOST, "port": settings.MILVUS_PORT}

    chunked = chunk_documents(docs)

    vector_store = Milvus.from_documents(
        documents=chunked,
        embedding=embeddings,
        collection_name=settings.MILVUS_COLLECTION,
        connection_args=connection_args,
        auto_id=True,
    )
    # Trigger a dummy search to confirm persistence
    _ = vector_store.similarity_search("test", k=1)

    logger.info("Stored %d chunks to Milvus collection '%s'.", len(chunked), settings.MILVUS_COLLECTION)
    return len(chunked)
