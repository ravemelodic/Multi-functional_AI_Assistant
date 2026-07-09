"""
Offline ingestion script: reads course / assignment data from CSV files,
chunks it, embeds it, and stores the vectors in the Milvus collection.

Run this once (or on a schedule) to populate the vector store that the
RAG retriever queries at runtime.

Data sources (CSV files in the ``data/`` directory):
- ``data/courses.csv``       — course listings
- ``data/assignments.csv``   — assignment briefs

Usage
-----
    python -m rag.ingest                  # uses config.ini
    python -m rag.ingest --config ../custom.ini

Prerequisites
-------------
- Milvus must be reachable (settings.MILVUS_*)
- Embedding endpoint must be configured (settings.EMBEDDING_*)
- CSV files must exist under ``data/`` (see examples in the repository)
"""

import argparse
import asyncio
import csv
import logging
import sys
from pathlib import Path

from langchain_milvus import Milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Add project root to sys.path so imports work when run as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.settings import settings

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("rag.ingest")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def read_courses_csv(path: Path) -> list[Document]:
    """Read course records from a CSV file and build Document objects."""
    docs: list[Document] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (
                f"Course Code: {row.get('course_code', '')}\n"
                f"Course Name: {row.get('course_name', '')}\n"
                f"Class Time: {row.get('class_time', '')}\n"
                f"Location: {row.get('location', '')}\n"
                f"Description: {row.get('description', '')}"
            )
            docs.append(
                Document(
                    page_content=text,
                    metadata={"course_code": row["course_code"], "source": "courses"},
                )
            )
    logger.info("Read %d course records from %s", len(docs), path.name)
    return docs


def read_assignments_csv(path: Path) -> list[Document]:
    """Read assignment records from a CSV file and build Document objects."""
    docs: list[Document] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (
                f"Course Code: {row.get('course_code', '')}\n"
                f"Assignment: {row.get('title', '')}\n"
                f"Deadline: {row.get('deadline', '')}\n"
                f"Description: {row.get('description', '')}"
            )
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "course_code": row["course_code"],
                        "source": "assignments",
                    },
                )
            )
    logger.info("Read %d assignment records from %s", len(docs), path.name)
    return docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    """Split large documents into smaller chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunked = splitter.split_documents(docs)
    logger.info("Chunked %d documents into %d chunks.", len(docs), len(chunked))
    return chunked


async def ingest(courses_path: Path = None, assignments_path: Path = None):
    """Main ingestion pipeline – reads CSV files → chunks → embeds → Milvus."""
    courses_path = courses_path or DATA_DIR / "courses.csv"
    assignments_path = assignments_path or DATA_DIR / "assignments.csv"

    logger.info("Starting Milvus ingestion from CSV files ...")

    all_docs: list[Document] = []

    if courses_path.exists():
        all_docs.extend(read_courses_csv(courses_path))
    else:
        logger.warning("Courses CSV not found: %s", courses_path)

    if assignments_path.exists():
        all_docs.extend(read_assignments_csv(assignments_path))
    else:
        logger.warning("Assignments CSV not found: %s", assignments_path)

    if not all_docs:
        logger.warning("No documents loaded – nothing to ingest.")
        return

    # Chunk
    chunked = chunk_documents(all_docs)

    # Build embeddings
    from rag.retriever import get_embeddings

    logger.info("Using embedding model: %s", settings.EMBEDDING_MODEL)
    embeddings = get_embeddings()

    # Build Milvus connection args
    if settings.MILVUS_URI:
        connection_args = {"uri": settings.MILVUS_URI}
        if settings.MILVUS_TOKEN:
            connection_args["token"] = settings.MILVUS_TOKEN
    else:
        connection_args = {
            "host": settings.MILVUS_HOST,
            "port": settings.MILVUS_PORT,
        }

    # Store chunks into Milvus
    logger.info(
        "Storing %d chunks into Milvus collection '%s' ...",
        len(chunked),
        settings.MILVUS_COLLECTION,
    )
    vector_store = Milvus.from_documents(
        documents=chunked,
        embedding=embeddings,
        collection_name=settings.MILVUS_COLLECTION,
        connection_args=connection_args,
        auto_id=True,
    )

    _ = vector_store.similarity_search("test", k=1)
    logger.info(
        "Ingestion complete – collection '%s' ready (%d chunks).",
        settings.MILVUS_COLLECTION,
        len(chunked),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest course data into Milvus")
    parser.add_argument(
        "--config",
        default="config.ini",
        help="Path to config.ini (default: ./config.ini)",
    )
    parser.add_argument(
        "--courses",
        default=None,
        help="Path to courses CSV (default: data/courses.csv)",
    )
    parser.add_argument(
        "--assignments",
        default=None,
        help="Path to assignments CSV (default: data/assignments.csv)",
    )
    args = parser.parse_args()

    # Reload settings from the provided config path
    from configs.settings import Settings

    _settings = Settings.from_ini(args.config)
    import configs.settings as mod_settings

    mod_settings.settings = _settings

    courses_path = Path(args.courses) if args.courses else None
    assignments_path = Path(args.assignments) if args.assignments else None

    asyncio.run(ingest(courses_path, assignments_path))
