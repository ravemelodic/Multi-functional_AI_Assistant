"""
FastAPI course-data management server for the Telegram chatbot RAG system.

Provides a web admin interface and REST API for uploading course /
assignment data directly into the Milvus vector database.  Data uploaded
here is immediately available to the Telegram bot's RAG retriever.

Endpoints
---------
GET  /admin          —  Web admin panel (drag-and-drop upload)
POST /api/upload     —  Upload CSV / JSON file
POST /api/ingest     —  Ingest raw JSON body
GET  /api/health     —  Health check
GET  /api/stats      —  Quick stats from Milvus

Usage
-----
    # Production
    uvicorn app.api:app --host 0.0.0.0 --port 8000

    # Development (hot-reload)
    python -m app.api
"""

import json as json_mod
import logging
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is on sys.path
HERE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(HERE))

from app.configs.settings import settings
from app.rag.parse_courses import parse_csv_content, parse_json_content, store_to_milvus

# ------------------------------------------------------------------ #
#  Logging                                                            #
# ------------------------------------------------------------------ #
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("app.api")

# ------------------------------------------------------------------ #
#  FastAPI app                                                        #
# ------------------------------------------------------------------ #
app = FastAPI(
    title="Course Data Manager — Chatbot RAG",
    description="Upload course / assignment data into Milvus for the Telegram chatbot.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =================================================================== #
#  Admin web UI                                                       #
# =================================================================== #

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse('<html><head><meta http-equiv="refresh" content="0; url=/admin"></head></html>')


@app.get("/admin", response_class=HTMLResponse)
async def admin_page():
    """Serve the admin web interface (single-page HTML)."""
    html_path = HERE / "app" / "api_templates" / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Admin page not found</h1><p>api_templates/index.html is missing.</p>")


# =================================================================== #
#  REST API                                                           #
# =================================================================== #

@app.get("/api/health")
async def health():
    """Lightweight health check."""
    return {
        "status": "ok",
        "milvus_host": settings.MILVUS_HOST,
        "milvus_port": settings.MILVUS_PORT,
        "collection": settings.MILVUS_COLLECTION,
    }


@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    source_type: str = Form("courses"),
):
    """
    Upload a CSV or JSON file containing course or assignment data.

    - ``source_type``: ``"courses"`` (default) or ``"assignments"``
    - Supports ``.csv`` and ``.json`` extensions.
    """
    if source_type not in ("courses", "assignments"):
        raise HTTPException(400, "source_type must be 'courses' or 'assignments'")

    content = (await file.read()).decode("utf-8")
    filename = file.filename or "upload"

    try:
        if filename.lower().endswith(".csv"):
            docs = parse_csv_content(content, source_type=source_type)
        elif filename.lower().endswith(".json"):
            docs = parse_json_content(content, source_type=source_type)
        else:
            # Auto-detect: JSON arrays start with [, CSV has commas + headers
            stripped = content.strip()
            if stripped.startswith("["):
                docs = parse_json_content(content, source_type=source_type)
            else:
                docs = parse_csv_content(content, source_type=source_type)

        if not docs:
            raise HTTPException(400, "No valid records found in the file. Check column names and data format.")

        stored = await store_to_milvus(docs)

        return {
            "success": True,
            "filename": filename,
            "source_type": source_type,
            "records_parsed": len(docs),
            "chunks_stored": stored,
            "message": f"✅ Imported {len(docs)} {source_type} ({stored} chunks into Milvus).",
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Upload failed")
        raise HTTPException(500, f"Upload failed: {exc}")


@app.post("/api/ingest")
async def ingest_json(body: dict):
    """
    Ingest course / assignment data as a raw JSON array.

    Request body::

        {
            "source_type": "courses",       # or "assignments"
            "data": [
                {
                    "course_code": "COMP7940",
                    "course_name": "AI and Chatbot Development",
                    "class_time": "Monday 14:30-17:15",
                    "location": "DLB 514",
                    "description": "..."
                }
            ]
        }
    """
    source_type = body.get("source_type", "courses")
    if source_type not in ("courses", "assignments"):
        raise HTTPException(400, "source_type must be 'courses' or 'assignments'")

    raw = body.get("data", body)
    json_str = json_mod.dumps(raw) if isinstance(raw, (list, dict)) else str(raw)

    try:
        docs = parse_json_content(json_str, source_type=source_type)
        if not docs:
            raise HTTPException(400, "No valid records found in payload.")

        stored = await store_to_milvus(docs)

        return {
            "success": True,
            "source_type": source_type,
            "records_parsed": len(docs),
            "chunks_stored": stored,
            "message": f"✅ Ingested {len(docs)} {source_type} ({stored} chunks).",
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Ingest failed")
        raise HTTPException(500, f"Ingest failed: {exc}")


@app.get("/api/stats")
async def stats():
    """
    Return a quick summary of what is stored in Milvus.

    Because Milvus does not expose a simple "count all" API without a
    query, this performs a broad similarity search and reports the
    unique course codes found in the top results.
    """
    try:
        from rag.retriever import get_retriever

        retriever = get_retriever()
        docs = await retriever.ainvoke("course")

        seen: dict[str, int] = {}
        for d in docs:
            code = d.metadata.get("course_code", "?")
            src = d.metadata.get("source", "?")
            key = f"{code} ({src})"
            seen[key] = seen.get(key, 0) + 1

        return {
            "total_samples_shown": len(docs),
            "entries": [{"key": k, "count": v} for k, v in sorted(seen.items())],
            "collection": settings.MILVUS_COLLECTION,
        }
    except Exception as exc:
        logger.warning("Stats query failed: %s", exc)
        return {
            "total_samples_shown": 0,
            "entries": [],
            "note": f"Milvus query failed: {exc}",
        }


# =================================================================== #
#  Direct entry point                                                  #
# =================================================================== #
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)
