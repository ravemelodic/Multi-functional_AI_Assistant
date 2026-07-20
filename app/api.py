"""
FastAPI course-data management server for the Telegram chatbot RAG system.

Provides a web admin interface and REST API for uploading course /
assignment data directly into the Milvus vector database.  Data uploaded
here is immediately available to the Telegram bot's RAG retriever.

Authentication
--------------
All endpoints require a valid API token, passed via one of:
- Header:  ``X-API-Key: <token>``
- Query:   ``?token=<token>``
- Cookie:  ``token=<token>``

Tokens are configured via the ``ADMIN_TOKENS_JSON`` environment variable
as a JSON dict mapping token → role ("upload" or "view").

Endpoints
---------
GET  /admin          —  Web admin panel (token-based login)
POST /api/auth/login —  Validate token, returns role
GET  /api/auth/me    —  Return current token info (requires auth)
POST /api/upload     —  Upload CSV / JSON file (role: upload)
POST /api/ingest     —  Ingest raw JSON body (role: upload)
GET  /api/health     —  Health check (no auth required)
GET  /api/stats      —  Quick stats from Milvus (role: view or upload)

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
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Response, Cookie, Query, Header, Depends
from fastapi.responses import HTMLResponse, JSONResponse
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
#  Audit log — records every admin action (who, what, when)           #
# ------------------------------------------------------------------ #
audit_logger = logging.getLogger("app.audit")
audit_logger.setLevel(logging.INFO)
try:
    os.makedirs(settings.LOG_DIR, exist_ok=True)
    audit_handler = logging.FileHandler(f"{settings.LOG_DIR}/audit.log", encoding="utf-8")
    audit_handler.setFormatter(logging.Formatter("%(asctime)s [AUDIT] %(message)s"))
    audit_logger.addHandler(audit_handler)
except Exception as exc:
    logger.warning("Audit log unavailable: %s", exc)


def log_audit(action: str, admin_id: str, detail: str, success: bool, ip: str = ""):
    """Write a structured audit entry."""
    status = "OK" if success else "FAIL"
    ip_part = f" | ip={ip}" if ip else ""
    audit_logger.info("%s | admin=%s | %s%s | status=%s", action, admin_id, detail, ip_part, status)


# ------------------------------------------------------------------ #
#  Upload validation constants                                        #
# ------------------------------------------------------------------ #
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_UPLOAD_EXTENSIONS = {".csv", ".json"}


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
#  Authentication helpers                                              #
# =================================================================== #

def _resolve_token(
    x_api_key: Optional[str] = Header(None),
    token_query: Optional[str] = Query(None, alias="token"),
    token_cookie: Optional[str] = Cookie(None, alias="token"),
) -> Optional[str]:
    """Resolve the bearer token from header, query, or cookie (in priority order)."""
    return x_api_key or token_query or token_cookie


async def verify_auth(
    token: Optional[str] = Depends(_resolve_token),
) -> str:
    """
    Dependency: verify the token exists in the admin_tokens dict.
    Returns the token string on success, raises 401 on failure.
    """
    if not token:
        raise HTTPException(status_code=401, detail="Missing authentication token")
    role = settings.admin_tokens.get(token)
    if role is None:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return token


def _admin_short_id(token: str) -> str:
    """Return a short, identifiable prefix of the admin token for audit logs."""
    return token[:12] + "..."


def _get_client_ip(request: Request = None) -> str:
    """Extract client IP from a FastAPI request."""
    if request is None:
        return ""
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else ""


async def verify_uploader(token: str = Depends(verify_auth)) -> str:
    """
    Dependency: verify the token has "upload" (or "admin") role.
    View-only admins are rejected with 403.
    """
    role = settings.admin_tokens.get(token, "")
    if role not in ("upload", "admin"):
        raise HTTPException(
            status_code=403,
            detail="Insufficient permissions — upload role required",
        )
    return token


# =================================================================== #
#  Admin web UI                                                       #
# =================================================================== #

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse('<html><head><meta http-equiv="refresh" content="0; url=/admin"></head></html>')


@app.get("/login", response_class=HTMLResponse)
async def login_page(token: Optional[str] = Query(None)):
    """Serve the login page, or auto-redirect if a valid token is provided."""
    if token and token in settings.admin_tokens:
        html = _render_admin_page(token)
        return HTMLResponse(html)
    return HTMLResponse(LOGIN_HTML)


@app.get("/admin", response_class=HTMLResponse)
async def admin_page(token: Optional[str] = Depends(_resolve_token)):
    """Serve the admin web interface. Requires a valid token."""
    if not token or token not in settings.admin_tokens:
        return HTMLResponse(LOGIN_HTML)
    html = _render_admin_page(token)
    return HTMLResponse(html)


def _render_admin_page(token: str) -> str:
    """Render the admin HTML with the token embedded for the frontend JS."""
    html_path = HERE / "app" / "api_templates" / "index.html"
    if not html_path.exists():
        return "<h1>Admin page not found</h1><p>api_templates/index.html is missing.</p>"

    role = settings.admin_tokens.get(token, "view")
    html = html_path.read_text(encoding="utf-8")
    # Inject token and role into the HTML head for the frontend to use
    inject = (
        f'<script>window.__TOKEN__ = {json_mod.dumps(token)}; '
        f'window.__ROLE__ = {json_mod.dumps(role)};</script>\n'
    )
    html = html.replace("<head>", "<head>\n" + inject)
    return html


# =================================================================== #
#  Auth API endpoints                                                  #
# =================================================================== #

@app.post("/api/auth/login")
async def api_auth_login(body: dict, request: Request = None):
    """Validate a token and return its role. Used by the frontend login form."""
    token = body.get("token", "")
    admin_id = _admin_short_id(token) if token else "?"
    ip = _get_client_ip(request)
    role = settings.admin_tokens.get(token)
    if role is None:
        log_audit("LOGIN_FAIL", admin_id, f"from {ip}", success=False, ip=ip)
        raise HTTPException(status_code=401, detail="Invalid token")
    log_audit("LOGIN_OK", admin_id, f"role={role}", success=True, ip=ip)
    return {"success": True, "role": role, "token": token}


@app.get("/api/auth/me")
async def api_auth_me(token: str = Depends(verify_auth)):
    """Return current session info for the authenticated admin."""
    role = settings.admin_tokens.get(token, "")
    return {"authenticated": True, "role": role, "can_upload": role in ("upload", "admin")}


# =================================================================== #
#  Login page HTML  (self-contained, no external deps)                 #
# =================================================================== #

LOGIN_HTML = """\
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Login — Course Data Manager</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh; display: flex; align-items: center; justify-content: center;
  }
  .login-card {
    background: #fff; border-radius: 16px; padding: 2.5rem;
    box-shadow: 0 20px 60px rgba(0,0,0,.2); width: 100%; max-width: 400px;
  }
  .login-card h1 { font-size: 1.5rem; margin-bottom: .25rem; }
  .login-card p { color: #6b7280; font-size: .875rem; margin-bottom: 1.5rem; }
  .login-card label { display: block; font-size: .875rem; font-weight: 500; margin-bottom: .5rem; }
  .login-card input[type="text"],
  .login-card input[type="password"] {
    width: 100%; padding: .75rem 1rem; border: 1px solid #d1d5db; border-radius: 8px;
    font-size: .875rem; margin-bottom: 1rem; transition: border-color .2s;
  }
  .login-card input:focus { outline: none; border-color: #6366f1; box-shadow: 0 0 0 3px rgba(99,102,241,.15); }
  .login-card button {
    width: 100%; padding: .75rem; background: #6366f1; color: #fff; border: none;
    border-radius: 8px; font-size: 1rem; font-weight: 500; cursor: pointer; transition: background .2s;
  }
  .login-card button:hover { background: #4f46e5; }
  .login-card button:disabled { opacity: .6; cursor: not-allowed; }
  .login-card .error { color: #dc2626; font-size: .8rem; margin-top: .75rem; display: none; }
  .login-card .error.show { display: block; }
</style>
</head>
<body>
<div class="login-card">
  <h1>🔐 Course Data Manager</h1>
  <p>Enter your admin token to access the management panel</p>
  <label for="tokenInput">Admin Token</label>
  <input type="password" id="tokenInput" placeholder="Paste your token here" autocomplete="off" autofocus>
  <button id="loginBtn" onclick="doLogin()">Login</button>
  <div class="error" id="errorMsg">Invalid token. Please check with your administrator.</div>
</div>
<script>
  async function doLogin() {
    const token = document.getElementById('tokenInput').value.trim();
    const btn = document.getElementById('loginBtn');
    const err = document.getElementById('errorMsg');
    if (!token) { err.textContent = 'Please enter a token.'; err.classList.add('show'); return; }
    btn.disabled = true; btn.textContent = 'Verifying...'; err.classList.remove('show');
    try {
      const r = await fetch('/api/auth/login', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({token})
      });
      if (!r.ok) { throw new Error('Invalid token'); }
      const d = await r.json();
      // Store in localStorage + redirect to admin page with token
      window.location.href = '/admin?token=' + encodeURIComponent(token);
    } catch (e) {
      err.textContent = 'Invalid token. Please check with your administrator.';
      err.classList.add('show');
    } finally {
      btn.disabled = false; btn.textContent = 'Login';
    }
  }
  document.getElementById('tokenInput').addEventListener('keydown', e => {
    if (e.key === 'Enter') doLogin();
  });
</script>
</body>
</html>
"""


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
    _token: str = Depends(verify_uploader),
    request: Request = None,
):
    """
    Upload a CSV or JSON file containing course or assignment data.

    - ``source_type``: ``"courses"`` (default) or ``"assignments"``
    - Supports ``.csv`` and ``.json`` extensions.
    - Max file size: 10 MB.
    """
    admin_id = _admin_short_id(_token)
    ip = _get_client_ip(request)
    source_type_ok = source_type in ("courses", "assignments")
    if not source_type_ok:
        log_audit("UPLOAD", admin_id, f"invalid source_type={source_type}", success=False, ip=ip)
        raise HTTPException(400, "source_type must be 'courses' or 'assignments'")

    # — File size validation ------------------------------------------------
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_SIZE:
        log_audit("UPLOAD", admin_id, f"file too large: {len(raw)} bytes", success=False, ip=ip)
        raise HTTPException(413, f"File too large ({len(raw)} bytes). Maximum is {MAX_UPLOAD_SIZE // (1024*1024)} MB.")

    # — File extension validation -------------------------------------------
    filename = file.filename or "upload"
    ext = Path(filename).suffix.lower()
    if ext and ext not in ALLOWED_UPLOAD_EXTENSIONS:
        log_audit("UPLOAD", admin_id, f"invalid extension: {ext}", success=False, ip=ip)
        raise HTTPException(400, f"File type '{ext}' not supported. Allowed: CSV, JSON.")

    content = raw.decode("utf-8")

    try:
        if ext == ".csv":
            docs = parse_csv_content(content, source_type=source_type)
        elif ext == ".json":
            docs = parse_json_content(content, source_type=source_type)
        else:
            # No extension — auto-detect
            stripped = content.strip()
            if stripped.startswith("["):
                docs = parse_json_content(content, source_type=source_type)
            else:
                docs = parse_csv_content(content, source_type=source_type)

        if not docs:
            log_audit("UPLOAD", admin_id, f"no valid records in {filename}", success=False, ip=ip)
            raise HTTPException(400, "No valid records found in the file. Check column names and data format.")

        stored = await store_to_milvus(docs)
        log_audit("UPLOAD", admin_id, f"{filename} | {len(docs)} records → {stored} chunks", success=True, ip=ip)

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
        log_audit("UPLOAD", admin_id, f"{filename} | error: {exc}", success=False, ip=ip)
        raise HTTPException(500, f"Upload failed: {exc}")


@app.post("/api/ingest")
async def ingest_json(
    body: dict,
    _token: str = Depends(verify_uploader),
    request: Request = None,
):
    """
    Ingest course / assignment data as a raw JSON array.
    """
    admin_id = _admin_short_id(_token)
    ip = _get_client_ip(request)
    source_type = body.get("source_type", "courses")
    if source_type not in ("courses", "assignments"):
        log_audit("INGEST", admin_id, f"invalid source_type={source_type}", success=False, ip=ip)
        raise HTTPException(400, "source_type must be 'courses' or 'assignments'")

    raw = body.get("data", body)
    json_str = json_mod.dumps(raw) if isinstance(raw, (list, dict)) else str(raw)
    # Cap ingested data size (rough check ~10 MB serialized)
    if len(json_str) > MAX_UPLOAD_SIZE:
        log_audit("INGEST", admin_id, f"payload too large: {len(json_str)} bytes", success=False, ip=ip)
        raise HTTPException(413, f"Payload too large ({len(json_str)} bytes). Maximum is {MAX_UPLOAD_SIZE // (1024*1024)} MB.")

    try:
        docs = parse_json_content(json_str, source_type=source_type)
        if not docs:
            log_audit("INGEST", admin_id, "no valid records in payload", success=False, ip=ip)
            raise HTTPException(400, "No valid records found in payload.")

        stored = await store_to_milvus(docs)
        log_audit("INGEST", admin_id, f"{len(docs)} records → {stored} chunks", success=True, ip=ip)

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
async def stats(_token: str = Depends(verify_auth)):
    """
    Return a quick summary of what is stored in Milvus.

    Because Milvus does not expose a simple "count all" API without a
    query, this performs a broad similarity search and reports the
    unique course codes found in the top results.
    """
    try:
        from app.rag.retriever import get_retriever

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
