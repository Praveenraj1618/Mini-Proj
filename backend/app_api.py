import json
import os
import sys
import threading
import time
import urllib.parse
import uuid
from contextlib import contextmanager
from email import policy
from email.parser import BytesParser
from http import cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from config import (MAX_UPLOAD_MB, check_api_key_configured, configured_llm_providers,
                    CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVAL_TOP_K, TOP_K_DENSE, TOP_K_BM25,
                    RRF_K, HYBRID_RETRIEVAL_ENABLED, RERANKER_ENABLED, OCR_ENABLED)
from core.chunker import LegalChunker
from core.document_loader import LegalDocumentLoader, LoadedDocument
from core.legal_reviewer import LegalReviewer
from langchain_core.messages import HumanMessage
from core.vector_store import LegalVectorStore


MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_JSON_BYTES = 1024 * 1024
MAX_SESSIONS = 100
SESSION_COOKIE = "legal_ai_session"
SESSION_DATA: Dict[str, Dict[str, Any]] = {}
SESSION_LOCK = threading.RLock()


from core.review_localization import localize_payload
from core.review_profiles import LANGUAGES, OCR_LANGUAGES, PROFILES

class RequestError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


class LegalAPIRequestHandler(BaseHTTPRequestHandler):
    """JSON API for the contract review application."""

    server_version = "LegalAI/2.2"

    def _session_id(self) -> str:
        if hasattr(self, "_cached_session_id"):
            return self._cached_session_id

        parsed = cookies.SimpleCookie(self.headers.get("Cookie", ""))
        morsel = parsed.get(SESSION_COOKIE)
        session_id = morsel.value if morsel else ""
        try:
            uuid.UUID(session_id)
        except (ValueError, AttributeError):
            session_id = str(uuid.uuid4())
            self._new_session_cookie = True

        with SESSION_LOCK:
            if session_id not in SESSION_DATA and len(SESSION_DATA) >= MAX_SESSIONS:
                idle = [key for key, value in SESSION_DATA.items() if not value.get("active_requests", 0)]
                if not idle:
                    self._session_unavailable = True
                    raise RequestError("All review sessions are busy. Please retry shortly.", 503)
                oldest = min(idle, key=lambda key: SESSION_DATA[key]["last_seen"])
                expired = SESSION_DATA.pop(oldest)
                self._close_reviewers(expired)
            session = SESSION_DATA.setdefault(
                session_id,
                {"doc1_reviewer": None, "doc2_reviewer": None, "last_seen": time.time(),
                 "request_lock": threading.RLock(), "active_requests": 0},
            )
            session["last_seen"] = time.time()

        self._cached_session_id = session_id
        return session_id

    def _session(self) -> Dict[str, Any]:
        session_id = self._session_id()
        with SESSION_LOCK:
            return SESSION_DATA[session_id]

    @staticmethod
    def _close_reviewers(session):
        for key in ("doc1_reviewer", "doc2_reviewer"):
            reviewer = session.get(key)
            if reviewer and reviewer.vector_store is not None:
                try:
                    reviewer.vector_store.close()
                except Exception:
                    # Cleanup must not turn a successful upload into a failed request.
                    pass
            session[key] = None

    @contextmanager
    def _locked_session(self):
        # Establish membership and pin the session atomically against eviction.
        with SESSION_LOCK:
            session = self._session()
            session["active_requests"] += 1
        try:
            with session["request_lock"]:
                yield session
        finally:
            with SESSION_LOCK:
                session["active_requests"] -= 1

    def _require_reviewer(self):
        reviewer = self._session().get("doc1_reviewer")
        if reviewer is None:
            raise RequestError("No document is loaded. Upload one first.")
        requested_id = self.headers.get("X-Document-Id")
        if requested_id and requested_id != reviewer.document_id:
            raise RequestError("The document changed. Retry using the current upload.", 409)
        self._apply_preferences(reviewer)
        return reviewer

    def _preferences(self):
        language = self.headers.get('X-Response-Language', 'en')
        category = self.headers.get('X-Document-Category', 'auto')
        if language not in LANGUAGES or (category != 'auto' and category not in PROFILES):
            raise RequestError('Unsupported response language or document category.')
        return language, category

    def _apply_preferences(self, reviewer):
        reviewer.set_preferences(*self._preferences())

    def _upload_language(self):
        language = self.headers.get('X-Source-Language', 'auto')
        if language not in OCR_LANGUAGES:
            raise RequestError('Unsupported source language.')
        return language

    def _set_headers(self, status: int, content_type: str, content_length: int = 0):
        self._response_started = True
        self.send_response(status)
        origin = self.headers.get("Origin")
        allowed_origins = {
            item.strip()
            for item in os.getenv(
                "CORS_ALLOWED_ORIGINS",
                "http://localhost:8080,http://127.0.0.1:8080,http://localhost:3000,http://localhost:5173",
            ).split(",")
            if item.strip()
        }
        if origin in allowed_origins:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Credentials", "true")
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Upload-Filename, X-Document-Id, X-Response-Language, X-Source-Language, X-Document-Category")
        self.send_header("Content-Type", content_type)
        if content_length:
            self.send_header("Content-Length", str(content_length))
        session_id = None if getattr(self, "_session_unavailable", False) else self._session_id()
        if session_id and getattr(self, "_new_session_cookie", False):
            self.send_header(
                "Set-Cookie",
                f"{SESSION_COOKIE}={session_id}; Path=/; HttpOnly; SameSite=Lax",
            )
        self.end_headers()

    def _send_bytes(self, content, content_type, status=200):
        if getattr(self, "_client_disconnected", False):
            return
        try:
            self._set_headers(status, content_type, len(content))
            self.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            self._client_disconnected = True
            self.close_connection = True
            self.log_message("Client disconnected while receiving %s; response stopped.", self.path)

    def _send_json(self, data: Any, status: int = 200):
        data = localize_payload(data, getattr(self, "headers", {}).get("X-Response-Language", "en"))
        encoded = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self._send_bytes(encoded, "application/json; charset=utf-8", status)

    def _read_body(self, maximum: int) -> bytes:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise RequestError("Invalid Content-Length header.") from exc
        if content_length <= 0:
            raise RequestError("Request body is required.")
        if content_length > maximum:
            raise RequestError(f"Request exceeds the {maximum // (1024 * 1024)} MB limit.", 413)
        return self.rfile.read(content_length)

    def _read_json(self) -> Dict[str, Any]:
        if self.headers.get_content_type() != "application/json":
            raise RequestError("Content-Type must be application/json.", 415)
        try:
            payload = json.loads(self._read_body(MAX_JSON_BYTES).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RequestError("Request body must contain valid UTF-8 JSON.") from exc
        if not isinstance(payload, dict):
            raise RequestError("JSON request body must be an object.")
        return payload

    def _read_uploaded_pdf(self) -> Tuple[bytes, str]:
        body = self._read_body(MAX_UPLOAD_BYTES + 64 * 1024)
        content_type = self.headers.get("Content-Type", "")
        media_type = self.headers.get_content_type()

        if media_type == "application/pdf":
            filename = self.headers.get("X-Upload-Filename", "uploaded.pdf")
            file_data = body
        elif media_type == "multipart/form-data":
            synthetic_message = (
                f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8")
                + body
            )
            message = BytesParser(policy=policy.default).parsebytes(synthetic_message)
            part = next(
                (candidate for candidate in message.iter_parts() if candidate.get_filename()),
                None,
            )
            if part is None:
                raise RequestError("Multipart request does not contain a file.")
            filename = part.get_filename() or "uploaded.pdf"
            file_data = part.get_payload(decode=True) or b""
        else:
            raise RequestError("Upload must use multipart/form-data or application/pdf.", 415)

        safe_name = Path(filename).name
        if not safe_name.lower().endswith(".pdf") or not file_data.startswith(b"%PDF-"):
            raise RequestError("Only valid PDF files are accepted.", 415)
        if len(file_data) > MAX_UPLOAD_BYTES:
            raise RequestError(f"PDF exceeds the {MAX_UPLOAD_MB} MB limit.", 413)
        return file_data, safe_name

    @staticmethod
    def _build_reviewer(file_data: bytes, filename: str, source_language: str = "auto") -> Tuple[LegalReviewer, int]:
        try:
            document = LegalDocumentLoader(source_language=source_language).load_pdf(file_data, file_name=filename)
        except (ValueError, RuntimeError) as exc:
            raise RequestError("The PDF could not be read. Check that it is valid and unlocked.", 422) from exc
        chunks = LegalChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP).chunk_document(document)
        if not chunks:
            detail = " ".join(document.warnings)
            raise RequestError("The PDF contains no readable document text. " + detail, 422)
        vector_store = LegalVectorStore()
        try:
            vector_store.build_index(chunks)
            return LegalReviewer(document, vector_store), len(chunks)
        except Exception:
            vector_store.close()
            raise

    def do_OPTIONS(self):
        self._set_headers(204, "application/json; charset=utf-8")

    def do_GET(self):
        try:
            with self._locked_session():
                self._handle_get()
        except RequestError as exc:
            self._send_json({"error": str(exc)}, exc.status)

    def _handle_get(self):
        path = urllib.parse.urlparse(self.path).path
        try:
            if path in ("/", "/index.html"):
                html_file = BASE_DIR.parent / "frontend" / "index.html"
                if not html_file.exists():
                    raise RequestError("frontend/index.html not found.", 404)
                content = html_file.read_bytes()
                self._send_bytes(content, "text/html; charset=utf-8")
                return

            if path in ("/api", "/api/status"):
                self._send_json({
                    "status": "online",
                    "service": "Legal AI Intelligence REST API",
                    "version": "2.2.0",
                    "llm_configured": check_api_key_configured(),
                    "llm_providers": configured_llm_providers(),
                    "vision_providers": configured_llm_providers(vision=True),
                    "ocr_enabled": OCR_ENABLED,
                    "languages": LANGUAGES,
                    "document_profiles": list(PROFILES.values()),
                    "retrieval": {
                        "chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP,
                        "top_k": RETRIEVAL_TOP_K, "dense_k": TOP_K_DENSE, "bm25_k": TOP_K_BM25,
                        "rrf_k": RRF_K, "hybrid": HYBRID_RETRIEVAL_ENABLED,
                        "reranker": HYBRID_RETRIEVAL_ENABLED and RERANKER_ENABLED,
                    },
                })
                return

            reviewer = None
            if path in ("/api/summary", "/api/heatmap", "/api/obligations", "/api/visuals"):
                reviewer = self._require_reviewer()
            if path == "/api/visuals":
                self._send_json({
                    "document_id": reviewer.document_id,
                    "visual_pages_total": reviewer.document.visual_pages_total,
                    "warnings": reviewer.document.warnings,
                    "previews": [{"page": img.page_num, "image_url": f"data:{img.mime_type};base64,{img.base64_data}"}
                                 for img in reviewer.document.images],
                })
                return

            if path == "/api/summary":
                classification = reviewer.classify_document()
                summary = reviewer.generate_executive_summary()
                self._send_json({
                    "classification": classification,
                    "summary": summary,
                    "diagnostic": reviewer.last_llm_diagnostic,
                })
            elif path == "/api/heatmap":
                self._send_json(reviewer.generate_clause_risk_heatmap())
            elif path == "/api/obligations":
                items = reviewer.extract_obligations()
                query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                if query.get("include_status") == ["1"]:
                    self._send_json({"items": items, **reviewer.obligation_status})
                else:
                    self._send_json(items)
            else:
                raise RequestError(f"Endpoint GET {path} not found.", 404)
        except RequestError as exc:
            self._send_json({"error": str(exc)}, exc.status)
        except Exception as exc:
            self.log_error("Unhandled GET error: %s", exc)
            if getattr(self, "_response_started", False):
                self.close_connection = True
            else:
                self._send_json({"error": "The server could not complete the request."}, 500)

    def do_POST(self):
        try:
            with self._locked_session():
                self._handle_post()
        except RequestError as exc:
            self._send_json({"error": str(exc)}, exc.status)

    def _handle_post(self):
        path = urllib.parse.urlparse(self.path).path
        try:
            if path == "/api/llm/check":
                self._read_json()
                reviewer = LegalReviewer(LoadedDocument("<none>", "connection-check", 0), None)
                try:
                    reviewer._invoke_llm([HumanMessage(content="Reply with the word OK.")], max_tokens=256)
                    self._send_json({"status": "ready", "message": reviewer.last_llm_diagnostic})
                except Exception as error:
                    self._send_json({"status": "failed", "diagnostic": reviewer._failure_details(error)})
                return

            if path == "/api/upload":
                self._preferences()
                source_language = self._upload_language()
                file_data, filename = self._read_uploaded_pdf()
                reviewer, chunk_count = self._build_reviewer(file_data, filename, source_language)
                self._apply_preferences(reviewer)
                session = self._session()
                self._close_reviewers(session)
                session["doc1_reviewer"] = reviewer
                session["doc2_reviewer"] = None
                classification = reviewer.classify_document()
                self._send_json({
                    "status": "success",
                    "file_name": reviewer.document.file_name,
                    "category_id": classification.get("category_id", "unknown"),
                    "negotiation_supported": classification.get("negotiation_supported", False),
                    "languages": reviewer.document.languages,
                    "doc_type": classification.get("doc_type", "Legal Agreement"),
                    "pages": reviewer.document.total_pages,
                    "words": reviewer.document.total_words,
                    "chunks": chunk_count,
                    "document_id": reviewer.document_id,
                    "ocr_pages": reviewer.document.ocr_pages,
                    "warnings": reviewer.document.warnings,
                })
                return

            if path == "/api/qa":
                question = str(self._read_json().get("question", "")).strip()
                if not question:
                    raise RequestError("A non-empty question is required.")
                reviewer = self._require_reviewer()
                self._send_json(reviewer.answer_query(question))
                return

            if path == "/api/negotiate":
                clause = str(self._read_json().get("clause", "")).strip()
                if not clause:
                    raise RequestError("A non-empty clause is required.")
                reviewer = self._session().get("doc1_reviewer")
                if reviewer is not None:
                    reviewer = self._require_reviewer()
                if reviewer is None:
                    empty_document = LoadedDocument(
                        file_path="<none>", file_name="standalone-clause", total_pages=0
                    )
                    reviewer = LegalReviewer(empty_document, None)
                self._apply_preferences(reviewer)
                self._send_json(reviewer.generate_negotiation_strategy(clause))
                return

            if path == "/api/visuals/analyze":
                reviewer = self._require_reviewer()
                self._send_json({"document_id": reviewer.document_id, "results": reviewer.analyze_visuals()})
                return

            if path == "/api/compare":
                reviewer_v1 = self._require_reviewer()
                file_data, filename = self._read_uploaded_pdf()
                reviewer_v2, _ = self._build_reviewer(file_data, filename, self._upload_language())
                try:
                    self._send_json(LegalReviewer.compare_contracts(reviewer_v1, reviewer_v2))
                finally:
                    if reviewer_v2.vector_store is not None:
                        reviewer_v2.vector_store.close()
                return

            raise RequestError(f"Endpoint POST {path} not found.", 404)
        except RequestError as exc:
            self._send_json({"error": str(exc)}, exc.status)
        except Exception as exc:
            self.log_error("Unhandled POST error: %s", exc)
            if getattr(self, "_response_started", False):
                self.close_connection = True
            else:
                self._send_json({"error": "The server could not complete the request."}, 500)


def run_server(port: int = 8080):
    server_address = ("", port)
    httpd = ThreadingHTTPServer(server_address, LegalAPIRequestHandler)
    print(f"Legal AI REST API Backend running at: http://localhost:{port}")
    httpd.serve_forever()


if __name__ == "__main__":
    run_server()
