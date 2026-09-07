import json
import os
import sys
import threading
import time
import urllib.parse
import uuid
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

from config import MAX_UPLOAD_MB, check_api_key_configured, configured_llm_providers
from core.chunker import LegalChunker
from core.document_loader import LegalDocumentLoader, LoadedDocument
from core.legal_reviewer import LegalReviewer
from core.vector_store import LegalVectorStore


MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_JSON_BYTES = 1024 * 1024
MAX_SESSIONS = 100
SESSION_COOKIE = "legal_ai_session"
SESSION_DATA: Dict[str, Dict[str, Any]] = {}
SESSION_LOCK = threading.Lock()


class RequestError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


class LegalAPIRequestHandler(BaseHTTPRequestHandler):
    """JSON API for the contract review application."""

    server_version = "LegalAI/2.1"

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
                oldest = min(SESSION_DATA, key=lambda key: SESSION_DATA[key]["last_seen"])
                del SESSION_DATA[oldest]
            session = SESSION_DATA.setdefault(
                session_id,
                {"doc1_reviewer": None, "doc2_reviewer": None, "last_seen": time.time()},
            )
            session["last_seen"] = time.time()

        self._cached_session_id = session_id
        return session_id

    def _session(self) -> Dict[str, Any]:
        session_id = self._session_id()
        with SESSION_LOCK:
            return SESSION_DATA[session_id]

    def _set_headers(self, status: int, content_type: str, content_length: int = 0):
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
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Upload-Filename")
        self.send_header("Content-Type", content_type)
        if content_length:
            self.send_header("Content-Length", str(content_length))
        session_id = self._session_id()
        if getattr(self, "_new_session_cookie", False):
            self.send_header(
                "Set-Cookie",
                f"{SESSION_COOKIE}={session_id}; Path=/; HttpOnly; SameSite=Lax",
            )
        self.end_headers()

    def _send_json(self, data: Any, status: int = 200):
        encoded = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self._set_headers(status, "application/json; charset=utf-8", len(encoded))
        self.wfile.write(encoded)

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
    def _build_reviewer(file_data: bytes, filename: str) -> Tuple[LegalReviewer, int]:
        document = LegalDocumentLoader().load_pdf(file_data, file_name=filename)
        chunks = LegalChunker().chunk_document(document)
        if not chunks:
            raise RequestError("The PDF contains no extractable text.", 422)
        vector_store = LegalVectorStore()
        vector_store.build_index(chunks)
        return LegalReviewer(document, vector_store), len(chunks)

    def do_OPTIONS(self):
        self._set_headers(204, "application/json; charset=utf-8")

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        try:
            if path in ("/", "/index.html"):
                html_file = BASE_DIR.parent / "frontend" / "index.html"
                if not html_file.exists():
                    raise RequestError("frontend/index.html not found.", 404)
                content = html_file.read_bytes()
                self._set_headers(200, "text/html; charset=utf-8", len(content))
                self.wfile.write(content)
                return

            if path in ("/api", "/api/status"):
                self._send_json({
                    "status": "online",
                    "service": "Legal AI Intelligence REST API",
                    "version": "2.1.0",
                    "llm_configured": check_api_key_configured(),
                    "llm_providers": configured_llm_providers(),
                })
                return

            reviewer = self._session().get("doc1_reviewer")
            if path in ("/api/summary", "/api/heatmap", "/api/obligations") and not reviewer:
                raise RequestError("No contract is loaded. Upload one first.")

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
                self._send_json(reviewer.extract_obligations())
            else:
                raise RequestError(f"Endpoint GET {path} not found.", 404)
        except RequestError as exc:
            self._send_json({"error": str(exc)}, exc.status)
        except Exception as exc:
            self.log_error("Unhandled GET error: %s", exc)
            self._send_json({"error": "The server could not complete the request."}, 500)

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        try:
            if path == "/api/upload":
                file_data, filename = self._read_uploaded_pdf()
                reviewer, chunk_count = self._build_reviewer(file_data, filename)
                session = self._session()
                session["doc1_reviewer"] = reviewer
                session["doc2_reviewer"] = None
                classification = reviewer.classify_document()
                self._send_json({
                    "status": "success",
                    "file_name": reviewer.document.file_name,
                    "doc_type": classification.get("doc_type", "Legal Agreement"),
                    "pages": reviewer.document.total_pages,
                    "words": reviewer.document.total_words,
                    "chunks": chunk_count,
                })
                return

            if path == "/api/qa":
                question = str(self._read_json().get("question", "")).strip()
                if not question:
                    raise RequestError("A non-empty question is required.")
                reviewer = self._session().get("doc1_reviewer")
                if not reviewer:
                    raise RequestError("No contract is loaded. Upload one first.")
                self._send_json(reviewer.answer_query(question))
                return

            if path == "/api/negotiate":
                clause = str(self._read_json().get("clause", "")).strip()
                if not clause:
                    raise RequestError("A non-empty clause is required.")
                reviewer = self._session().get("doc1_reviewer")
                if reviewer is None:
                    empty_document = LoadedDocument(
                        file_path="<none>", file_name="standalone-clause", total_pages=0
                    )
                    reviewer = LegalReviewer(empty_document, None)
                self._send_json(reviewer.generate_negotiation_strategy(clause))
                return

            if path == "/api/compare":
                reviewer_v1 = self._session().get("doc1_reviewer")
                if not reviewer_v1:
                    raise RequestError("Upload the first contract before comparing a second version.")
                file_data, filename = self._read_uploaded_pdf()
                reviewer_v2, _ = self._build_reviewer(file_data, filename)
                self._session()["doc2_reviewer"] = reviewer_v2
                self._send_json(LegalReviewer.compare_contracts(reviewer_v1, reviewer_v2))
                return

            raise RequestError(f"Endpoint POST {path} not found.", 404)
        except RequestError as exc:
            self._send_json({"error": str(exc)}, exc.status)
        except Exception as exc:
            self.log_error("Unhandled POST error: %s", exc)
            self._send_json({"error": "The server could not complete the request."}, 500)


def run_server(port: int = 8080):
    server_address = ("", port)
    httpd = ThreadingHTTPServer(server_address, LegalAPIRequestHandler)
    print(f"Legal AI REST API Backend running at: http://localhost:{port}")
    httpd.serve_forever()


if __name__ == "__main__":
    run_server()
