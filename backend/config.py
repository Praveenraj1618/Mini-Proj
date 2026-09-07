import os
import sys
from pathlib import Path
from typing import List, Tuple
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables from .env file (check backend dir and parent root)
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
load_dotenv(dotenv_path=BASE_DIR / ".env")
load_dotenv(dotenv_path=BASE_DIR.parent / ".env")


# Embedding Configuration
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Chunking & Retrieval Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "3"))

# Hybrid Retrieval & Reranking Configuration
HYBRID_RETRIEVAL_ENABLED = os.getenv("HYBRID_RETRIEVAL_ENABLED", "true").lower() == "true"
TOP_K_DENSE = int(os.getenv("TOP_K_DENSE", "10"))
TOP_K_BM25 = int(os.getenv("TOP_K_BM25", "10"))
RRF_K = int(os.getenv("RRF_K", "60"))
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))
LLM_REQUEST_TIMEOUT = float(os.getenv("LLM_REQUEST_TIMEOUT", "45"))


def configured_llm_providers() -> List[str]:
    """Return configured providers in the order in which they will be tried."""
    if os.getenv("LEGAL_AI_OFFLINE", "false").lower() == "true":
        return []

    available = {
        "xai": bool(os.getenv("XAI_API_KEY")),
        "openai": bool(os.getenv("OPENAI_API_KEY")),
        "groq": bool(os.getenv("GROQ_API_KEY")),
        "gemini": bool(os.getenv("GEMINI_API_KEY")),
    }
    requested_order = [
        item.strip().lower()
        for item in os.getenv("LLM_PROVIDER_ORDER", "xai,openai,groq,gemini").split(",")
        if item.strip()
    ]
    return [name for name in requested_order if available.get(name)]


def get_llm_candidates(
    vision: bool = False,
    temperature: float = 0.1,
) -> List[Tuple[str, ChatOpenAI]]:
    """Build configured LLM clients in failover order."""
    candidates: List[Tuple[str, ChatOpenAI]] = []

    for provider in configured_llm_providers():
        if provider == "xai":
            candidates.append((
                "xAI",
                ChatOpenAI(
                    model=os.getenv("XAI_MODEL", "grok-4"),
                    api_key=os.environ["XAI_API_KEY"],
                    base_url="https://api.x.ai/v1",
                    temperature=temperature,
                    max_retries=1,
                    request_timeout=LLM_REQUEST_TIMEOUT,
                ),
            ))
        elif provider == "openai":
            candidates.append((
                "OpenAI",
                ChatOpenAI(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o"),
                    api_key=os.environ["OPENAI_API_KEY"],
                    temperature=temperature,
                    max_retries=1,
                    request_timeout=LLM_REQUEST_TIMEOUT,
                ),
            ))
        elif provider == "groq":
            model_name = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
            default_effort = "none" if model_name.startswith("qwen/") else "low"
            candidates.append((
                "Groq",
                ChatOpenAI(
                    model=model_name,
                    api_key=os.environ["GROQ_API_KEY"],
                    base_url="https://api.groq.com/openai/v1",
                    temperature=temperature,
                    reasoning_effort=os.getenv("GROQ_REASONING_EFFORT", default_effort),
                    extra_body={"include_reasoning": False},
                    max_retries=1,
                    request_timeout=LLM_REQUEST_TIMEOUT,
                ),
            ))
        elif provider == "gemini":
            candidates.append((
                "Gemini",
                ChatOpenAI(
                    model=os.getenv("GEMINI_MODEL", "gemini-3.7-flash"),
                    api_key=os.environ["GEMINI_API_KEY"],
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                    temperature=temperature,
                    max_retries=1,
                    request_timeout=LLM_REQUEST_TIMEOUT,
                ),
            ))

    return candidates


def get_llm(vision: bool = False, temperature: float = 0.1):
    """
    Return the first configured LLM client.

    LegalReviewer uses get_llm_candidates() to fail over between providers; this
    function remains as the single-client compatibility entry point.
    """
    candidates = get_llm_candidates(vision=vision, temperature=temperature)
    if candidates:
        return candidates[0][1]

    # Compatibility placeholder. Callers normally guard this with
    # check_api_key_configured(), so it should never make a network request.
    return ChatOpenAI(
        model="grok-4",
        api_key="NO_KEY_SET",
        base_url="https://api.x.ai/v1",
        temperature=temperature,
    )

def check_api_key_configured() -> bool:
    """Returns True if any supported LLM API key is configured."""
    return bool(configured_llm_providers())
