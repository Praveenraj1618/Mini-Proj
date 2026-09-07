import re
from typing import Any, List, Tuple, Dict
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from config import RERANKER_MODEL, RRF_K, TOP_K_DENSE, TOP_K_BM25, RERANKER_ENABLED
from core.chunker import DocumentChunk, is_evaluation_artifact_text
from core.vector_store import LegalVectorStore

def tokenize_legal_text(text: str) -> List[str]:
    """Tokenizes text for BM25, stripping punctuation and converting to lowercase."""
    cleaned = re.sub(r"[^\w\s§]", " ", text.lower())
    return [token for token in cleaned.split() if len(token) > 1]


def expand_legal_query(query: str) -> str:
    """Add conservative legal synonyms for lexical candidate generation."""
    lowered = query.lower()
    expansions = []
    rules = (
        (("party", "parties", "entity", "incorporation"), "customer service provider contracting entities company corporation limited llc pvt ltd"),
        (("payment", "invoice", "fee", "late"), "fees invoicing payable overdue interest taxes"),
        (("terminate", "termination", "breach", "default"), "term non-renewal cure notice suspension remedies"),
        (("governing", "jurisdiction", "dispute", "arbitration"), "applicable law venue seat arbitrator resolution"),
        (("intellectual property", "ownership", "owns", "license"), "right title interest customer data materials work product"),
        (("confidential", "non-disclosure", "trade secret"), "confidential information receiving disclosing disclosure protective treatment"),
        (("liability", "damages", "indemnification"), "limitation aggregate cap indirect consequential indemnify claims"),
        (("notice", "notices"), "written courier email receipt days"),
    )
    for triggers, addition in rules:
        if any(trigger in lowered for trigger in triggers):
            expansions.append(addition)
    return " ".join([query, *expansions])


def is_question_list_chunk(text: str) -> bool:
    """Detect prompt/benchmark appendices that list questions without answers."""
    return is_evaluation_artifact_text(text)

class BM25Retriever:
    """
    Sparse keyword retriever using the BM25Okapi algorithm.
    Ideal for matching exact legal terms, clause numbers, dates, and party names.
    """

    def __init__(self, chunks: List[DocumentChunk]):
        self.chunks = [chunk for chunk in chunks if not is_evaluation_artifact_text(chunk.text)]
        self.corpus_tokens = [tokenize_legal_text(c.text) for c in self.chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens) if self.corpus_tokens else None

    def search(self, query: str, top_k: int = 10) -> List[Tuple[DocumentChunk, float]]:
        if not self.bm25 or not self.chunks:
            return []

        tokenized_query = tokenize_legal_text(expand_legal_query(query))
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        # Pair chunk with score
        scored_chunks = list(zip(self.chunks, scores))
        # Sort descending by score
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_k]


class LegalCrossEncoderReranker:
    """
    Cross-Encoder Reranker that scores and re-orders document chunks based on exact query interaction.
    """

    def __init__(self, model_name: str = RERANKER_MODEL):
        self.model_name = model_name
        self._cross_encoder = None

    @property
    def cross_encoder(self) -> Any:
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder
            self._cross_encoder = CrossEncoder(self.model_name, device="cpu")
        return self._cross_encoder

    def rerank(self, query: str, candidate_docs: List[Document], top_k: int = 5) -> List[Tuple[Document, float]]:
        if not candidate_docs:
            return []

        pairs = [[query, doc.page_content] for doc in candidate_docs]
        scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
        scored_results = []
        for doc, score in zip(candidate_docs, scores):
            numeric_score = float(score)
            doc.metadata["reranker_score"] = round(numeric_score, 4)
            scored_results.append((doc, numeric_score))

        scored_results.sort(key=lambda x: x[1], reverse=True)
        return scored_results[:top_k]


class LegalHybridRetriever:
    """
    Hybrid Retrieval Engine combining Dense Vector Search (ChromaDB)
    and Sparse Lexical Search (BM25Okapi) using Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, vector_store: LegalVectorStore, chunks: List[DocumentChunk], rrf_k: int = RRF_K):
        self.vector_store = vector_store
        self.chunks = chunks
        self.bm25_retriever = BM25Retriever(chunks)
        self.reranker = LegalCrossEncoderReranker()
        self.rrf_k = rrf_k
        self._artifact_chunk_count = sum(
            is_evaluation_artifact_text(chunk.text) for chunk in chunks
        )

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        dense_k: int = TOP_K_DENSE,
        bm25_k: int = TOP_K_BM25,
        use_reranker: bool = RERANKER_ENABLED,
    ) -> List[Document]:
        """
        Executes hybrid retrieval:
        1. Dense retrieval via ChromaDB
        2. Sparse retrieval via BM25
        3. Reciprocal Rank Fusion (RRF) to merge candidate lists
        4. Cross-Encoder reranking (optional)
        """
        # 1. Dense retrieval
        dense_pool_k = min(len(self.chunks), dense_k + self._artifact_chunk_count)
        dense_docs = [
            doc for doc in self.vector_store.similarity_search(query, k=dense_pool_k)
            if not is_question_list_chunk(doc.page_content)
        ][:dense_k]
        
        # 2. Sparse BM25 retrieval
        bm25_results = [
            result for result in self.bm25_retriever.search(query, top_k=bm25_k)
            if not is_question_list_chunk(result[0].text)
        ]
        
        # Convert BM25 chunks to Document objects
        bm25_docs = []
        for chunk, score in bm25_results:
            doc = Document(
                page_content=chunk.text,
                metadata={
                    "chunk_id": chunk.chunk_id,
                    "page_number": chunk.page_number,
                    "doc_name": chunk.doc_name,
                    "char_count": chunk.char_count,
                    "bm25_score": round(score, 4),
                },
            )
            bm25_docs.append(doc)

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        # Rank dense results
        for rank, doc in enumerate(dense_docs, start=1):
            chunk_id = doc.metadata.get("chunk_id", str(hash(doc.page_content)))
            doc_map[chunk_id] = doc
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (self.rrf_k + rank))

        # Rank BM25 results
        for rank, doc in enumerate(bm25_docs, start=1):
            chunk_id = doc.metadata.get("chunk_id", str(hash(doc.page_content)))
            doc_map[chunk_id] = doc
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (self.rrf_k + rank))

        # Sort combined candidate pool by RRF score descending
        fused_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        for chunk_id, score in fused_chunks:
            doc_map[chunk_id].metadata["rrf_score"] = round(score, 6)
        candidate_docs = [doc_map[chunk_id] for chunk_id, _ in fused_chunks]

        # 4. Cross-Encoder Reranking
        if use_reranker and candidate_docs:
            try:
                reranked_tuples = self.reranker.rerank(query, candidate_docs, top_k=top_k)
                return [doc for doc, _ in reranked_tuples]
            except Exception as exc:
                # Retrieval remains available offline if the optional reranker model
                # has not been downloaded yet.
                for doc in candidate_docs:
                    doc.metadata["reranker_error"] = str(exc)

        return candidate_docs[:top_k]
