import uuid
from typing import List, Tuple, Optional
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import DEFAULT_EMBEDDING_MODEL
from core.chunker import DocumentChunk

class LegalVectorStore:
    """
    Vector storage and retrieval layer for legal document chunks using ChromaDB and HuggingFace embeddings.
    """

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, persist_directory: Optional[str] = None):
        self.model_name = model_name
        self.persist_directory = persist_directory
        self._embeddings = None
        self.collection_name = f"legal_{uuid.uuid4().hex}"
        self._vector_db: Optional[Chroma] = None
        self._indexed_count = 0
        self._chunks: List[DocumentChunk] = []

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._embeddings

    def build_index(self, chunks: List[DocumentChunk]) -> Chroma:
        """
        Creates or replaces the Chroma vector store with chunk embeddings and attached page metadata.
        """
        if not chunks:
            raise ValueError("No chunks provided to build vector index.")

        documents = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk.text,
                metadata={
                    "chunk_id": chunk.chunk_id,
                    "page_number": chunk.page_number,
                    "doc_name": chunk.doc_name,
                    "char_count": chunk.char_count,
                },
            )
            documents.append(doc)

        self.close()
        self._vector_db = Chroma.from_documents(
            documents=documents,
            collection_name=self.collection_name,
            ids=[chunk.chunk_id for chunk in chunks],
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
        )
        self._indexed_count = len(documents)
        self._chunks = list(chunks)
        return self._vector_db

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieves top-k relevant document chunks for a given query.
        """
        if self._vector_db is None:
            raise RuntimeError("Vector store has not been indexed yet. Call build_index first.")
        return self._vector_db.similarity_search(query, k=k)

    def similarity_search_with_score(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        Retrieves top-k document chunks with similarity distance scores.
        """
        if self._vector_db is None:
            raise RuntimeError("Vector store has not been indexed yet. Call build_index first.")
        return self._vector_db.similarity_search_with_score(query, k=k)

    @property
    def total_chunks(self) -> int:
        return self._indexed_count

    @property
    def chunks(self) -> List[DocumentChunk]:
        """Returns the chunks used to build this in-memory index."""
        return list(self._chunks)

    def close(self):
        """Delete only this document's collection; never another reviewer's index."""
        if self._vector_db is not None:
            self._vector_db.delete_collection()
            self._vector_db = None
        self._indexed_count = 0
        self._chunks = []
