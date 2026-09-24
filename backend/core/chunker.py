from dataclasses import dataclass
import re
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.document_loader import LoadedDocument
from config import CHUNK_SIZE, CHUNK_OVERLAP

@dataclass
class DocumentChunk:
    chunk_id: str
    page_number: int
    text: str
    doc_name: str
    char_count: int


def is_evaluation_artifact_text(text: str) -> bool:
    """Detect non-contract benchmark notes and question appendices."""
    normalized = " ".join(text.lower().split())
    artifact_markers = (
        "evaluation design notes — do not use as contract terms",
        "evaluation design notes - do not use as contract terms",
        "known evaluation targets",
        "suggested benchmark questions",
    )
    if any(marker in normalized for marker in artifact_markers):
        return True

    segments = [
        segment.strip()
        for segment in re.split(r"[\n•]+", text)
        if segment.strip()
    ]
    question_segments = sum(segment.endswith("?") for segment in segments)
    return question_segments >= 3 and question_segments / max(1, len(segments)) >= 0.5

class LegalChunker:
    """
    Splits legal documents while maintaining page-level grounding, clause boundaries, and section context.
    """

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n\n",
                "\n\n",
                "\nARTICLE ",
                "\nSECTION ",
                "\nClause ",
                "\n§",
                "\n1.",
                "\n2.",
                "\n3.",
                "\n(a)",
                "\n(b)",
                "\n",
                "।",
                "॥",
                ". ",
                "; ",
                " ",
                "",
            ],
            keep_separator=True,
        )

    def chunk_document(self, doc: LoadedDocument) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        chunk_counter = 1

        for page in doc.pages:
            if not page.text.strip():
                continue
            if is_evaluation_artifact_text(page.text):
                continue

            split_texts = self.splitter.split_text(page.text)
            for split_idx, text_segment in enumerate(split_texts):
                cleaned = text_segment.strip()
                if not cleaned:
                    continue

                chunk_obj = DocumentChunk(
                    chunk_id=f"{doc.file_name}_p{page.page_num}_c{split_idx + 1}_{chunk_counter}",
                    page_number=page.page_num,
                    text=cleaned,
                    doc_name=doc.file_name,
                    char_count=len(cleaned),
                )
                chunks.append(chunk_obj)
                chunk_counter += 1

        return chunks
