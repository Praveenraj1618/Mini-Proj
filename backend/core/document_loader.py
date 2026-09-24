"""Page-grounded PDF extraction with optional local OCR and visual previews."""
import base64
import os
import unicodedata
from pathlib import Path
from core.review_profiles import OCR_LANGUAGES, detect_languages
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Union

import pymupdf as fitz
from config import OCR_ENABLED, OCR_LANGUAGE, OCR_DPI, OCR_TESSDATA, MAX_VISUAL_PAGES

# PyMuPDF is not thread-safe. Serialize PDF operations used by the threaded API.
PDF_LOCK = threading.RLock()


@dataclass
class ExtractedImage:
    page_num: int
    image_index: int
    base64_data: str
    mime_type: str = "image/png"
    width: int = 0
    height: int = 0


@dataclass
class PageData:
    page_num: int
    text: str
    char_count: int
    image_count: int
    extraction_method: str = "native"


@dataclass
class LoadedDocument:
    file_path: str
    file_name: str
    total_pages: int
    pages: List[PageData] = field(default_factory=list)
    images: List[ExtractedImage] = field(default_factory=list)
    full_text: str = ""
    total_words: int = 0
    total_chars: int = 0
    ocr_pages: List[int] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    visual_pages_total: int = 0
    languages: List[str] = field(default_factory=list)


class LegalDocumentLoader:
    def __init__(self, min_image_dim: int = 80, ocr_enabled: bool = OCR_ENABLED, source_language: str = "auto"):
        if source_language not in OCR_LANGUAGES:
            raise ValueError("Unsupported source language")
        self.source_language = source_language
        self.ocr_language = OCR_LANGUAGE if source_language == "auto" else OCR_LANGUAGES[source_language]
        self.min_image_dim = min_image_dim
        self.ocr_enabled = ocr_enabled

    def load_pdf(self, source: Union[str, bytes], file_name: Optional[str] = None) -> LoadedDocument:
        with PDF_LOCK:
            return self._load_pdf(source, file_name)

    def _load_pdf(self, source, file_name):
        if isinstance(source, bytes):
            if not source.startswith(b"%PDF-"):
                raise ValueError("Uploaded content is not a valid PDF file.")
            document = fitz.open(stream=source, filetype="pdf")
            resolved_path = "<memory>"
            resolved_name = os.path.basename(file_name or "uploaded.pdf")
        else:
            document = fitz.open(source)
            resolved_path = str(source)
            resolved_name = os.path.basename(source)

        with document:
            if document.needs_pass:
                raise ValueError("Password-protected PDFs must be unlocked before upload.")
            result = LoadedDocument(resolved_path, resolved_name, len(document))
            full_text = []
            for page in document:
                page_num = page.number + 1
                text = page.get_text("text").strip()
                method = "native"
                images = [i for i in page.get_image_info()
                          if i["width"] >= self.min_image_dim and i["height"] >= self.min_image_dim]
                drawings = bool(page.get_drawings())
                image_coverage = max(
                    (fitz.Rect(i["bbox"]).get_area() / max(page.rect.get_area(), 1) for i in images),
                    default=0,
                )
                # Also handle a scan beneath a small native-text header/footer.
                suspicious = any(c == '\ufffd' or (unicodedata.category(c) == 'Cc' and not c.isspace()) for c in text)
                suspicious = suspicious or (self.source_language in {'ta','hi'} and len(text) > 30 and (self.source_language not in detect_languages(text) or "unknown" in detect_languages(text)))
                if suspicious:
                    result.warnings.append(f'Page {page_num}: native text encoding may be damaged or differ from the selected language; OCR recovery requested.')
                needs_ocr = suspicious or (bool(images) and (len(text) < 80 or image_coverage >= 0.35)) or (not text and drawings)
                if needs_ocr:
                    if self.ocr_enabled:
                        try:
                            data_path = Path(fitz.get_tessdata(OCR_TESSDATA))
                            if any(not (data_path / (code + '.traineddata')).is_file() for code in self.ocr_language.split('+')):
                                raise RuntimeError('Required OCR language data is missing')
                            textpage = page.get_textpage_ocr(
                                language=self.ocr_language, dpi=OCR_DPI,
                                full=suspicious or not bool(text), tessdata=OCR_TESSDATA,
                            )
                            text = page.get_text("text", textpage=textpage).strip()
                            method = "ocr-recovery" if suspicious else "ocr" if not page.get_text("text").strip() else "native+ocr"
                            result.ocr_pages.append(page_num)
                        except Exception:
                            result.warnings.append(
                                f"Page {page_num}: OCR failed. Install Tesseract language data "
                                f"for '{self.ocr_language}' and set OCR_TESSDATA if needed; "
                                "this page may have incomplete searchable text."
                            )
                    else:
                        result.warnings.append(f"Page {page_num}: OCR is disabled; scanned text may be missing.")
                    if not text:
                        result.warnings.append(f"Page {page_num}: no readable text was extracted.")

                # Page rendering captures vector signatures/stamps as well as images.
                if images or drawings:
                    result.visual_pages_total += 1
                    if len(result.images) < MAX_VISUAL_PAGES:
                        try:
                            scale = min(1.5, 1600 / max(page.rect.width, page.rect.height, 1))
                            pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
                            result.images.append(ExtractedImage(
                                page_num, 1, base64.b64encode(pixmap.tobytes("png")).decode("ascii"),
                                width=pixmap.width, height=pixmap.height,
                            ))
                        except Exception:
                            result.warnings.append(f"Page {page_num}: visual preview could not be rendered.")

                result.pages.append(PageData(page_num, text, len(text), len(images), method))
                result.total_words += len(text.split())
                result.total_chars += len(text)
                if text:
                    full_text.append(f"--- [Page {page_num}] ---\n{text}")
            result.languages = detect_languages(" ".join(p.text for p in result.pages))
            result.full_text = "\n\n".join(full_text)
            if result.visual_pages_total > len(result.images):
                result.warnings.append(
                    f"Visual previews cover {len(result.images)} of {result.visual_pages_total} visual pages "
                    f"(MAX_VISUAL_PAGES={MAX_VISUAL_PAGES}). Text extraction still covers all pages."
                )
            return result
