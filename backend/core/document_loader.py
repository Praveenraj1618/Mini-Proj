import os
import pymupdf as fitz  # Modern PyMuPDF API
import base64
from io import BytesIO
from PIL import Image
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union

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

class LegalDocumentLoader:
    """
    High-fidelity PDF document loader tailored for legal documents.
    Extracts text per page, preserves structural layout, and captures visual assets (tables, signatures, stamps).
    """

    def __init__(self, min_image_dim: int = 80):
        self.min_image_dim = min_image_dim

    def load_pdf(self, source: Union[str, bytes], file_name: Optional[str] = None) -> LoadedDocument:
        """Load a PDF from a filesystem path or directly from uploaded bytes."""
        if isinstance(source, bytes):
            if not source.startswith(b"%PDF-"):
                raise ValueError("Uploaded content is not a valid PDF file.")
            doc = fitz.open(stream=source, filetype="pdf")
            resolved_path = "<memory>"
            resolved_name = os.path.basename(file_name or "uploaded.pdf")
        else:
            if not os.path.exists(source):
                raise FileNotFoundError(f"Document not found at: {source}")
            doc = fitz.open(source)
            resolved_path = source
            resolved_name = os.path.basename(source)
        total_pages = len(doc)

        pages_data: List[PageData] = []
        extracted_images: List[ExtractedImage] = []
        full_text_parts = []
        total_words = 0
        total_chars = 0

        for page_idx in range(total_pages):
            page = doc[page_idx]
            page_num = page_idx + 1
            
            # Extract formatted text
            text = page.get_text("text")
            # Strip excessive null or strange whitespace artifacts
            cleaned_text = text.strip()
            
            char_count = len(cleaned_text)
            word_count = len(cleaned_text.split())
            total_chars += char_count
            total_words += word_count
            
            # Extract images / tables on this page
            image_list = page.get_images(full=True)
            page_img_count = 0

            for img_idx, img_info in enumerate(image_list):
                try:
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image.get("image")
                    image_ext = base_image.get("ext", "png").lower()
                    width = base_image.get("width", 0)
                    height = base_image.get("height", 0)

                    # Filter out tiny decorative icons / 1px spacers
                    if width < self.min_image_dim or height < self.min_image_dim:
                        continue

                    # Standardize mime type
                    mime_type = f"image/{image_ext}" if image_ext in ["png", "jpeg", "jpg", "webp"] else "image/png"
                    
                    # Convert to base64
                    base64_str = base64.b64encode(image_bytes).decode("utf-8")
                    
                    extracted_images.append(
                        ExtractedImage(
                            page_num=page_num,
                            image_index=img_idx + 1,
                            base64_data=base64_str,
                            mime_type=mime_type,
                            width=width,
                            height=height,
                        )
                    )
                    page_img_count += 1
                except Exception:
                    # Ignore corrupted embedded images gracefully
                    continue

            pages_data.append(
                PageData(
                    page_num=page_num,
                    text=cleaned_text,
                    char_count=char_count,
                    image_count=page_img_count,
                )
            )

            if cleaned_text:
                full_text_parts.append(f"--- [Page {page_num}] ---\n{cleaned_text}")

        doc.close()

        return LoadedDocument(
            file_path=resolved_path,
            file_name=resolved_name,
            total_pages=total_pages,
            pages=pages_data,
            images=extracted_images,
            full_text="\n\n".join(full_text_parts),
            total_words=total_words,
            total_chars=total_chars,
        )
