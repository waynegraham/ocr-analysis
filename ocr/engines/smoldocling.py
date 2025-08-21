import time
import json
from pathlib import Path

from utils.segment import segment_text

# Strict: only Docling
try:
    # Adjust the import path if your installed docling version uses a different module layout
    from docling.document_converter import DocumentConverter
except Exception as e:
    raise ImportError(
        "Docling is required for the 'smoldocling' engine. "
        "Install it in your image (e.g., add `docling` to requirements). "
        f"Import error: {e}"
    )

# Initialize once at import; if this fails, we want a hard error
_dl_converter = DocumentConverter()

def _extract_text_with_docling(pdf_path: Path) -> str:
    """
    Convert the PDF with Docling and aggregate page text.
    Depending on Docling version, page text accessors may differ.
    This implementation tries page-level text first, then a whole-doc export.
    """
    result = _dl_converter.convert(str(pdf_path))

    # Try page-by-page extraction
    texts = []
    try:
        pages = getattr(result.document, "pages", None)
        if pages is not None:
            for page in pages:
                # Prefer explicit page text API if available
                if hasattr(page, "to_text"):
                    t = page.to_text()
                elif hasattr(page, "text"):
                    t = page.text
                else:
                    t = ""
                if t:
                    texts.append(t)
    except Exception:
        # If page iteration fails, fall back to a whole-doc export
        texts = []

    if not texts:
        # Whole document fallback
        try:
            if hasattr(result.document, "export_to_text"):
                whole = result.document.export_to_text()
                if whole:
                    return whole
        except Exception:
            pass

    return "\n".join(texts)

def ocr(pdf_path):
    """
    SmolDocling (Docling-only) OCR/parse:
      Returns:
        {
          "segments": [...],           # sentence/paragraph segments
          "avg_confidence": None,      # Docling does not emit token confidences
          "processing_time": seconds
        }
    """
    pdf_path = Path(pdf_path)
    start = time.time()

    text = _extract_text_with_docling(pdf_path)
    segments = segment_text(text)
    duration = round(time.time() - start, 2)

    print(f"📊 {pdf_path.name} - SmolDocling processed in {duration}s")

    return {
        "segments": segments,
        "avg_confidence": None,   # no per-token confidence from Docling
        "processing_time": duration,
    }

def to_json(result):
    return json.dumps(result, indent=2)