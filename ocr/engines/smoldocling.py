from __future__ import annotations
import os, time, json
from pathlib import Path
from typing import Dict, Any, List

from docling.document_converter import DocumentConverter
from utils.segment import segment_pages_to_sentences

# Initialize converter once
# You can pass kwargs to adjust models / disable GPU if needed.
# E.g., converter = DocumentConverter(use_gpu=False)
_use_gpu_env = os.getenv("DOCLING_GPU")
_use_gpu = _use_gpu_env and _use_gpu_env.strip().lower() in ("1", "true", "yes")
converter = DocumentConverter(use_gpu=_use_gpu)

def to_json(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)

def _extract_page_texts(doc) -> List[str]:
    """
    Extract plain text for each page from a Docling document object.
    Fallback: if page text not available, join block/text lines per page.
    """
    pages: List[str] = []
    # Docling 2.x exposes a "pages" iterable; adapt if your version differs.
    for p in doc.pages:
        # Try the simplest path first
        txt = (getattr(p, "text", None) or "").strip()
        if not txt:
            # Fallback: accumulate from elements
            lines = []
            try:
                for b in p.blocks or []:
                    t = getattr(b, "text", None)
                    if t: lines.append(str(t))
            except Exception:
                pass
            txt = "\n".join(lines).strip()
        pages.append(txt)
    return pages

def ocr(pdf_path: Path) -> Dict[str, Any]:
    t0 = time.time()
    # Convert
    result = converter.convert(pdf_path)       # returns a DoclingDocument
    doc = result.document

    # Per-page text
    page_texts = _extract_page_texts(doc)

    # Segment + mapping
    segments, seg2page = segment_pages_to_sentences(page_texts)

    try:
        fsize = pdf_path.stat().st_size
    except Exception:
        fsize = -1

    return {
        "engine": "smoldocling",
        "filesize_bytes": fsize,
        "num_pages": len(page_texts),
        "pages": [{"text": t} for t in page_texts],
        "segments": segments,
        "segment2page": seg2page,
        "avg_confidence": None,                 # docling doesn't expose it
        "engine_time": round(time.time() - t0, 2),
    }