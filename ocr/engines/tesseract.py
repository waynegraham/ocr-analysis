# ocr/engines/tesseract.py
from __future__ import annotations
import os, time, json
from pathlib import Path
from typing import Dict, Any, List

import pytesseract
from pytesseract import Output
import pypdfium2 as pdfium
import numpy as np

from utils.segment import segment_pages_to_sentences


def _pdf_pages_to_numpy(pdf_path: Path, scale: float = 2.0) -> List[np.ndarray]:
    """Render each page of the PDF into a numpy RGB array using pdfium."""
    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        out: List[np.ndarray] = []
        for i in range(len(doc)):
            page = doc.get_page(i)
            pil = page.render(scale=scale).to_pil().convert("RGB")
            page.close()
            out.append(np.array(pil))
        return out
    finally:
        doc.close()


def _ocr_page(arr: np.ndarray) -> tuple[str, List[float]]:
    """Run Tesseract OCR on a page image, return text and word-level confidences."""
    data = pytesseract.image_to_data(arr, output_type=Output.DICT)
    texts, confs = [], []
    n = len(data["text"])
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = -1
        if txt:
            texts.append(txt)
            if conf >= 0:
                confs.append(conf)
    return " ".join(texts).strip(), confs


def to_json(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def ocr(pdf_path: Path) -> Dict[str, Any]:
    """OCR the PDF using Tesseract, return unified schema with avg_confidence + engine_time."""
    t0 = time.time()
    scale = float(os.getenv("TESSERACT_PDF_SCALE", "2.0"))
    imgs = _pdf_pages_to_numpy(pdf_path, scale=scale)

    page_texts: List[str] = []
    confs_all: List[float] = []
    for arr in imgs:
        txt, confs = _ocr_page(arr)
        page_texts.append(txt or "")
        confs_all.extend(confs or [])

    # Sentence segmentation + mapping
    segments, seg2page = segment_pages_to_sentences(page_texts)

    # File size
    try:
        fsize = pdf_path.stat().st_size
    except Exception:
        fsize = -1

    avg_conf = (sum(confs_all) / len(confs_all)) if confs_all else None
    return {
        "engine": "tesseract",
        "filesize_bytes": fsize,
        "num_pages": len(page_texts),
        "pages": [{"text": t} for t in page_texts],
        "segments": segments,
        "segment2page": seg2page,
        "avg_confidence": avg_conf,
        "engine_time": round(time.time() - t0, 2),
    }