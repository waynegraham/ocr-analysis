from __future__ import annotations
import os, time, json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pypdfium2 as pdfium

from paddleocr import PaddleOCR
from utils.segment import segment_pages_to_sentences

# ---------------- Config ----------------
def _want_gpu() -> bool:
    env = os.getenv("PPOCR_GPU")
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes")
    # let Paddle choose; default False for cross-env safety
    return False

_GPU = _want_gpu()

# Languages: 'en' for English; add others if needed.
# rec, det models auto-selected by PaddleOCR for lang='en' (PP-OCRv5 series)
_OCR = PaddleOCR(lang='en', use_gpu=_GPU, show_log=False)

def _pdf_pages_to_numpy(pdf_path: Path, scale: float = 2.0) -> List[np.ndarray]:
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

def _ocr_page(arr: np.ndarray) -> Tuple[str, List[float]]:
    # PaddleOCR expects file path or numpy array (H,W,C) BGR? It accepts RGB ndarray too.
    # result is list: for each block: [ [box], (text, conf) ] or similar
    res = _OCR.ocr(arr, cls=True)
    texts, confs = [], []
    if res and isinstance(res, list):
        # res is [ page_result ], and page_result is list of lines
        page_result = res[0] if len(res) > 0 else []
        for line in page_result or []:
            try:
                txt = (line[1][0] or "").strip()
                conf = float(line[1][1]) if line[1][1] is not None else 0.0
            except Exception:
                txt, conf = "", 0.0
            if txt:
                texts.append(" ".join(txt.split()))
                confs.append(conf)
    return " ".join(texts).strip(), confs

def to_json(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)

def ocr(pdf_path: Path) -> Dict[str, Any]:
    t0 = time.time()
    scale = float(os.getenv("PPOCR_PDF_SCALE", "2.0"))
    imgs = _pdf_pages_to_numpy(pdf_path, scale=scale)

    page_texts: List[str] = []
    confs_all: List[float] = []
    for arr in imgs:
        txt, confs = _ocr_page(arr)
        page_texts.append(txt or "")
        confs_all.extend(confs or [])

    segments, seg2page = segment_pages_to_sentences(page_texts)

    try:
        fsize = pdf_path.stat().st_size
    except Exception:
        fsize = -1

    avg_conf = (sum(confs_all) / len(confs_all)) if confs_all else None

    return {
        "engine": "paddleocr",
        "filesize_bytes": fsize,
        "num_pages": len(page_texts),
        "pages": [{"text": t} for t in page_texts],
        "segments": segments,
        "segment2page": seg2page,
        "avg_confidence": avg_conf,
        "engine_time": round(time.time() - t0, 2),
    }