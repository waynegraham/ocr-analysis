from __future__ import annotations
import os, time, json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pypdfium2 as pdfium
from easyocr import Reader
import torch

from utils.segment import segment_pages_to_sentences


def _want_gpu() -> bool:
    env = os.getenv("EASYOCR_GPU")
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes")
    return torch.cuda.is_available()

_GPU = _want_gpu()
_READER = Reader(lang_list=["en"], gpu=_GPU)

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
    preds = _READER.readtext(arr, detail=1, paragraph=False)
    texts, confs = [], []
    for p in preds:
        try:
            t = (p[1] or "").strip()
            c = float(p[2]) if len(p) > 2 else 0.0
        except Exception:
            t, c = "", 0.0
        if t:
            texts.append(" ".join(t.split()))
            confs.append(c)
    return " ".join(texts).strip(), confs

def to_json(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)

def ocr(pdf_path: Path) -> Dict[str, Any]:
    t0 = time.time()
    scale = float(os.getenv("EASYOCR_PDF_SCALE", "2.0"))
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
        "engine": "easyocr",
        "filesize_bytes": fsize,
        "num_pages": len(page_texts),
        "pages": [{"text": t} for t in page_texts],
        "segments": segments,
        "segment2page": seg2page,
        "avg_confidence": avg_conf,
        "engine_time": round(time.time() - t0, 2),
    }