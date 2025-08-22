#!/usr/bin/env python3
# ocr/runner.py
import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import pypdfium2 as pdfium  # make sure this is in your requirements

PDF_DIR_DEFAULT = "/app/data/pdfs"
OUTPUT_DIR_DEFAULT = "/app/data/ocr_output"


def get_engine(engine_name: str):
    """
    Dynamically import an engine module from engines/<engine_name>.py
    The module must expose:
      - ocr(pdf_path: Path) -> Dict
      - to_json(data: Dict) -> str
    """
    return importlib.import_module(f"engines.{engine_name}")


def _filesize_bytes(path: Path) -> int:
    try:
        return path.stat().st_size
    except Exception:
        return -1


def _num_pages_from_pdf(path: Path) -> Optional[int]:
    try:
        doc = pdfium.PdfDocument(str(path))
        n = len(doc)
        doc.close()
        return n
    except Exception:
        return None


def _normalize_pages(pages_field) -> List[Dict[str, str]]:
    """
    Accepts:
      - None
      - List[str]
      - List[{"text": "..."}]
    Returns List[{"text": str}, ...]
    """
    pages_norm: List[Dict[str, str]] = []
    if not pages_field:
        return pages_norm
    if isinstance(pages_field, list):
        for p in pages_field:
            if isinstance(p, str):
                pages_norm.append({"text": p})
            elif isinstance(p, dict):
                # common keys: text, content, value
                t = p.get("text") or p.get("content") or p.get("value") or ""
                pages_norm.append({"text": t})
            else:
                pages_norm.append({"text": ""})
    return pages_norm


def _infer_segment2page_from_pages(segments: List[str], pages: List[Dict[str, str]]) -> List[Optional[int]]:
    """
    Heuristic: for each segment, find the first page whose text contains a short probe of the segment.
    Returns a list of page indices (or None) of same length as segments.
    """
    if not segments or not pages:
        return [None] * len(segments)

    # pre-normalize page text
    norm_pages = []
    for p in pages:
        txt = " ".join((p.get("text") or "").split()).lower()
        norm_pages.append(txt)

    seg2page: List[Optional[int]] = []
    for seg in segments:
        needle = " ".join((seg or "").split()).lower()
        if not needle:
            seg2page.append(None)
            continue
        probe = needle[:120]  # limit for speed and tolerance
        found = None
        for i, hay in enumerate(norm_pages):
            if probe and probe in hay:
                found = i
                break
        seg2page.append(found)
    return seg2page


def _ensure_enriched_fields(raw: Dict[str, Any], pdf_path: Path) -> Dict[str, Any]:
    """
    Enforce presence of:
      - filesize_bytes
      - num_pages
      - pages: [{text:...}]
      - segments: [str,...]  (kept as-is)
      - segment2page: [int|None, ...] (prefer engine-provided; else infer if pages exist; else None)
    Also keeps existing fields like engine, avg_confidence, processing_time, etc.
    """
    data = dict(raw) if raw else {}

    # filesize
    if "filesize_bytes" not in data or not isinstance(data.get("filesize_bytes"), int):
        data["filesize_bytes"] = _filesize_bytes(pdf_path)

    # pages (normalize)
    data["pages"] = _normalize_pages(data.get("pages"))

    # num_pages
    if "num_pages" in data and isinstance(data["num_pages"], int) and data["num_pages"] >= 0:
        pass
    else:
        # prefer pages length if present; otherwise read PDF
        if data["pages"]:
            data["num_pages"] = len(data["pages"])
        else:
            n = _num_pages_from_pdf(pdf_path)
            data["num_pages"] = int(n) if n is not None else -1

    # segments (keep as-is; ensure list[str])
    segs = data.get("segments") or []
    norm_segs: List[str] = []
    for s in segs:
        if isinstance(s, str):
            norm_segs.append(s)
        elif isinstance(s, dict):
            # try common keys
            text = s.get("text") or s.get("content") or s.get("value")
            norm_segs.append(text if isinstance(text, str) else "")
        else:
            norm_segs.append("")
    data["segments"] = norm_segs

    # segment2page
    s2p = data.get("segment2page")
    valid_s2p = isinstance(s2p, list) and len(s2p) == len(norm_segs) and all(
        (isinstance(x, int) or x is None) for x in s2p
    )

    if valid_s2p:
        data["segment2page"] = s2p
    else:
        # Try to infer if pages present
        if data["pages"]:
            data["segment2page"] = _infer_segment2page_from_pages(norm_segs, data["pages"])
        else:
            data["segment2page"] = [None] * len(norm_segs)

    return data


def main():
    parser = argparse.ArgumentParser(description="Run OCR engines over PDFs and produce enriched JSON.")
    parser.add_argument("--engine",
                        required=True,
                        choices=["tesseract", "easyocr", "paddleocr", "smoldocling"],
                        help="Which OCR engine to use (must exist in engines/<name>.py)")
    parser.add_argument("--pdf-dir", default=PDF_DIR_DEFAULT, help="Directory with input PDFs")
    parser.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT, help="Base output directory")
    parser.add_argument("--overwrite", action="store_true", help="Re-process even if the output JSON exists")
    args = parser.parse_args()

    engine_mod = get_engine(args.engine)

    pdf_dir = Path(args.pdf_dir)
    out_base = Path(args.output_dir)
    out_engine_dir = out_base / args.engine
    out_engine_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    print(f"🗂  Found {len(pdfs)} PDF(s) in {pdf_dir}")

    done = 0
    for pdf_path in pdfs:
        doc_id = pdf_path.stem
        out_path = out_engine_dir / f"{doc_id}.json"

        if out_path.exists() and not args.overwrite:
            print(f"↪️  Skip {doc_id} (exists). Use --overwrite to re-run.")
            done += 1
            continue

        print(f"🔍 {args.engine}: {doc_id}")
        t0 = time.time()
        try:
            raw = engine_mod.ocr(pdf_path)
            # ensure enrichment
            enriched = _ensure_enriched_fields(raw, pdf_path)
            # add/overwrite processing_time (engine may set its own; we keep the outer wall time)
            enriched["processing_time"] = round(time.time() - t0, 2)
            # ensure engine tag
            enriched.setdefault("engine", args.engine)

            out_path.write_text(engine_mod.to_json(enriched), encoding="utf-8")
            print(f"✅ wrote {out_path.name}  "
                  f"(pages={enriched.get('num_pages')}, segs={len(enriched.get('segments', []))}, "
                  f"size={enriched.get('filesize_bytes')} bytes, {enriched['processing_time']}s)")
            done += 1
        except Exception as e:
            print(f"❌ Error {doc_id}: {e}", file=sys.stderr)

    print(f"🏁 Finished {done}/{len(pdfs)} file(s). Output: {out_engine_dir}")


if __name__ == "__main__":
    main()