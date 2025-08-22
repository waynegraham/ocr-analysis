#!/usr/bin/env python3
"""
Index OCR output JSONs into Solr.

Usage:
  python tools/index_to_solr.py \
    --engine tesseract \
    --ocr-dir data/ocr_output \
    --solr http://localhost:8983/solr/trustees \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --vector-field vector \
    --commit
"""

import argparse, json, sys, time, os
from pathlib import Path
from typing import List, Dict
import requests

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


def load_ocr_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"❌ Failed to load {path}: {e}", file=sys.stderr)
        return None

def _coerce_text(item):
    """
    Accepts either a string or an object with a 'text' field and returns text.
    Returns '' when nothing can be extracted.
    """
    if item is None:
        return ""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("text", "segment", "content", "value"):
            v = item.get(key)
            if isinstance(v, str):
                return v
    return ""

def build_solr_docs_for_file(json_data, doc_id: str, engine: str,
                             embedder=None, vector_field=None, batch_size=64):
    """Turn OCR JSON into Solr docs with optional embeddings."""
    docs = []
    segments_str: list[str] = []

    meta = {
        "doc_id": doc_id,
        "ocr_source": engine,
        "avg_confidence": json_data.get("avg_confidence"),
        "ocr_processing_time": json_data.get("processing_time"),
        "num_pages": json_data.get("num_pages"),
        "filesize_bytes": json_data.get("filesize_bytes"),
    }

    # Prefer explicit segments
    items = json_data.get("segments")
    if items is None:
        # fallback: build from pages list
        items = [p.get("text","") if isinstance(p, dict) else (p or "") for p in json_data.get("pages", [])]

    # Normalize to strings
    normalized = [_coerce_text(it).strip() for it in items]
    normalized = [t for t in normalized if t]

    # Optional page mapping
    seg2page = json_data.get("segment2page")
    if not isinstance(seg2page, list) or len(seg2page) != len(normalized):
        seg2page = [None] * len(normalized)

    for idx, (text, pidx) in enumerate(zip(normalized, seg2page)):
        doc = {
            "id": f"{doc_id}#{idx}",
            "text": text,
            "segment_index": idx,
            "page_index": pidx if isinstance(pidx, int) else None,
            **meta,
        }
        docs.append(doc)
        segments_str.append(text)

    # Embeddings (unchanged)
    if embedder and docs:
        vectors = embedder.encode(segments_str, normalize_embeddings=True, batch_size=batch_size)
        for doc, vec in zip(docs, vectors):
            doc[vector_field] = vec.tolist()

    return docs


def post_solr_docs(solr_update_url: str, docs: List[Dict], commit: bool, retries: int = 3):
    payload = json.dumps(docs, ensure_ascii=False)
    params = {"commit": "true"} if commit else {}
    headers = {"Content-Type": "application/json"}
    last_err = None
    last_text = None

    for attempt in range(1, retries + 1):
        try:
            r = requests.post(
                solr_update_url, params=params,
                data=payload.encode("utf-8"),
                headers=headers, timeout=60
            )
            if r.status_code >= 400:
                last_err = f"{r.status_code} {r.reason}"
                last_text = r.text
                raise requests.HTTPError(last_err)
            return
        except Exception as e:
            last_err = e
            time.sleep(1.5 * attempt)

    raise RuntimeError(
        f"Failed posting to Solr after {retries} attempts: {last_err}\n"
        f"----- Solr said -----\n{last_text}\n---------------------"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, help="OCR engine name (tesseract, easyocr, etc.)")
    ap.add_argument("--ocr-dir", default="data/ocr_output", help="Base OCR output dir")
    ap.add_argument("--solr", required=True, help="Solr core URL, e.g. http://localhost:8983/solr/trustees")
    ap.add_argument("--commit", action="store_true", help="Commit after indexing")
    ap.add_argument("--model", help="Sentence-transformers model to embed segments")
    ap.add_argument("--vector-field", default="vector", help="Solr vector field name")
    ap.add_argument("--embed-batch-size", type=int, default=64, help="Batch size for embedding encode()")
    ap.add_argument("--source-collection", default="stanford_trustees", help="Constant collection field")
    args = ap.parse_args()

    ocr_path = Path(args.ocr_dir) / args.engine
    if not ocr_path.exists():
        print(f"❌ No OCR output dir {ocr_path}", file=sys.stderr)
        sys.exit(1)

    # Load embedder if needed
    embedder = None
    if args.model:
        if SentenceTransformer is None:
            print("❌ sentence-transformers not installed", file=sys.stderr)
            sys.exit(1)
        device = "cuda" if (hasattr(__import__("torch"), "cuda") and __import__("torch").cuda.is_available()) else "cpu"
        print(f"🔧 Loading embedding model on {device} → {args.model}")
        embedder = SentenceTransformer(args.model, device=device)
        try:
            dim = embedder.get_sentence_embedding_dimension()
        except Exception:
            dim = len(embedder.encode(["test"], normalize_embeddings=True)[0])
        print(f"ℹ️ Embedding dimension: {dim} — match Solr schema vectorDimension")

    # Iterate files
    buffer = []
    update_url = args.solr.rstrip("/") + "/update"
    for json_file in sorted(ocr_path.glob("*.json")):
        doc_id = json_file.stem
        data = load_ocr_json(json_file)
        if not data:
            continue
        docs = build_solr_docs_for_file(
            data, doc_id, args.engine,
            embedder=embedder, vector_field=args.vector_field,
            batch_size=args.embed_batch_size
        )
        # add source_collection constant
        for d in docs:
            d["source_collection"] = args.source_collection
        buffer.extend(docs)

        # flush in chunks of 100
        if len(buffer) >= 100:
            print(f"⚡ Posting {len(buffer)} docs...")
            post_solr_docs(update_url, buffer, commit=args.commit)
            buffer = []

    if buffer:
        print(f"⚡ Posting final {len(buffer)} docs...")
        post_solr_docs(update_url, buffer, commit=args.commit)

    print("✅ Done.")


if __name__ == "__main__":
    main()