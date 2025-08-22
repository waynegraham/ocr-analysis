#!/usr/bin/env python3
import argparse, csv, json, os, sys, time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import requests

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


# ---------------- Metrics ----------------
def precision_at_k(rels: List[int], k: int) -> float:
    rel_k = rels[:k]
    return sum(rel_k) / max(k, 1)


def recall_at_k(rels: List[int], k: int, num_gold: int) -> float:
    if num_gold <= 0:
        return 0.0
    return sum(rels[:k]) / num_gold


def mrr_at_k(rels: List[int], k: int) -> float:
    for i, r in enumerate(rels[:k], start=1):
        if r:
            return 1.0 / i
    return 0.0


def ndcg_at_k(rels: List[int], k: int) -> float:
    def dcg(rs): return sum((1.0 / (i+1)) for i, r in enumerate(rs) if r)
    ideal_hits = min(sum(rels), k)
    idcg = sum((1.0 / (i+1)) for i in range(ideal_hits))
    if idcg == 0.0:
        return 0.0
    return dcg(rels[:k]) / idcg


# ------------- IO helpers -------------
def load_ground_truth(path: Path) -> List[Dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                items.append(json.loads(s))
            except Exception as e:
                print(f"⚠️ Skipping malformed line {ln}: {e}", file=sys.stderr)
    return items


def gold_ids(entry: Dict) -> List[str]:
    gold = []
    for ans in entry.get("answers", []):
        doc_id = ans.get("doc_id")
        seg = ans.get("segment_index")
        if doc_id is None:
            continue
        if seg is None:
            gold.append(doc_id)  # doc-level fallback
        else:
            gold.append(f"{doc_id}#{seg}")
    return gold


# ------------- Solr calls -------------
def post_form(url: str, params: List[Tuple[str, str]], timeout: int = 30) -> Tuple[Dict, float]:
    t0 = time.perf_counter()
    r = requests.post(url, data=params, timeout=timeout)
    lat = (time.perf_counter() - t0) * 1000.0
    if r.status_code >= 400:
        raise RuntimeError(f"{r.status_code} {r.reason}: {r.text[:800]}")
    return r.json(), lat


def solr_ping(solr_base: str, core: str):
    url = f"{solr_base.rstrip('/')}/{core}/admin/ping"
    r = requests.get(url, timeout=10)
    if r.status_code >= 400:
        raise RuntimeError(f"Solr ping failed: {r.status_code} {r.reason}: {r.text[:400]}")
    return r.json()


def solr_bm25(solr_base, core, qtext, fl, rows, fqs=None, start=0):
    url = f"{solr_base.rstrip('/')}/{core}/select"
    params = [
        ("q", qtext or "*:*"),
        ("defType", "edismax"),
        ("df", "text"),               # <-- add this (or use qf below)
        # ("qf", "text^1 title^2"),   # alternative: specify multiple fields with boosts
        ("fl", fl),
        ("rows", str(rows)),
        ("start", str(start)),
    ]
    if fqs:
        for f in fqs: params.append(("fq", f))
    data, lat = post_form(url, params)
    return data.get("response", {}).get("docs", []), lat


def solr_knn(solr_base, core, qtext, fl, rows, embedder, fqs=None, start=0):
    vec = embedder.encode([qtext], normalize_embeddings=True)[0]
    qvec = [float(x) for x in vec]
    fl = fl or "id,score"
    fqs = fqs or []

    url = f"{solr_base.rstrip('/')}/{core}/select"
    headers = {"Content-Type": "application/json"}

    query_vector_str = "[" + ",".join(map(str, qvec)) + "]"

    payload = {
        "query": f"{{!knn f=vector topK={rows}}}{query_vector_str}",
        "limit": rows,
        "fields": fl,
        "start": start,
    }
    if fqs:
        payload["filter"] = list(fqs)

    r = requests.post(url, data=json.dumps(payload).encode("utf-8"),
                      headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get("response", {}).get("docs", []), data


def solr_hybrid(solr_base, core, qtext, vec, fl, topk, fqs=None, start=0):
    url = f"{solr_base.rstrip('/')}/{core}/select"
    csv_vec = ",".join(str(float(x)) for x in vec)
    params = [
        ("q", qtext or "*:*"),
        ("defType", "edismax"),
        ("df", "text"),               # <-- add this (or use qf below)
        # ("qf", "text^1 title^2"),
        ("rq", "{!knn f=vector topK=%d}" % topk),
        ("v", csv_vec),
        ("fl", fl),
        ("rows", str(topk)),
        ("start", str(start)),
        ("mm", "1<75%"),
    ]
    if fqs:
        for f in fqs: params.append(("fq", f))
    data, lat = post_form(url, params)
    return data.get("response", {}).get("docs", []), lat


# ------------- Judging -------------
def judge_results(docs: List[Dict], gold_ids_set: set, gold_docids_set: set, k: int) -> Tuple[float, float, float, float, List[int]]:
    rels = []
    for d in docs[:k]:
        did = d.get("id", "")
        docid = d.get("doc_id", "")
        is_rel = (did in gold_ids_set) or (gold_docids_set and docid in gold_docids_set)
        rels.append(1 if is_rel else 0)
    num_gold = max(1, len(gold_ids_set) if gold_ids_set else len(gold_docids_set))
    p = precision_at_k(rels, k)
    r = recall_at_k(rels, k, num_gold)
    mrr = mrr_at_k(rels, k)
    nd = ndcg_at_k(rels, k)
    return p, r, mrr, nd, rels


# ------------- Main -------------
def main():
    ap = argparse.ArgumentParser(description="Evaluate Solr retrieval (BM25, kNN, Hybrid).")
    ap.add_argument("--ground-truth", required=True, help="Path to ground_truth.jsonl")
    ap.add_argument("--solr-url", default="http://localhost:8983/solr", help="Base Solr URL")
    ap.add_argument("--core", default="trustees", help="Solr core/collection")
    ap.add_argument("--mode", choices=["bm25", "knn", "hybrid", "all"], default="all")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--filters", action="append", default=[])
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--out-csv", default="runs/eval_results.csv")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out_csv).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.verbose:
        print(f"🧪 eval_retrieval starting")
        print(f"  ground_truth = {Path(args.ground_truth).resolve()}")
        print(f"  solr = {args.solr_url.rstrip('/')}/{args.core}")
        print(f"  mode = {args.mode}, k = {args.k}, filters = {args.filters}")
        print(f"  out_csv = {out_path}")

    # Sanity: ground truth
    gt_path = Path(args.ground_truth)
    if not gt_path.exists():
        print(f"❌ ground truth file not found: {gt_path.resolve()}", file=sys.stderr)
        sys.exit(1)
    gt = load_ground_truth(gt_path)
    if args.verbose:
        print(f"  loaded {len(gt)} ground-truth rows")

    if not gt:
        print("❌ ground-truth is empty. Add at least one question.", file=sys.stderr)
        sys.exit(1)

    # Sanity: Solr ping
    try:
        pong = solr_ping(args.solr_url, args.core)
        if args.verbose:
            print(f"  Solr ping: {pong.get('status','OK')}")
    except Exception as e:
        print(f"❌ Solr not reachable: {e}", file=sys.stderr)
        sys.exit(1)

    # Embedder if needed
    embedder = None
    if args.mode in ("knn", "hybrid", "all"):
        if SentenceTransformer is None:
            print("❌ sentence-transformers not installed; required for kNN/hybrid.", file=sys.stderr)
            sys.exit(1)
        embedder = SentenceTransformer(args.model)
        if args.verbose:
            try:
                dim = embedder.get_sentence_embedding_dimension()
            except Exception:
                dim = len(embedder.encode(["test"], normalize_embeddings=True)[0])
            print(f"  embedder = {args.model} (dim={dim})")

    modes = [args.mode] if args.mode != "all" else ["bm25", "knn", "hybrid"]
    fl = "id,score,doc_id,segment_index,text,meeting_date,ocr_source"

    rows: List[Dict] = []
    for qi, qobj in enumerate(gt, 1):
        qid = qobj.get("qid", f"Q{qi:04d}")
        qtext = qobj.get("query", "")
        gold_seg_ids = set(gold_ids(qobj))
        gold_doc_ids = set(a.get("doc_id") for a in qobj.get("answers", []) if a.get("doc_id"))

        if args.verbose:
            print(f"\n→ [{qi}/{len(gt)}] {qid}: “{qtext}”  gold segs={len(gold_seg_ids)} docs={len(gold_doc_ids)}")

        # prepare vector once
        qvec = None
        if embedder is not None:
            qvec = embedder.encode([qtext], normalize_embeddings=True)[0].tolist()

        for mode in modes:
            try:
                if mode == "bm25":
                    docs, lat = solr_bm25(args.solr_url, args.core, qtext, fl, args.k, fqs=args.filters)
                elif mode == "knn":
                    if qvec is None:
                        raise RuntimeError("kNN requires embeddings.")
                    docs, lat = solr_knn(args.solr_url, args.core, qvec, fl, args.k, embedder, fqs=args.filters)
                else:
                    if qvec is None:
                        raise RuntimeError("Hybrid requires embeddings.")
                    docs, lat = solr_hybrid(args.solr_url, args.core, qtext, qvec, fl, args.k, embedder, fqs=args.filters)

                p, r, mrr, ndcg, rels = judge_results(docs, gold_seg_ids, gold_doc_ids, args.k)
                if args.verbose:
                    print(f"   [{mode}] hits={sum(rels)} p@{args.k}={p:.3f} mrr={mrr:.3f} ndcg={ndcg:.3f} lat={lat:.1f}ms")

                rows.append({
                    "qid": qid, "mode": mode, "k": args.k,
                    "p_at_k": round(p, 4), "r_at_k": round(r, 4),
                    "mrr_at_k": round(mrr, 4), "ndcg_at_k": round(ndcg, 4),
                    "latency_ms": round(lat, 2), "hits": sum(rels),
                })
            except Exception as e:
                err = str(e)
                if args.verbose:
                    print(f"   [{mode}] ❌ {err}")
                rows.append({
                    "qid": qid, "mode": mode, "k": args.k,
                    "p_at_k": 0.0, "r_at_k": 0.0, "mrr_at_k": 0.0, "ndcg_at_k": 0.0,
                    "latency_ms": -1, "hits": 0, "error": err[:300],
                })

    # Per-mode aggregates
    modes_set = sorted(set(r["mode"] for r in rows))
    for m in modes_set:
        mode_rows = [r for r in rows if r["mode"] == m]
        ok_rows = [r for r in mode_rows if r.get("latency_ms", -1) >= 0]
        if not ok_rows:
            continue
        rows.append({
            "qid": "AGGREGATE", "mode": m, "k": args.k,
            "p_at_k": round(sum(r["p_at_k"] for r in ok_rows) / len(ok_rows), 4),
            "r_at_k": round(sum(r["r_at_k"] for r in ok_rows) / len(ok_rows), 4),
            "mrr_at_k": round(sum(r["mrr_at_k"] for r in ok_rows) / len(ok_rows), 4),
            "ndcg_at_k": round(sum(r["ndcg_at_k"] for r in ok_rows) / len(ok_rows), 4),
            "latency_ms": round(sum(r["latency_ms"] for r in ok_rows) / len(ok_rows), 2),
            "hits": sum(r["hits"] for r in ok_rows),
        })

    # Write CSV (use fixed field order so we can write even if some rows had errors)
    fieldnames = ["qid", "mode", "k", "p_at_k", "r_at_k", "mrr_at_k", "ndcg_at_k", "latency_ms", "hits", "error"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            if "error" not in r: r["error"] = ""
            w.writerow(r)

    print(f"\n✅ Wrote results: {out_path}")
    if args.verbose:
        print(f"   rows written: {len(rows)}")


if __name__ == "__main__":
    main()