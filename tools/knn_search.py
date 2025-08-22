from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import argparse, csv, json, os, sys, time

import requests

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

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


def solr_knn(solr_base, core, qvec, fl, rows, fqs=None, start=0):
    fl = fl or "id,score"
    fqs = fqs or []

    url = f"{solr_base.rstrip('/')}/{core}/select"
    headers = {"Content-Type": "application/json"}

    query_vector_str = "[" + ",".join(map(str, qvec)) + "]"

    payload = {
        "query": f"{{!knn f=vector topK={rows}}}{query_vector_str}",
        "limit": rows,
        "fields": fl
    }

    if fqs:
        payload["filter"] = list(fqs)

    t0 = time.perf_counter()
    r = requests.post(url, data=json.dumps(payload).encode("utf-8"),
                      headers=headers, timeout=30)
    lat = (time.perf_counter() - t0) * 1000.0
    r.raise_for_status()
    data = r.json()
    return data.get("response", {}).get("docs", []), lat


def post_form(url: str, params: List[Tuple[str, str]], timeout: int = 30) -> Tuple[Dict, float]:
    t0 = time.perf_counter()
    r = requests.post(url, data=params, timeout=timeout)
    lat = (time.perf_counter() - t0) * 1000.0
    if r.status_code >= 400:
        raise RuntimeError(f"{r.status_code} {r.reason}: {r.text[:800]}")
    return r.json(), lat


def solr_hybrid(solr_base, core, qtext, qvec, fl, topk, fqs=None, start=0):
    url = f"{solr_base.rstrip('/')}/{core}/select"
    # This legacy format for hybrid is not standard.
    # A common way is Reciprocal Rank Fusion (RRF) via the /rerank handler,
    # but for a simpler demo, we'll stick to the rq/v params.
    # NOTE: This requires Solr config to map 'v' to the vector query parser.

    print(f"  [solr_hybrid] q='{qtext}' topk={topk} fqs={fqs}")

    params = [
        ("q", qtext or "*:*"),
        ("df", "text"),
        ("defType", "edismax"),
        ("df", "text"),
        # ("rq", f"{{!knn f=vector topK={topk}}}[{','.join(map(str, vec))}]"),
        ("fl", fl or "id,score"),
        ("rows", str(topk)),
        ("start", str(start)),
    ]
    if fqs:
        for f in fqs:
            params.append(("fq", f))
    data, lat = post_form(url, params)
    return data.get("response", {}).get("docs", []), lat


def main():
    solr_base = "http://localhost:8983/solr"
    core = "trustees"
    gt_path = Path("./annotations/ground_truth.jsonl")
    out_path= Path("./runs/eval_knn_test.csv").resolve()

    gt = load_ground_truth(gt_path)
    print(f"  loaded {len(gt)} ground-truth rows")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    fl = "id,score,doc_id,segment_index,text,meeting_date,ocr_source"
    rows: List[Dict] = []

    for qi, qobj in enumerate(gt, 1):
        qid = qobj.get("qid", f"Q{qi:04d}")
        qtext = qobj.get("query", "")
        gold_seg_ids = set(gold_ids(qobj))
        gold_doc_ids = set(a.get("doc_id") for a in qobj.get("answers", []) if a.get("doc_id"))

        print(f"\n→ [{qi}/{len(gt)}] {qid}: “{qtext}”  gold segs={len(gold_seg_ids)} docs={len(gold_doc_ids)}")

        qvec = embedder.encode([qtext], normalize_embeddings=True)[0].tolist()

        docs, lat = solr_hybrid(solr_base, core, qvec, fl, 10, None)
        p, r, mrr, ndcg, rels = judge_results(docs, gold_seg_ids, gold_doc_ids, 10)

        rows.append({
            "qid": qid, "mode": "knn", "k": 10,
            "p_at_k": p,
            "r_at_k": r,
            "mrr_at_k": mrr,
            "ndcg_at_k": ndcg,
            "latency_ms": lat,
            "hits": sum(rels),
        })

        print(f"   [hybrid] hits={sum(rels)} p@{10}={p:.3f} mrr={mrr:.3f} ndcg={ndcg:.3f} lat={lat:.1f}ms")

        fieldnames = ["qid", "mode", "k", "p_at_k", "r_at_k", "mrr_at_k", "ndcg_at_k", "latency_ms", "hits", "error"]
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                if "error" not in r: r["error"] = ""
                w.writerow(r)
        

if __name__ == "__main__":
    main()