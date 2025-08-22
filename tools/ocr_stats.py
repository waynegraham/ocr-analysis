#!/usr/bin/env python3
"""
Aggregate OCR outputs and generate summary CSVs + charts.

Assumes OCR JSON files are stored like:
  ocr_output/
    tesseract/<doc_id>.json
    easyocr/<doc_id>.json
    paddleocr/<doc_id>.json
    smoldocling/<doc_id>.json

Each JSON (as produced by your updated engines/runner) includes:
  engine, filesize_bytes, num_pages, pages:[{text}], segments:[...],
  segment2page:[...], avg_confidence (may be null), engine_time, processing_time

Outputs:
  reports/ocr_docs.csv
  reports/ocr_engine_summary.csv
  reports/fig_*.png   (charts)
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import math
import statistics as stats

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_ocr_docs(base_dir: Path) -> pd.DataFrame:
    """
    Walk ocr_output/*/*.json and load per-document records into a dataframe.
    """
    rows: List[Dict[str, Any]] = []
    for engine_dir in sorted((base_dir).glob("*")):
        if not engine_dir.is_dir():
            continue
        engine = engine_dir.name
        for jf in sorted(engine_dir.glob("*.json")):
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"⚠️  Skipping {jf.name}: {e}")
                continue

            doc_id = jf.stem
            num_pages = data.get("num_pages")
            pages = data.get("pages") or []
            segments = data.get("segments") or []
            seg2page = data.get("segment2page") or []
            filesize = data.get("filesize_bytes")
            avg_conf = data.get("avg_confidence", None)
            engine_time = data.get("engine_time", None)
            processing_time = data.get("processing_time", None)

            # Derived
            num_segments = len(segments)
            segs_per_page = (num_segments / num_pages) if (isinstance(num_pages, int) and num_pages > 0) else None
            pages_missing = None
            if isinstance(num_pages, int):
                pages_missing = max(0, num_pages - len(pages))

            rows.append({
                "doc_id": doc_id,
                "engine": engine,
                "filesize_bytes": filesize,
                "num_pages": num_pages,
                "num_pages_list_entries": len(pages),
                "pages_missing_entries": pages_missing,
                "num_segments": num_segments,
                "avg_confidence": avg_conf,
                "engine_time_s": engine_time,
                "processing_time_s": processing_time,
                "segments_per_page": segs_per_page,
                "has_seg2page": int(bool(seg2page and len(seg2page) == num_segments)),
                "bytes_per_page": (filesize / num_pages) if (isinstance(filesize, (int, float)) and isinstance(num_pages, int) and num_pages > 0) else None,
                "pages_per_second_engine": (num_pages / engine_time) if (isinstance(num_pages, int) and num_pages > 0 and isinstance(engine_time, (int, float)) and engine_time > 0) else None,
                "pages_per_second_wall": (num_pages / processing_time) if (isinstance(num_pages, int) and num_pages > 0 and isinstance(processing_time, (int, float)) and processing_time > 0) else None,
            })

    df = pd.DataFrame(rows)
    return df


def summarize_by_engine(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groupwise summary with mean/median and simple counts.
    """
    if df.empty:
        return pd.DataFrame()

    agg = df.groupby("engine").agg(
        docs=("doc_id", "count"),
        pages=("num_pages", "sum"),
        segments=("num_segments", "sum"),
        mean_pages=("num_pages", "mean"),
        median_pages=("num_pages", "median"),
        mean_segments=("num_segments", "mean"),
        median_segments=("num_segments", "median"),
        mean_avg_conf=("avg_confidence", "mean"),
        median_avg_conf=("avg_confidence", "median"),
        mean_proc_s=("processing_time_s", "mean"),
        p95_proc_s=("processing_time_s", lambda x: np.nan if len(x)==0 else float(np.nanpercentile(x, 95))),
        mean_engine_s=("engine_time_s", "mean"),
        p95_engine_s=("engine_time_s", lambda x: np.nan if len(x)==0 else float(np.nanpercentile(x, 95))),
        mean_pps_engine=("pages_per_second_engine", "mean"),
        mean_pps_wall=("pages_per_second_wall", "mean"),
        mean_segs_per_page=("segments_per_page", "mean"),
        pct_with_seg2page=("has_seg2page", "mean"),  # 0-1
        mean_bytes_per_page=("bytes_per_page", "mean"),
    ).reset_index()

    # prettier %
    agg["pct_with_seg2page"] = (agg["pct_with_seg2page"] * 100.0).round(1)
    return agg


def save_csvs(df_docs: pd.DataFrame, df_eng: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    docs_csv = out_dir / "ocr_docs.csv"
    eng_csv = out_dir / "ocr_engine_summary.csv"
    df_docs.to_csv(docs_csv, index=False)
    df_eng.to_csv(eng_csv, index=False)
    print(f"📄 Wrote {docs_csv}")
    print(f"📄 Wrote {eng_csv}")


def style_axes(ax, title: str, xlabel: str, ylabel: str):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)


def plot_scatter_conf_vs_time(df: pd.DataFrame, out_dir: Path):
    plt.figure()
    ax = plt.gca()
    for engine, sub in df.groupby("engine"):
        ax.scatter(sub["processing_time_s"], sub["avg_confidence"], label=engine, alpha=0.7)
    style_axes(ax, "Avg OCR confidence vs wall-clock time", "Processing time (s)", "Avg confidence")
    ax.legend()
    p = out_dir / "fig_confidence_vs_time.png"
    plt.tight_layout(); plt.savefig(p, dpi=150); plt.close()
    print(f"🖼  {p}")


def plot_box_time_by_engine(df: pd.DataFrame, out_dir: Path):
    plt.figure()
    ax = plt.gca()
    data = [sub["processing_time_s"].dropna() for _, sub in df.groupby("engine")]
    labels = [eng for eng, _ in df.groupby("engine")]
    ax.boxplot(data, labels=labels, showfliers=False)
    style_axes(ax, "Processing time by engine", "Engine", "Processing time (s)")
    p = out_dir / "fig_time_box_by_engine.png"
    plt.tight_layout(); plt.savefig(p, dpi=150); plt.close()
    print(f"🖼  {p}")


def plot_box_conf_by_engine(df: pd.DataFrame, out_dir: Path):
    plt.figure()
    ax = plt.gca()
    data = [sub["avg_confidence"].dropna() for _, sub in df.groupby("engine")]
    labels = [eng for eng, _ in df.groupby("engine")]
    ax.boxplot(data, labels=labels, showfliers=False)
    style_axes(ax, "Average confidence by engine", "Engine", "Avg confidence")
    p = out_dir / "fig_conf_box_by_engine.png"
    plt.tight_layout(); plt.savefig(p, dpi=150); plt.close()
    print(f"🖼  {p}")


def plot_hist_pages(df: pd.DataFrame, out_dir: Path):
    plt.figure()
    ax = plt.gca()
    max_pages = int(df["num_pages"].dropna().max()) if not df["num_pages"].dropna().empty else 0
    bins = min(max(10, max_pages), 60)
    ax.hist(df["num_pages"].dropna(), bins=bins)
    style_axes(ax, "Histogram of pages per document (all engines combined)", "Pages per document", "Count")
    p = out_dir / "fig_pages_hist.png"
    plt.tight_layout(); plt.savefig(p, dpi=150); plt.close()
    print(f"🖼  {p}")


def plot_bar_engine_throughput(df: pd.DataFrame, out_dir: Path):
    # mean pages/sec (wall) by engine
    agg = df.groupby("engine")["pages_per_second_wall"].mean().reset_index()
    plt.figure()
    ax = plt.gca()
    ax.bar(agg["engine"], agg["pages_per_second_wall"])
    style_axes(ax, "Throughput (pages/sec, wall-clock) by engine", "Engine", "Pages/sec")
    p = out_dir / "fig_throughput_bar.png"
    plt.tight_layout(); plt.savefig(p, dpi=150); plt.close()
    print(f"🖼  {p}")


def make_plots(df_docs: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Filters for plots that need both axes present
    df_ct = df_docs[(df_docs["processing_time_s"].notna()) & (df_docs["avg_confidence"].notna())]
    if not df_ct.empty:
        plot_scatter_conf_vs_time(df_ct, out_dir)
    else:
        print("ℹ️  Skip confidence vs time plot (missing fields).")

    if df_docs["processing_time_s"].notna().sum() > 0:
        plot_box_time_by_engine(df_docs, out_dir)
    else:
        print("ℹ️  Skip time boxplot (no processing_time_s).")

    if df_docs["avg_confidence"].notna().sum() > 0:
        plot_box_conf_by_engine(df_docs, out_dir)
    else:
        print("ℹ️  Skip confidence boxplot (no avg_confidence).")

    if df_docs["num_pages"].notna().sum() > 0:
        plot_hist_pages(df_docs, out_dir)
    else:
        print("ℹ️  Skip pages histogram (no num_pages).")

    if df_docs["pages_per_second_wall"].notna().sum() > 0:
        plot_bar_engine_throughput(df_docs, out_dir)
    else:
        print("ℹ️  Skip throughput bar (no pages_per_second_wall).")


def main():
    ap = argparse.ArgumentParser(description="Generate OCR stats and charts.")
    ap.add_argument("--ocr-output", default="data/ocr_output", help="Base dir containing per-engine OCR JSONs")
    ap.add_argument("--reports", default="reports", help="Output directory for CSVs and PNGs")
    args = ap.parse_args()

    base = Path(args.ocr_output).resolve()
    out_dir = Path(args.reports).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 Reading OCR outputs from {base}")
    df_docs = load_ocr_docs(base)
    if df_docs.empty:
        print("❌ No OCR JSON files found.")
        return

    # Coerce numeric columns
    num_cols = [
        "filesize_bytes","num_pages","num_pages_list_entries","pages_missing_entries",
        "num_segments","avg_confidence","engine_time_s","processing_time_s",
        "segments_per_page","has_seg2page","bytes_per_page",
        "pages_per_second_engine","pages_per_second_wall",
    ]
    for c in num_cols:
        if c in df_docs.columns:
            df_docs[c] = pd.to_numeric(df_docs[c], errors="coerce")

    # Save per-doc table
    df_docs = df_docs.sort_values(["engine","doc_id"])
    eng_summary = summarize_by_engine(df_docs)

    save_csvs(df_docs, eng_summary, out_dir)
    make_plots(df_docs, out_dir)

    # Quick console summary
    print("\n=== Engine Summary ===")
    if not eng_summary.empty:
        print(eng_summary.to_string(index=False))
    else:
        print("No summary available (empty input).")


if __name__ == "__main__":
    main()