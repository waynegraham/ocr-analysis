import os
import json
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple

import requests
import streamlit as st
from sentence_transformers import SentenceTransformer
import pypdfium2 as pdfium
from PIL import Image

# ===================== Config =====================
SOLR_URL       = os.getenv("SOLR_URL", "http://localhost:8983/solr")
SOLR_CORE      = os.getenv("SOLR_CORE", "trustees")
VECTOR_FIELD   = os.getenv("VECTOR_FIELD", "vector")
MODEL_NAME     = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
RETURN_FIELDS = "id,score,text,doc_id,segment_index,meeting_date,ocr_source,title,page_index"

# Where PDFs live and where OCR JSONs live (by engine)
PDF_DIR        = Path(os.getenv("PDF_DIR", "data/pdfs"))
OCR_OUTPUT_DIR = Path(os.getenv("OCR_OUTPUT_DIR", "ocr_output"))

# Cache thumbnails here
THUMB_DIR      = Path(os.getenv("THUMB_DIR", ".thumb_cache"))
THUMB_DIR.mkdir(parents=True, exist_ok=True)

# ===================== Model ======================
@st.cache_resource(show_spinner=False)
def load_model(name: str):
    return SentenceTransformer(name)

embedder = load_model(MODEL_NAME)


# ===================== Solr =======================
def solr_knn(query: str, topk: int = 10, filters: Optional[List[str]] = None):
    vec = embedder.encode([query], normalize_embeddings=True)[0]
    qvec = [float(x) for x in vec]

    url = f"{SOLR_URL.rstrip('/')}/{SOLR_CORE}/select"
    headers = {'Content-type': 'application/json'}

    query_vector_str = "[" + ",".join(map(str, vec)) + "]"

    #  KNN for broad compatibility (uses 'v=' with a CSV vector)
    payload = {
        "query": f"{{!knn f={VECTOR_FIELD} topK={topk}}}[{','.join(map(str, qvec))}]",
        "limit": topk,
        "fields": RETURN_FIELDS
    }

    # data_form = list(params.items())
    if filters:
        payload["filter"] = filters

    # r = requests.post(url, data=data_form, timeout=30)
    r = requests.post(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        timeout=30
    )
    r.raise_for_status()
    data = r.json()
    return data.get("response", {}).get("docs", []), data


# ================= Page inference & thumbnails =================
def _load_ocr_pages_for_doc(doc_id: str, ocr_source: str) -> List[str]:
    """
    Load per-page OCR text from your OCR JSON (if available).
    Supports shapes:
      pages: ["text", ...] or pages: [{"text": "..."}, ...]
    Returns list[str] (one per page). If not found, returns [].
    """
    json_path = OCR_OUTPUT_DIR / ocr_source / f"{doc_id}.json"
    if not json_path.exists():
        return []
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    pages = data.get("pages")
    if not pages:
        return []

    out = []
    for p in pages:
        if isinstance(p, str):
            out.append(p)
        elif isinstance(p, dict):
            t = p.get("text") or p.get("content") or ""
            out.append(t)
        else:
            out.append("")
    return out

def infer_page_index(doc_id: str, ocr_source: str, segment_text: str) -> Optional[int]:
    """
    Try to locate which page contains the segment_text by searching OCR 'pages'.
    Returns 0-based page index, or None if not found.
    """
    pages = _load_ocr_pages_for_doc(doc_id, ocr_source)
    if not pages:
        return None
    # Simple case-insensitive containment check with whitespace normalization
    needle = " ".join(segment_text.split()).lower()
    if not needle:
        return None

    # Require a minimal substring (avoid tiny words); use first 40 chars if long
    probe = needle if len(needle) <= 120 else needle[:120]
    for i, page_txt in enumerate(pages):
        hay = " ".join((page_txt or "").split()).lower()
        if probe and probe in hay:
            return i
    return None

def pdf_path_for_doc(doc_id: str) -> Path:
    return PDF_DIR / f"{doc_id}.pdf"

def render_page_thumbnail(doc_id: str, page_index: int, scale: float = 1.5) -> Optional[Image.Image]:
    """
    Render a single PDF page to a PIL image and cache it.
    scale ~1.5 → about 108 dpi; increase for sharper thumbnails.
    """
    pdf_path = pdf_path_for_doc(doc_id)
    if not pdf_path.exists():
        return None

    # Cache key
    key = hashlib.md5(f"{pdf_path}-{page_index}-{scale}".encode("utf-8")).hexdigest()
    thumb_path = THUMB_DIR / f"{doc_id}_p{page_index+1}_{key}.png"
    if thumb_path.exists():
        try:
            return Image.open(thumb_path)
        except Exception:
            pass

    try:
        doc = pdfium.PdfDocument(str(pdf_path))
        n_pages = len(doc)
        if page_index < 0 or page_index >= n_pages:
            return None
        page = doc.get_page(page_index)
        # 72 dpi base; scale multiplies
        pil_image = page.render(scale=scale).to_pil()
        page.close()
        doc.close()
        pil_image.save(thumb_path)
        return pil_image
    except Exception:
        return None

# ===================== UI =========================
st.set_page_config(layout="wide", page_title="Ground Truth Annotator (with thumbnails)")
st.title("📝 Ground Truth Annotator (with PDF thumbnails)")

with st.expander("ℹ️ Instructions", expanded=True):
    st.markdown("""
    **What is this?**
    
    This annotator helps build a **ground‑truth dataset** for evaluating retrieval (Solr kNN, BM25, hybrid) and RAG.  
    You’ll review candidate snippets for each question, see the **PDF page thumbnail**, and select the best evidence.
                
    ### Why it matters
    - Creates reliable **gold labels** for metrics like Precision@k, Recall@k, MRR, nDCG.
    - Lets us compare OCR engines (Tesseract/EasyOCR/Paddle/SmolDocling), segmentations, and embedding models.
    - Enables RAG evaluation (does the answer cite the correct meeting and quote?).

    ### What you’ll do
                
    1. **Load questions** (`questions.jsonl`). Each line:
    ```json
    {"qid":"Q0001", "query":"When did the Board discuss tuition increases?", "notes":"(optional) context"}
    ```
    2.	The tool **embeds the query** and runs **Solr kNN** search to fetch likely passages.
    3.	For each candidate, you’ll see:
        - **the segment text**,
        - metadata (date, engine, doc/segment ID, score),
        - a **PDF page thumbnail** (auto‑inferred; may default to page 1 if not found)
    4.	Click **“Select this segment”** to mark it as a positive evidence for the question.
                
    ### What gets saved?
                
    Each click appends one JSON line to the file you choose (default: `annotations/ground_truth.jsonl`):
    ```json
    {
      "qid": "Q0001",
        "query": "When did the Board discuss tuition increases?",
        "answers": [{
            "doc_id": "SC1010_s01_ss01_b01_f01_1907-05-31",
            "segment_index": 42,
            "meeting_date": "1907-05-31",
            "evidence_text": "…segment text…",
            "verdict": "positive"
        }],
        "notes": "optional notes"
    }
    ```    
    ### Annotation guidelines

    - **Be strict**: Choose passages that *directly answer* the question.
    - **Earliest mention** tasks: pick the **earliest dated** segment that clearly mentions the topic.
    - **Quotes** tasks: ensure the passage contains the phrase (case/punct normalized OK).
    - **Unsure?** Add a note. It’s better to flag ambiguity than guess.
    - Mark only **one** "positve" per question for now (we can support multi-gold later).
                
    ### Filters & settings
    - **Top‑K**: number of candidates to review per query (sidebar).
    - **Engine filter**: limit to Tesseract/EasyOCR/Paddle/SmolDocling for targeted comparison.
    - **Ouput path**: where annotations are written (sidebar).
                
    #### Folder layout the app expects
    ```bash
    PDF_DIR/                   # e.g., data/pdfs
    <doc_id>.pdf
    OCR_OUTPUT_DIR/
    <engine>/                  # e.g., ocr_output/easyocr
        <doc_id>.json          # may include {"pages":[...]} for accurate thumbnails
    ```    
                
    ## Troubleshooting
    - **No candidates**: check Solr is running and that VECTOR_FIELD and core env vars are set.
    - **400 from Solr**: schema mismatch or handler syntax—try without filters; verify vector dimension.
    - **Wrong/blank thumbnail**: engine JSON may not contain pages; the tool falls back to page 1.
                
    **Env vars** (can be set before `streamlit run`):

    ```bash
    SOLR_URL=http://localhost:8983/solr
    SOLR_CORE=trustees
    VECTOR_FIELD=vector
    MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
    PDF_DIR=data/pdfs
    OCR_OUTPUT_DIR=ocr_output
    THUMB_DIR=.thumb_cache
    ```
                
    """)

st.sidebar.header("Settings")
topk = st.sidebar.slider("Top‑K candidates", 3, 25, 10)
filters = []
engine_filter = st.sidebar.multiselect("Filter by OCR engine", ["tesseract", "easyocr", "paddleocr", "smoldocling"])
if engine_filter:
    filters.append("ocr_source:(" + " OR ".join(engine_filter) + ")")

out_path = st.sidebar.text_input("Save annotations to", "annotations/ground_truth.jsonl")
Path(out_path).parent.mkdir(parents=True, exist_ok=True)

st.sidebar.caption(f"PDF_DIR: {PDF_DIR}")
st.sidebar.caption(f"OCR_OUTPUT_DIR: {OCR_OUTPUT_DIR}")

st.sidebar.markdown("—")
st.sidebar.markdown("**About**")
st.sidebar.caption(
    "This tool builds gold labels for evaluating Solr kNN/BM25/Hybrid and RAG. "
    "Use filters to focus on specific OCR engines. Annotations append to a JSONL file."
)

# Questions input
uploaded = st.file_uploader("Upload questions.jsonl", type=["jsonl"])
if uploaded:
    questions = [json.loads(l) for l in uploaded]
else:
    st.info("Upload a questions.jsonl file (each line: {\"qid\":..., \"query\":...}) to start.")
    st.stop()

if not questions:
    st.warning("No questions found in the file.")
    st.stop()

qid_idx = st.number_input("Question index", 0, len(questions)-1, 0, step=1)
q = questions[qid_idx]
st.subheader(f"Q{qid_idx+1}: {q.get('query','')}")
notes = st.text_area("Notes", value=q.get("notes",""))

with st.spinner("Searching…"):
    # candidates = solr_knn(q.get("query", ""), topk=topk, filters=filters)
    candidates, raw = solr_knn(q.get("query", ""), topk=topk, filters=filters)

if not candidates:
    st.warning("No candidates found.")
    st.stop()

# Show candidates with thumbnails
chosen_idx = None
for i, d in enumerate(candidates):
    seg_text = d.get("text", "")
    doc_id   = d.get("doc_id", "")
    engine   = d.get("ocr_source", "")
    date     = d.get("meeting_date", "")
    seg_idx  = d.get("segment_index", "")

    page_idx = d.get("page_index", None)
    if page_idx is None:
        # graceful fallback for older docs that lack mapping
        page_idx = infer_page_index(doc_id, engine, seg_text) or 0

    # Render thumbnail
    thumb = render_page_thumbnail(doc_id, page_idx, scale=1.6)

    with st.container(border=True):
        cols = st.columns([0.32, 0.68])
        with cols[0]:
            if thumb is not None:
                st.image(thumb, caption=f"{doc_id}.pdf • p.{page_idx+1}", use_container_width=True)
            else:
                st.write("🖼️ (no thumbnail)")
                st.caption(f"{doc_id}.pdf • p.{page_idx+1}")
        with cols[1]:
            st.markdown(f"**{d.get('title') or doc_id}**")
            st.caption(f"Date: `{date}` • Engine: `{engine}` • Segment: `{seg_idx}` • Score: {d.get('score',0):.4f}")
            st.write(seg_text)
            if st.button(f"✅ Select this segment (#{i+1})", key=f"pick_{i}"):
                chosen_idx = i

if chosen_idx is not None:
    d = candidates[chosen_idx]
    ann = {
        "qid": q.get("qid", f"Q{qid_idx+1:04d}"),
        "query": q.get("query",""),
        "answers": [{
            "doc_id": d.get("doc_id",""),
            "segment_index": d.get("segment_index"),
            "meeting_date": d.get("meeting_date",""),
            "evidence_text": d.get("text",""),
            "verdict": "positive"
        }],
        "notes": notes
    }
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(ann, ensure_ascii=False) + "\n")
    st.success(f"Saved annotation to {out_path}")