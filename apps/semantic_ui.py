import os
import json
import requests
import streamlit as st

from sentence_transformers import SentenceTransformer

# ---------- Config ----------
SOLR_URL      = os.getenv("SOLR_URL", "http://localhost:8983/solr")
SOLR_CORE     = os.getenv("SOLR_CORE", "trustees")
VECTOR_FIELD  = os.getenv("VECTOR_FIELD", "vector")
MODEL_NAME    = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_TOP_K = int(os.getenv("TOP_K", "10"))

# Fields to return in results
RETURN_FIELDS = "id,score,text,title,meeting_date,ocr_source,segment_index,doc_id"

# ---------- Model ----------
@st.cache_resource(show_spinner=False)
def load_model(name: str):
    return SentenceTransformer(name)

embedder = load_model(MODEL_NAME)

# ---------- Solr ----------
def solr_knn_search(query_vec, k=10, filters=None):
    url = f"{SOLR_URL.rstrip('/')}/{SOLR_CORE}/select"
    headers = {"Content-Type": "application/json"}

    # Convert vector to Solr format string
    query_vector_str = "[" + ",".join(map(str, query_vec)) + "]"

    # ---- Attempt 1: JSON Request API (Solr 9.4+)
    payload = {
        "query": f"{{!knn f={VECTOR_FIELD} topK={k}}}{query_vector_str}",
        "limit": k,
        "fields": RETURN_FIELDS
    }
    if filters:
        payload["filter"] = filters

    try:
        r = requests.post(url, data=json.dumps(payload).encode("utf-8"),
                          headers=headers, timeout=30)
        if r.status_code >= 400:
            raise requests.HTTPError(f"{r.status_code} {r.reason}: {r.text}", response=r)
        data = r.json()
        return data.get("response", {}).get("docs", []), data
    except requests.HTTPError as e:
        st.warning(f"JSON KNN failed, trying legacy KNN. Solr said: {e}")

    # ---- Attempt 2: Legacy Param API ({!knn f=... topK=...}) expects 'v'
    # Use POST form-encoded to avoid giant URL
    params = {
        "q": f"{{!knn f={VECTOR_FIELD} topK={k}}}",
        "fl": RETURN_FIELDS,
        "rows": str(k),
        "v": ",".join(str(float(x)) for x in query_vec),  # <— IMPORTANT: 'v', not 'vector'
    }

    # Filters become repeated fq params
    data_form = []
    for key, val in params.items():
        data_form.append((key, val))
    if filters:
        for f in filters:
            data_form.append(("fq", f))

    r = requests.post(url, data=data_form, timeout=30)
    if r.status_code >= 400:
        raise requests.HTTPError(f"Legacy KNN also failed: {r.status_code} {r.reason}: {r.text}", response=r)
    data = r.json()
    return data.get("response", {}).get("docs", []), data

# ---------- UI ----------
st.set_page_config(page_title="Semantic Search (Solr + Embeddings)", layout="wide")

st.sidebar.title("Settings")
st.sidebar.markdown("Configure Solr & model via environment variables:")
st.sidebar.code(
    "SOLR_URL, SOLR_CORE, VECTOR_FIELD, MODEL_NAME\n"
    "e.g. export SOLR_URL=http://localhost:8983/solr"
)

st.title("🔎 Semantic Search — Stanford Trustees Minutes")
st.caption(f"Solr core: `{SOLR_CORE}`  |  Model: `{MODEL_NAME}`  |  Vector field: `{VECTOR_FIELD}`")

query = st.text_input("Enter a natural-language query", placeholder="e.g., When were campus expansion plans first discussed?")
top_k = st.slider("Top‑K results", min_value=1, max_value=50, value=DEFAULT_TOP_K)

# Optional filters
with st.expander("Filters"):
    engines = st.multiselect("OCR engine filter", ["tesseract", "easyocr", "paddleocr", "smoldocling"])
    date_start = st.text_input("Start date (YYYY-MM-DD)")
    date_end   = st.text_input("End date (YYYY-MM-DD)")

do_search = st.button("Search")

if do_search and query.strip():
    with st.spinner("Embedding query and searching..."):
        # Embed query
        qvec = embedder.encode([query], normalize_embeddings=True)[0]

        # Build filters
        filters = []
        if engines:
            # ocr_source:(eng1 OR eng2 ...)
            filters.append("ocr_source:(" + " OR ".join(engines) + ")")
        # date filter only if your meeting_date is stored as a string YYYY-MM-DD; if it's a date field, use proper Solr syntax
        if date_start and date_end:
            # string range works if meeting_date is yyyy-mm-dd; for true date field use meeting_date:[startT00:00:00Z TO endT23:59:59Z]
            filters.append(f"meeting_date:[{date_start} TO {date_end}]")

        # Query Solr
        docs, raw = solr_knn_search(qvec, k=top_k, filters=filters)

    st.subheader(f"Results ({len(docs)})")
    if not docs:
        st.info("No matches found.")
    else:
        for d in docs:
            with st.container(border=True):
                # Header line with score + basic metadata
                left, right = st.columns([0.75, 0.25])
                with left:
                    title = d.get("title") or d.get("doc_id") or d.get("id")
                    st.markdown(f"**{title}**")
                    md = d.get("meeting_date", "")
                    doc_id = d.get("doc_id", "")
                    src = d.get("ocr_source", "")
                    segi = d.get("segment_index", "")
                    st.caption(f"doc_id: `data/pdfs/{doc_id}.pdf` • Engine: `{src}` • Segment: `{segi}` • ID: `{d.get('id','')}`")
                with right:
                    st.metric("Score", f"{d.get('score', 0):.4f}")

                # Snippet
                st.write(d.get("text", "")[:1200])

        with st.expander("Raw Solr JSON"):
            st.code(json.dumps(raw, indent=2)[:5000], language="json")

else:
    st.info("Type a query and hit Search.")