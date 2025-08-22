# OCR Pipeline

This project explores the effectiveness of semantic search (using Apache Solr with dense vector support) and retrieval-augmented generation (RAG) systems over a corpus of OCR-processed PDFs. It includes benchmarking of multiple OCR engines, document segmentation, and a pipeline for comparing retrieval and generation quality.

---

## ✅ Features So Far

### ✅ OCR Benchmarking Pipeline

- Modular architecture for running different OCR engines:
  - Tesseract
  - EasyOCR (GPU/CPU autodetect)
- Paragraph or sentence segmentation using NLTK
- Confidence scoring (EasyOCR) and structured JSON output
- Designed for headless, batch-friendly execution in Docker

### ✅ Solr with Dense Vector Support

- Solr 9.9 with `knn_vector` field for semantic retrieval
- Custom schema with meeting metadata and text fields
- Supports embedding ingestion via Python

### ✅ Dockerized & Cross-platform

- Runs locally on macOS with CPU
- Supports GPU acceleration via Docker (CUDA) on WSL/Linux
- Cross-platform compatibility built into Compose

---

## 🐳 Running the OCR Pipeline

### 1. Place PDFs

Add all source PDFs to:

```bash
./data/pdfs/
```

### 2. Build the OCR container

```bash
docker compose build ocr-runner
```

### 3. Run Tesseract OCR

```bash
docker compose run ocr-runner --engine tesseract
```

### 4. Run EasyOCR (auto GPU if available)

```bash
docker compose run ocr-runner-gpu --engine easyocr
```

### 5. Run PaddleOCR

```bash
docker compose run ocr-runner-gpu --engine paddleocr
```

## Output Format (per document)

Each engine creates:

```bash
ocr_output/{engine}/{doc_id}.json
```

Example:

```json
{
  "segments": [
    "Meeting held January 14, 1955.",
    "Budget discussion: President Smith stated the need for restraint."
  ],
  "avg_confidence": 0.873
}
```

## Index to Solr

```bash
python tools/index_to_solr.py \
    --engine tesseract \
    --ocr-dir data/ocr_output \
    --solr http://localhost:8983/solr/trustees \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --vector-field vector \
    --commit
```

## Environment Configuration

### GPU on WSL or Linux

Ensure you have:

- NVIDIA Container Toolkit installed
- Compose service includes:

```yaml
deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
```

### NLTK Preloading

All required NLTK resources (`punkt`, `punkt_tab`) are preinstalled in the container at `/usr/local/nltk_data`.

## Clean Up

```bash
docker compose down -v
docker builder prune -f
```

## TODO

- Update annotator to prefer page_index (no inference).
- Update indexer to set an indexed/stored page_index field.
- Add pages, file size, page count, and segment2page indexes to OCR JSON
- In Semantic Search UI, add checkbox for "Use Hybrid (BB25+KNN)
- Add Next/Prev buttons wired to start for both knn and hybrid functions
- Show Solr QTime and client latency
- Implement a segmenters module and try
  - sentence (punkt), paragraph, sliding window (~220 tokens, stride 120)
  - Encode & index each variant (tag segmenter field)
  - Re-run `eval_retrieval.py` with `--filter segmenter:window220`
- Add a second vector field for Solr at 768-dim 
- Test models
  - 382-dim (`all-MiniLM-L6-v2`, `intfloat/e5-small-v2`)
  - 768-dim: (`bge-base-en-v1.5`, `intfloat/e5-base`)
- Extend indexer/UI to Switch VECTOR_FIELD based on model; compare metrics & latency
- Create reports
  - Reads `runs/*.csv`
  - Output Markdown/HTML with tables (p@1, p@5, MRR, nDCG, latency) grouped by **engine x segmenter x model x mode**.
  - Include 2-3 "case study" queries showing wins/fails with snippet + metadata
- Add a toggle to mark partial or negative
- 