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
