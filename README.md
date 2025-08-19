# OCR Pipeline

```bash
cd ocr
docker build -t ocr-runner .
docker run --rm -v $(pwd)/../data:/app/data -v $(pwd)/../ocr_output:/app/ocr_output ocr-runner python runner.py --engine tesseract
```

docker-compose run ocr-runner --engine tesseract

docker-compose run ocr-runner --engine tesseract --pdf-dir /app/data/pdfs