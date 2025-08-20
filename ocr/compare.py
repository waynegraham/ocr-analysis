import json
from pathlib import Path

def get_confidences(engine):
    root = Path(f"ocr_output/{engine}")
    for file in sorted(root.glob("*.json")):
        data = json.loads(file.read_text())
        print(f"{file.stem},{data.get('avg_confidence', 'N/A')}")

# Compare:
get_confidences("easyocr")
get_confidences("tesseract")