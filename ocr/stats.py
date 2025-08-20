import json
from pathlib import Path

def summarize(engine):
    records = []
    for file in Path(f"data/ocr_output/{engine}").glob("*.json"):
        data = json.loads(file.read_text())
        records.append({
            "doc": file.stem,
            "avg_conf": data.get("avg_confidence"),
            "time_sec": data.get("processing_time")
        })
    return records

# Example usage
from pprint import pprint
pprint(summarize("tesseract"))