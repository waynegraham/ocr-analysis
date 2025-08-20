import easyocr
from pdf2image import convert_from_path
from utils.segment import segment_text
import json
import torch
import time
import numpy as np

def ocr(pdf_path):
    from pdf2image import convert_from_path
    import easyocr
    from utils.segment import segment_text

    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available()) # or set to False

    start_time = time.time()

    images = convert_from_path(str(pdf_path))

    full_text = ""
    confidences = []

    for img in images:
        img_np = np.array(img)  # Convert PIL to NumPy
        result = reader.readtext(img_np)
        page_text = []
        for item in result:
            text = item[1]
            conf = item[2]
            page_text.append(text)
            confidences.append(conf)
        full_text += " ".join(page_text) + "\n"

    segments = segment_text(full_text)
    avg_conf = round(sum(confidences) / len(confidences), 3) if confidences else 0.0
    processing_time = round(time.time() - start_time, 2)

    print(f"📊 {pdf_path.name} - EasyOCR avg confidence: {avg_conf} ({processing_time}s)")

    return {
        "segments": segments,
        "avg_confidence": avg_conf,
        "processing_time": processing_time
    }

def to_json(result):
    return json.dumps(result, indent=2)