import time
import json
import numpy as np
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
from utils.segment import segment_text

# Init OCR engine (auto detects GPU if available and paddle is installed
# correctly)

import torch

ocr_engine = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    use_gpu=torch.cuda.is_available()
)


print("🔧 Paddle device:", paddle.get_device()) 


def ocr(pdf_path):
    start_time = time.time()

    images = convert_from_path(str(pdf_path))
    full_text = ""
    confidences = []

    for img in images:
        img_np = np.array(img)
        result = ocr_engine.ocr(img_np, cls=True)

        # result: [[(bbox, (text, confidence)), ...]]
        for line in result[0]:
            text, conf = line[1]
            if text.strip():
                full_text += text.strip() + " "
                confidences.append(conf)

        full_text += "\n"

    segments = segment_text(full_text)
    avg_conf = round(sum(confidences) / len(confidences), 3) if confidences else 0.0
    processing_time = round(time.time() - start_time, 2)

    print(f"📊 {pdf_path.name} - PaddleOCR avg confidence: {avg_conf} ({processing_time}s)")

    return {
        "segments": segments,
        "avg_confidence": avg_conf,
        "processing_time": processing_time
    }


def to_json(result):
    return json.dumps(result, indent=2)
