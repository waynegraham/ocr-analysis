# ocr/engines/paddleocr.py
import time, json, numpy as np
from pathlib import Path
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
from utils.segment import segment_text

# ocr_engine = PaddleOCR(
#     use_angle_cls=True,
#     lang='en'
# )

ocr_engine = PaddleOCR(
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="PP-OCRv5_mobile_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False) 


def ocr(pdf_path):
    pdf_path = Path(pdf_path)
    start_time = time.time()

    images = convert_from_path(str(pdf_path))
    full_text, confidences = "", []

    # for img in images:
    #     img_np = np.array(img)
    #     result = ocr_engine.ocr(img_np)

    #     if not result:
    #         full_text += "\n"
    #         continue

    #     page_items = result[0]
    #     tokens = []
    #     for line in page_items:
    #         text, conf = line[1]
    #         text = (text or "").strip()
    #         if text:
    #             tokens.append(text)
    #             try:
    #                 confidences.append(float(conf))
    #             except Exception:
    #                 pass
    #     full_text += " ".join(tokens) + "\n"

    segments = segment_text(full_text)
    avg_conf = round(sum(confidences) / len(confidences), 3) if confidences else 0.0
    processing_time = round(time.time() - start_time, 2)

    print(f"📊 {pdf_path.name} - PaddleOCR(v3) avg confidence: {avg_conf} ({processing_time}s)")
    return {"segments": segments, "avg_confidence": avg_conf, "processing_time": processing_time}


def to_json(result):
    return json.dumps(result, indent=2)