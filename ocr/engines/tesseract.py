import pytesseract
from pdf2image import convert_from_path
from utils.segment import segment_text
import json

def ocr(pdf_path):
    images = convert_from_path(str(pdf_path))
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img, lang="eng") + "\n"

        if not text.strip():
            raise ValueError("Tesseract returned no text!")

    return segment_text(text)

def to_json(segments):
    return json.dumps({"segments": segments}, indent=2)