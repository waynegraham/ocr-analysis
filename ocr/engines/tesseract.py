import pytesseract
from pdf2image import convert_from_path
from utils.segment import segment_text
import json
import pandas as pd
from io import StringIO
import csv


def ocr(pdf_path):
    images = convert_from_path(str(pdf_path))
    full_text = ""
    confidences = []

    for img in images:
        # Extract TSV data (includes confidence)
        tsv_data = pytesseract.image_to_data(img, lang="eng", output_type=pytesseract.Output.STRING)
        # df = pd.read_csv(StringIO(tsv_data), sep="\t")
        try:
            df = pd.read_csv(
                StringIO(tsv_data),
                sep="\t",
                quoting=csv.QUOTE_NONE,
                engine="python",
                on_bad_lines="skip"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to parse TSV data from Tesseract: {e}")

        # Keep non-empty, valid-confidence words
        for _, row in df.iterrows():
            text = str(row.get("text", "")).strip()
            conf = row.get("conf", -1)

            if text and conf != -1:
                full_text += text + " "
                try:
                    conf_float = float(conf)
                    if conf_float >= 0:
                        confidences.append(conf_float)
                except ValueError:
                    continue

        full_text += "\n"

    segments = segment_text(full_text)

    avg_conf = round(sum(confidences) / len(confidences), 2) if confidences else 0.0
    print(f"📊 {pdf_path.name} - Tesseract avg confidence: {avg_conf}")

    return {
        "segments": segments,
        "avg_confidence": avg_conf
    }


def to_json(result):
    return json.dumps(result, indent=2)