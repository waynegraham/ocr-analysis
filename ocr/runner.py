import argparse
from pathlib import Path
import importlib
import time

PDF_DIR = Path("/app/data/pdfs")
OUTPUT_BASE = Path("/app/data/ocr_output")


def get_engine(engine_name):
    return importlib.import_module(f"engines.{engine_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--engine",
        required=True,
        choices=["tesseract", "easyocr", "paddleocr", "smoldocling"],
    )
    parser.add_argument("--pdf-dir", default="../data/pdfs")
    parser.add_argument("--output-dir", default="../ocr_output")

    args = parser.parse_args()

    engine = get_engine(args.engine)

    print(f"🗂 Found {len(list(PDF_DIR.glob('*.pdf')))} PDF files in {PDF_DIR}")

    for pdf_file in sorted(PDF_DIR.glob("*.pdf")):
        doc_id = pdf_file.stem
        output_path = OUTPUT_BASE / args.engine / f"{doc_id}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists():
            print(f"✅ Skipping {doc_id} (already processed)")
            continue

        print(f"🔍 Processing {doc_id} with {args.engine}")
        start_time = time.time()

        try:
            result = engine.ocr(pdf_file)
            result["processing_time"] = round(time.time() - start_time, 2)  # ⏱️ Add timing
            output_path.write_text(engine.to_json(result), encoding="utf-8")
            print(f"✅ Wrote {output_path.name} ({result['processing_time']}s)")
        except Exception as e:
            print(f"❌ Error with {doc_id}: {e}")


if __name__ == "__main__":
    main()
