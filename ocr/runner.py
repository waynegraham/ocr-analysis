import argparse
from pathlib import Path
import importlib

PDF_DIR = Path("/app/data/pdfs")
OUTPUT_DIR = Path("/app/data/ocr_output")


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

    for pdf_file in sorted(PDF_DIR.glob("*.pdf")):
        doc_id = pdf_file.stem
        output_path = OUTPUT_DIR / args.engine / f"{doc_id}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"🗂 Found {len(list(PDF_DIR.glob('*.pdf')))} PDF files in {PDF_DIR}")

        if output_path.exists():
            print(f"✅ Skipping {doc_id} (already processed)")
            continue

        print(f"🔍 Processing {doc_id} with {args.engine}")
        try:
            segments = engine.ocr(pdf_file)
            output_path.write_text(engine.to_json(segments), encoding="utf-8")
        except Exception as e:
            print(f"❌ Error with {doc_id}: {e}")


if __name__ == "__main__":
    main()
