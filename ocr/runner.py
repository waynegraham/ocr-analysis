import argparse
from pathlib import Path
import importlib
import time
import traceback
import sys


def get_engine(engine_name: str):
    return importlib.import_module(f"engines.{engine_name}")


def find_pdfs(root: Path):
    # Robust discovery: recurse and accept .pdf/.PDF
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"]


def main():
    parser = argparse.ArgumentParser(description="Batch OCR runner")
    parser.add_argument(
        "--engine",
        required=True,
        choices=["tesseract", "easyocr", "paddleocr", "smoldocling"],
        help="Which OCR engine module to use (engines/<name>.py)",
    )
    parser.add_argument(
        "--pdf-dir",
        default="/app/data/pdfs",
        help="Directory containing PDFs (recursively scanned)",
    )
    parser.add_argument(
        "--output-dir",
        default="/app/ocr_output",
        help="Directory to write JSON outputs into (per-engine subdir)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap on number of PDFs to process (for quick tests)",
    )

    args = parser.parse_args()

    try:
        engine = get_engine(args.engine)
    except Exception as e:
        print(f"❌ Failed to import engine '{args.engine}': {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    pdf_dir = Path(args.pdf_dir)
    output_base = Path(args.output_dir)

    if not pdf_dir.exists():
        print(f"❌ PDF directory does not exist: {pdf_dir}", file=sys.stderr)
        sys.exit(1)

    output_base.mkdir(parents=True, exist_ok=True)

    pdf_files = find_pdfs(pdf_dir)
    if args.max_files:
        pdf_files = pdf_files[: args.max_files]

    print(f"🗂 Found {len(pdf_files)} PDF file(s) in {pdf_dir}")

    if not pdf_files:
        print("⚠️ No PDFs found. Check your mount paths and pdf directory.")
        return

    processed = 0
    for pdf_file in sorted(pdf_files):
        doc_id = pdf_file.stem
        engine_out_dir = output_base / args.engine
        engine_out_dir.mkdir(parents=True, exist_ok=True)
        output_path = engine_out_dir / f"{doc_id}.json"

        if output_path.exists():
            print(f"✅ Skipping {doc_id} (already processed)")
            continue

        print(f"🔍 Processing {doc_id} with {args.engine}")
        start_time = time.time()
        try:
            result = engine.ocr(pdf_file)  # expected to return a dict
            # Ensure dict and add timing if not already set
            if not isinstance(result, dict):
                result = {"segments": result}
            result.setdefault("processing_time", round(time.time() - start_time, 2))
            output_path.write_text(engine.to_json(result), encoding="utf-8")
            print(f"✅ Wrote {output_path.name} ({result['processing_time']}s)")
            processed += 1
        except KeyboardInterrupt:
            print("⏹️ Interrupted by user.")
            break
        except Exception as e:
            dur = round(time.time() - start_time, 2)
            print(f"❌ Error with {doc_id} after {dur}s: {e}", file=sys.stderr)
            traceback.print_exc()

    print(f"🏁 Done. Processed {processed} file(s). Outputs: {output_base / args.engine}")


if __name__ == "__main__":
    main()