import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def summarize_ocr(ocr_output_dir="data/ocr_output", output_dir="summary", plot=True):
    ocr_base = Path(ocr_output_dir)
    output_base = Path(output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    records = []

    for engine_dir in ocr_base.glob("*"):
        if not engine_dir.is_dir():
            continue
        engine = engine_dir.name
        for json_file in engine_dir.glob("*.json"):
            try:
                data = json.loads(json_file.read_text())
                records.append({
                    "engine": engine,
                    "document": json_file.stem,
                    "avg_confidence": data.get("avg_confidence", None),
                    "processing_time_sec": data.get("processing_time", None)
                })
            except Exception as e:
                print(f"❌ Error reading {json_file.name}: {e}")

    df = pd.DataFrame(records)

    if df.empty:
        print("⚠️ No data found to summarize.")
        return

    # Save CSV
    csv_path = output_base / "ocr_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved summary CSV to {csv_path}")

    # Plot if requested
    if plot:
        plt.figure(figsize=(10, 6))
        for engine in df["engine"].unique():
            subset = df[df["engine"] == engine]
            plt.scatter(subset["processing_time_sec"], subset["avg_confidence"],
                        label=engine, alpha=0.7)
        plt.xlabel("Processing Time (seconds)")
        plt.ylabel("Average Confidence")
        plt.title("OCR Performance: Confidence vs Processing Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path = output_base / "ocr_confidence_vs_time.png"
        plt.savefig(plot_path)
        print(f"✅ Saved plot to {plot_path}")

if __name__ == "__main__":
    summarize_ocr()