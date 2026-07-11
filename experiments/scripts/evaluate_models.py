import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.common import ensure_directories, PROJECT_ROOT


def main() -> None:
    ensure_directories()
    metrics_paths = [
        Path(__file__).resolve().parents[1] / "outputs" / "condo_metrics.csv",
        Path(__file__).resolve().parents[1] / "outputs" / "lowdensity_metrics.csv",
        Path(__file__).resolve().parents[1] / "outputs" / "multiunit_metrics.csv",
    ]
    frames = []
    for path in metrics_paths:
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError("No evaluation metrics were found. Train the models first.")

    comparison = pd.concat(frames, ignore_index=True)
    comparison = comparison[["Model", "MAE", "RMSE", "R2", "MAPE (%)", "Training Time (s)", "Prediction Time (s)"]] 

    baseline_path = PROJECT_ROOT / "outputs" / "model_comparison.csv"
    if baseline_path.exists():
        baseline_metrics = pd.read_csv(baseline_path)
        if not baseline_metrics.empty:
            baseline_row = baseline_metrics.iloc[0].copy()
            baseline_row = pd.DataFrame([{
                "Model": "Current Single Model",
                "MAE": baseline_row.get("MAE", ""),
                "RMSE": baseline_row.get("RMSE", ""),
                "R2": baseline_row.get("R2", ""),
                "MAPE (%)": baseline_row.get("MAPE (%)", ""),
                "Training Time (s)": baseline_row.get("Training Time (s)", ""),
                "Prediction Time (s)": baseline_row.get("Prediction Time (s)", ""),
            }])
            comparison = pd.concat([baseline_row, comparison], ignore_index=True)

    output_path = Path(__file__).resolve().parents[1] / "outputs" / "comparison_report.csv"
    comparison.to_csv(output_path, index=False)
    print(f"Comparison report saved -> {output_path}")
    print(comparison)


if __name__ == "__main__":
    main()
