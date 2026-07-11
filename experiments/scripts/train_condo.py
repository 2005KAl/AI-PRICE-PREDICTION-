import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.common import (
    ensure_directories,
    train_and_evaluate_model,
)


def main() -> None:
    ensure_directories()
    dataset_path = Path(__file__).resolve().parents[1] / "data" / "condo_dataset.csv"
    preprocessor_path = Path(__file__).resolve().parents[1] / "preprocessing" / "condo_preprocessor.pkl"
    model_path = Path(__file__).resolve().parents[1] / "models" / "Condo_Model.pkl"
    metrics_path = Path(__file__).resolve().parents[1] / "outputs" / "condo_metrics.csv"
    output_dir = Path(__file__).resolve().parents[1] / "outputs"
    train_and_evaluate_model("Condo", dataset_path, preprocessor_path, model_path, metrics_path, output_dir)


if __name__ == "__main__":
    main()
