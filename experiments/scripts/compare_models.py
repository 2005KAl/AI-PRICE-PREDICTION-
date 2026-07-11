import sys
from pathlib import Path
import joblib

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.common import ensure_directories, CURRENT_MODEL_PATH


def main() -> None:
    ensure_directories()
    current_model = joblib.load(CURRENT_MODEL_PATH)
    comparison_rows = [{"Model": "Current Single Model", "Type": "Baseline", "Model Object": current_model}]
    for name in ["Condo", "LowDensity", "MultiUnit"]:
        model_path = Path(__file__).resolve().parents[1] / "models" / f"{name}_Model.pkl"
        if model_path.exists():
            comparison_rows.append({"Model": f"{name} Model", "Type": "Experimental", "Model Object": joblib.load(model_path)})
    print("Comparison setup complete")
    print([item["Model"] for item in comparison_rows])


if __name__ == "__main__":
    main()
