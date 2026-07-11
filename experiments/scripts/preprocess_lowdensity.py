import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.common import build_preprocessor, load_master_dataset, split_dataset, save_preprocessor, ensure_directories, build_feature_frame


def main() -> None:
    ensure_directories()
    df = split_dataset(load_master_dataset(), "lowdensity")
    feature_frame = build_feature_frame(df)
    X = feature_frame.drop(columns=["price"])

    preprocessor = build_preprocessor(X)
    preprocessor.fit(X)

    save_preprocessor(preprocessor, "lowdensity_preprocessor.pkl")
    print(f"Low density preprocessing completed for {len(df)} rows")


if __name__ == "__main__":
    main()
