import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.common import load_master_dataset, split_dataset, save_dataset, ensure_directories


def main() -> None:
    ensure_directories()
    master_df = load_master_dataset()

    condo_df = split_dataset(master_df, "condo")
    lowdensity_df = split_dataset(master_df, "lowdensity")
    multiunit_df = split_dataset(master_df, "multiunit")

    save_dataset(condo_df, "condo_dataset.csv")
    save_dataset(lowdensity_df, "lowdensity_dataset.csv")
    save_dataset(multiunit_df, "multiunit_dataset.csv")

    print("\nDataset split completed successfully.")


if __name__ == "__main__":
    main()
