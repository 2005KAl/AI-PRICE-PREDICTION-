from pathlib import Path
import pandas as pd
import os

print("="*60)
print("SEARCHING FOR DATASET")
print("="*60)

current = Path(__file__).resolve().parent

# Go one level up (project root)
project_root = current.parent

print("Current Folder :", current)
print("Project Root   :", project_root)

dataset_path = None

for file in project_root.rglob("ml_dataset.csv"):
    dataset_path = file
    break

if dataset_path is None:
    print("\n❌ ml_dataset.csv was NOT found.")
    print("\nSearching project:")
    print(project_root)
    exit()

print("\n✅ Dataset Found:")
print(dataset_path)

df = pd.read_csv(dataset_path)

print("\nDataset Shape:", df.shape)

print("\nHome Types:")
print(df["home_type"].unique())

print("\nNeighbourhood Samples:")
print(df["Neighbourhood"].unique()[:20])

subset = df[
    (df["estimated_area_sqft"] >= 600) &
    (df["estimated_area_sqft"] <= 700)
]

print("\n600-700 sqft Properties:", len(subset))

print("\nPrice Statistics")
print(subset["price"].describe())

print("=" * 60)
print("HOME TYPE BUCKET COUNTS")
print("=" * 60)

print(df["home_type_bucket"].value_counts())

print("\n")

print("=" * 60)
print("HOME TYPE vs HOME TYPE BUCKET")
print("=" * 60)

print(df[["home_type", "home_type_bucket"]].drop_duplicates())