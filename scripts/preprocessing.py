import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ==========================================================
# PROJECT PATHS
# ==========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

DATASET_PATH = os.path.join(
    PROCESSED_DIR,
    "ml_dataset.csv"
)

# ==========================================================
# LOAD DATASET
# ==========================================================

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

df = pd.read_csv(DATASET_PATH)

print(f"Dataset Shape : {df.shape}")

print("\nColumns")

print(df.columns.tolist())

print("\nMissing Values Before Processing\n")

print(df.isnull().sum())

# ==========================================================
# REMOVE UNNECESSARY COLUMNS
# ==========================================================

columns_to_drop = [

    # Data Leakage
    "list_price",
    "sold_price",

    # Remove duplicate neighbourhood
    "neighbourhood",

    # GeoJSON Metadata
    "_id",
    "AREA_ID",
    "AREA_ATTR_ID",
    "PARENT_AREA_ID",
    "AREA_SHORT_CODE",
    "AREA_LONG_CODE",
    "AREA_NAME",
    "AREA_DESC",
    "CLASSIFICATION",
    "CLASSIFICATION_CODE",
    "OBJECTID"

]

df.drop(
    columns=[c for c in columns_to_drop if c in df.columns],
    inplace=True
)

print("\nDropped Columns")

print(columns_to_drop)

# ==========================================================
# TARGET
# ==========================================================

y = df["price"]

X = df.drop(columns=["price"])

# ==========================================================
# FEATURE TYPES
# ==========================================================

categorical_features = [

    "home_type",

    "home_type_bucket",

    "Neighbourhood"

]

numerical_features = [

    col

    for col in X.columns

    if col not in categorical_features

]

print("\nNumerical Features")

print(numerical_features)

print("\nCategorical Features")

print(categorical_features)

# ==========================================================
# NUMERICAL PIPELINE
# ==========================================================

numeric_transformer = Pipeline(

    steps=[

        ("imputer", SimpleImputer(strategy="median")),

        ("scaler", StandardScaler())

    ]

)

# ==========================================================
# CATEGORICAL PIPELINE
# ==========================================================

categorical_transformer = Pipeline(

    steps=[

        ("imputer", SimpleImputer(strategy="most_frequent")),

        (

            "encoder",

            OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

        )

    ]

)

# ==========================================================
# COLUMN TRANSFORMER
# ==========================================================

preprocessor = ColumnTransformer(

    transformers=[

        (

            "num",

            numeric_transformer,

            numerical_features

        ),

        (

            "cat",

            categorical_transformer,

            categorical_features

        )

    ]

)

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42

)

print("\nTrain Shape :", X_train.shape)

print("Test Shape  :", X_test.shape)

# ==========================================================
# FIT PREPROCESSOR
# ==========================================================

print("\nFitting Preprocessor...")

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)

print("Preprocessing Completed")

# ==========================================================
# FEATURE NAMES
# ==========================================================

feature_names = preprocessor.get_feature_names_out()

print("\nTotal Features After Encoding :", len(feature_names))

# ==========================================================
# SAVE PREPROCESSOR
# ==========================================================

joblib.dump(

    preprocessor,

    os.path.join(

        MODEL_DIR,

        "preprocessor.pkl"

    )

)

joblib.dump(

    feature_names,

    os.path.join(

        MODEL_DIR,

        "feature_names.pkl"

    )

)

joblib.dump(

    X.columns.tolist(),

    os.path.join(

        MODEL_DIR,

        "original_features.pkl"

    )

)

# ==========================================================
# SAVE TRAIN TEST DATA
# ==========================================================

X_train_df = pd.DataFrame(

    X_train_processed,

    columns=feature_names

)

X_test_df = pd.DataFrame(
    X_test_processed,
    columns=feature_names,
    index=X_test.index
)

X_train_df.to_csv(

    os.path.join(

        PROCESSED_DIR,

        "X_train.csv"

    ),

    index=False

)

X_test_df.to_csv(

    os.path.join(

        PROCESSED_DIR,

        "X_test.csv"

    ),

    index=False

)

y_train.to_csv(

    os.path.join(

        PROCESSED_DIR,

        "y_train.csv"

    ),

    index=False

)

y_test.to_csv(

    os.path.join(

        PROCESSED_DIR,

        "y_test.csv"

    ),

    index=False

)

# ==========================================================
# SUMMARY
# ==========================================================

print("\n")

print("=" * 60)

print("PREPROCESSING COMPLETED")

print("=" * 60)

print(f"Training Samples : {len(X_train_df)}")

print(f"Testing Samples  : {len(X_test_df)}")

print(f"\nOriginal Features : {len(X.columns)}")

print(f"Final Features    : {len(feature_names)}")

print("\nSaved Files")

print("✓ X_train.csv")

print("✓ X_test.csv")

print("✓ y_train.csv")

print("✓ y_test.csv")

print("✓ preprocessor.pkl")

print("✓ feature_names.pkl")

print("✓ original_features.pkl")

print("\nPreprocessing Pipeline Ready for Model Training")