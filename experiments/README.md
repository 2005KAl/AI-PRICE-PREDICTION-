# Experimental Multi-Model House Price Pipeline

This folder is a completely isolated experimental pipeline for the Toronto House Price Prediction project. It does not modify the existing Flask app, current preprocessing flow, or the original model artifacts.

## Folder Structure

- data/: split datasets for each experimental group
- preprocessing/: fitted preprocessing pipelines for each group
- models/: specialized XGBoost models
- outputs/: evaluation metrics, plots, and comparison outputs
- scripts/: modular training and testing scripts

## Workflow

1. Split the master dataset into three category-specific datasets:
   - Condo
   - Low Density (Detached, Semi Detached, House)
   - Multi Unit (Townhouse, Duplex, Triplex, Fourplex)
2. Fit a separate preprocessing pipeline for each dataset.
3. Train a dedicated XGBoost model for each group.
4. Generate evaluation metrics, feature importance, SHAP, and error analysis plots.
5. Compare the specialized models with the existing single-model baseline.

## How to Train

Run the following commands from the project root:

```bash
python experiments/scripts/split_dataset.py
python experiments/scripts/preprocess_condo.py
python experiments/scripts/preprocess_lowdensity.py
python experiments/scripts/preprocess_multiunit.py
python experiments/scripts/train_condo.py
python experiments/scripts/train_lowdensity.py
python experiments/scripts/train_multiunit.py
python experiments/scripts/evaluate_models.py
```

## How to Test

```bash
python experiments/scripts/test_predictions.py
```

## How to Compare

```bash
python experiments/scripts/compare_models.py
```

## How to Add New Models

1. Create a new dataset filter in the shared utilities.
2. Add a new training script following the existing pattern.
3. Save outputs in the isolated experiment folders.
4. Update the README and comparison script if needed.

## Experiment Results (automatically generated)

- Comparison report: `experiments/outputs/comparison_report.csv`
- Per-model metrics: `experiments/outputs/condo_metrics.csv`, `experiments/outputs/lowdensity_metrics.csv`, `experiments/outputs/multiunit_metrics.csv`
- Models: `experiments/models/Condo_Model.pkl`, `experiments/models/LowDensity_Model.pkl`, `experiments/models/MultiUnit_Model.pkl`
- Preprocessors: `experiments/preprocessing/condo_preprocessor.pkl`, `experiments/preprocessing/lowdensity_preprocessor.pkl`, `experiments/preprocessing/multiunit_preprocessor.pkl`
- Test predictions report: `experiments/outputs/test_predictions_report.csv`
- Feature importance and SHAP outputs: `experiments/outputs/feature_importance/` and `experiments/outputs/shap/`

## Branch & Commit

I will push these experiment files to a new branch named `experiments/multi-model`. Commit message used:

```
Add isolated experiments multi-model pipeline: datasets, preprocessors, models, scripts, outputs, README
```

## Notes

- This `experiments/` folder is fully isolated and does not modify existing project files, the Flask app, or previously trained models.
- SHAP is attempted for each model but will gracefully fall back if unsupported by the model/feature encoding or library compatibility.
- You may see sklearn version warnings when loading old pickles; they are informational.

## Next steps

- To push the experiments to GitHub, run the commands below from the project root (these were executed by the assistant):

```bash
git checkout -b experiments/multi-model
git add experiments/
git commit -m "Add isolated experiments multi-model pipeline: datasets, preprocessors, models, scripts, outputs, README"
git push -u origin experiments/multi-model
```

If you want me to create a pull request after pushing, say so and I'll open one and provide the PR link.

---

## Manual (Step-by-step) Guide

This section is a complete manual to run, reproduce and inspect the experimental pipeline locally. Follow each step from a command prompt opened at the project root (the folder containing `app.py`).

1) Create / activate your Python environment

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# or on cmd
.\.venv\Scripts\activate.bat
# or macOS / Linux
source .venv/bin/activate
```

2) Install required packages

The experiments use `xgboost`, `shap`, `pandas`, `scikit-learn`, and `flask` for the UI. Install them with:

```bash
pip install -r requirements.txt
pip install xgboost shap flask
```

3) Verify the master processed dataset exists

Confirm the project processed dataset is present at `data/processed/ml_dataset.csv`. This pipeline reads that file to create the experimental splits.

```bash
python -c "import pandas as pd; print(pd.read_csv('data/processed/ml_dataset.csv').shape)"
```

4) Create the experimental datasets

Run the splitter to produce the three working CSVs in `experiments/data/`:

```bash
python experiments/scripts/split_dataset.py
```

You should see these files:

- `experiments/data/condo_dataset.csv`
- `experiments/data/lowdensity_dataset.csv`
- `experiments/data/multiunit_dataset.csv`

5) Fit per-group preprocessors

Each dataset gets its own preprocessor saved to `experiments/preprocessing/`:

```bash
python experiments/scripts/preprocess_condo.py
python experiments/scripts/preprocess_lowdensity.py
python experiments/scripts/preprocess_multiunit.py
```

6) Train each specialized model (with hyperparameter tuning)

Train and evaluate; outputs are saved under `experiments/models/` and `experiments/outputs/`.

```bash
python experiments/scripts/train_condo.py
python experiments/scripts/train_lowdensity.py
python experiments/scripts/train_multiunit.py
```

7) Generate comparison reports

Aggregate the per-model metrics into a single comparison report:

```bash
python experiments/scripts/evaluate_models.py
```

8) Test prediction behavior vs baseline

Run the quick test that generates `experiments/outputs/test_predictions_report.csv` comparing current single-model predictions to the new specialized models for representative home types:

```bash
python experiments/scripts/test_predictions.py
```

9) Run the isolated experimental UI (optional)

The UI is intentionally isolated at `experiments/ui/` and runs on port 5001 by default.

```bash
pip install flask pandas
cd experiments/ui
python app.py
# open http://localhost:5001
```

10) Review outputs and artifacts

Key locations:

- Datasets: `experiments/data/`
- Preprocessors: `experiments/preprocessing/`
- Models: `experiments/models/`
- Outputs: `experiments/outputs/` (metrics, comparison report, feature importance, shap, error analysis)
- UI: `experiments/ui/`

11) Push changes and create PR (already done by assistant)

If you need to push locally (instead of using the branch already created), run:

```bash
git checkout -b experiments/multi-model
git add experiments/
git commit -m "Add isolated experiments multi-model pipeline and UI"
git push -u origin experiments/multi-model
```

To open a PR on GitHub, go to the URL suggested by `git push` or use the GitHub web UI.

12) Troubleshooting

- If `xgboost` or `shap` fails to install, ensure Microsoft Visual C++ build tools are available on Windows, or use a conda environment: `conda create -n exp python=3.10 && conda activate exp && conda install -c conda-forge xgboost shap`.
- If you see sklearn pickle version warnings when loading older pickles, these are informational; re-generate the preprocessor/model under this environment to avoid them.
- SHAP generation may be skipped for some models if XGBoost's categorical splitting or feature encoding isn't compatible — the pipeline falls back and saves a placeholder image.

13) Reverting the experiment branch

If you want to remove the `experiments/multi-model` branch from the remote:

```bash
git push origin --delete experiments/multi-model
git branch -D experiments/multi-model
```

---

If you'd like, I can open the GitHub Pull Request for the `experiments/multi-model` branch and add a short PR description and reviewers. Say `open PR` and I will create it and provide the PR link.
