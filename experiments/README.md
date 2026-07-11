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
