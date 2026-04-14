# B2B trial-to-paid conversion (time-aware ML)

Portfolio project: predict **trial-to-paid conversion** within a **120-day** window from product usage and firmographic features. Uses a **time-based train/validation split**, leakage checks, a **business heuristic baseline**, **logistic regression**, **random forest**, and **XGBoost**, with **probability calibration**, **PR/ROC**, and **SHAP** / permutation importance. Includes **monthly aggregation** (MAE / MAPE) aligned with go-to-market reporting.

This repository reflects an independent technical exercise; **no employer or product name** is used in the title or description.

## Data

Add your parquet files under `data/`:

- `Train.parquet`
- `Test.parquet`

Parquet files are **not** shipped here (privacy / confidentiality). Place them locally to execute the full pipeline.

## Run

```bash
cd trial-conversion-prediction-ml
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/trial_conversion_analysis.ipynb
```

Run Jupyter from the **repository root** so paths like `data/Train.parquet` resolve.

## Stack

Python, pandas, scikit-learn, XGBoost, SHAP, matplotlib/seaborn, pyarrow.

## Note

The notebook is intended to showcase **methodology and code quality**; reported metrics depend on the private dataset you attach under `data/`.
