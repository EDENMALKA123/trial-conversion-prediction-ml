# Trial conversion prediction (time-aware ML)

Take-home style project: predict **trial-to-paid conversion** within a **120-day** window from product usage and firmographic features. Uses a **strict time-based train/validation split**, leakage checks, a simple **business baseline**, **logistic regression**, **random forest**, and **XGBoost**, with **probability calibration**, **PR/ROC**, and **SHAP** / permutation importance. Includes **monthly aggregation** (MAE / MAPE) to align model scores with go-to-market reporting.

## Data

Place the assignment files in `data/`:

- `Train.parquet`
- `Test.parquet`

These files are **not** included in this repository (privacy / NDA). If you are the owner of the data, copy them locally before running.

## Run

```bash
cd atera-trial-conversion-ml
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/atera_ds_home_assignment.ipynb
```

Run Jupyter from the **repository root** so paths like `data/Train.parquet` resolve.

## Stack

Python, pandas, scikit-learn, XGBoost, SHAP, matplotlib/seaborn, pyarrow.

## Note

For employers: this repo showcases methodology and code structure; scores in the notebook reflect the original assignment setup on the private dataset.
