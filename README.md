# Churn prediction – exam case (Maskininlärning och Deep Learning)

Predicts which subscription customers will churn within 30 days, using `data/customer_churn_exam.csv`.
**Recommended model:** Logistic Regression (scikit-learn pipeline), test ROC-AUC 0.76 / PR-AUC 0.41.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run
Open `notebooks/churn_analysis.ipynb` and run all cells (≈2 minutes on a laptop CPU).
Tested with Python 3.13.7 on macOS.

## Structure
```
data/        raw dataset
notebooks/   churn_analysis.ipynb – full analysis (sections 1–9)
figures/     plots used in the report and slides
models/      churn_pipeline.joblib (final pipeline) + model_metadata.json (threshold, schema, versions)
report/      technical report (PDF)
slides/      presentation
```