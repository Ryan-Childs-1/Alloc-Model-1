# Allocation AI — Keras Neural Network Final Alloc. Filler

Flat-file Streamlit app for Daily Allocation CSV/XLSX files formatted like the submitted Sportsman's Warehouse allocation exports.

## Files

Keep all files in the same GitHub folder:

```text
app.py
requirements.txt
README.md
allocation_ai_keras_nn_model.keras
allocation_ai_keras_preprocessor.joblib
training_metrics.json
keras_nn_validation_sample.csv
train_keras_nn.py
```

## How to run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud

1. Create a GitHub repo.
2. Upload all files in this folder directly into the repo root.
3. In Streamlit Cloud, set the main file path to `app.py`.
4. Deploy.

## Workflow

1. Upload a Daily Allocation CSV/XLSX.
2. Click **Run Keras Neural Network and Fill Final Alloc.**
3. Download the edited XLSX or CSV.

## Model notes

The included Keras model was trained on all six submitted Daily Allocation files. The app keeps all decision-signal rows: rows marked `Allocate`, rows marked `Review`, rows with positive `Alloc. Rec.`, and rows with positive historical `Final Alloc.`. Blank/no-signal rows are protected with business rules so the model does not fill every row.

The program fixes the previous export error:

```text
ValueError: cannot convert float NaN to integer
```

The XLSX writer now uses NaN-safe column width logic for all-blank columns.
