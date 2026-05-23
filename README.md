# Allocation AI — Keras Neural Network Streamlit App

This is a flat-file Streamlit app designed for the Daily Allocation CSV/XLSX format submitted by Ryan.

## Files

Place all files in the same GitHub repo folder:

```text
app.py
requirements.txt
README.md
allocation_ai_keras_nn_model.keras
allocation_ai_keras_preprocessor.joblib
training_metrics.json
keras_nn_validation_sample.csv
```

There are no nested folders required.

## What it does

The app loads the included Keras neural network and fills the `Final Alloc.` column for a newly uploaded Daily Allocation file.

It handles the submitted CSV structure, including:

- first-row notes / blank row before the real headers
- real headers on row 2
- duplicate columns such as `MIL` / `MIL.1` and `FLM` / `FLM.1`
- trailing unnamed columns
- `Z - No Alloc.` rows
- blank `Final Alloc.` values representing zero/no allocation

## Model behavior

The model predicts a raw neural-network allocation value, then applies internal business-safety rules:

- FLM rounding
- no-allocation flag protection
- blank/non-signal row protection
- D60 + one FLM final-supply cap
- DC available cap
- audit columns for review

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Create a GitHub repo.
2. Upload all files from this folder to the root of the repo.
3. Go to Streamlit Cloud.
4. Select the repo.
5. Set the app file to `app.py`.
6. Deploy.

## Continue training

Use the `Continue Training` tab to upload corrected allocation files. The app has no model-setting UI; it uses fixed internal training parameters. After training, download both:

```text
allocation_ai_keras_nn_model.keras
allocation_ai_keras_preprocessor.joblib
```

Replace the old versions in GitHub with the new versions.

## Training summary for included model

See `training_metrics.json` for metrics generated during the build.
