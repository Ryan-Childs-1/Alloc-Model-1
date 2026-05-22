# Allocation AI Streamlit App

Flat-file Streamlit project for Daily Allocation CSV/XLSX files formatted like the submitted Sportsman's Warehouse exports.

## Files

- `app.py` — full Streamlit app, parser, model training, prediction, post-processing, export logic.
- `requirements.txt` — Streamlit Cloud dependencies.
- `allocation_ai_general_model.joblib` — pretrained general model trained on the submitted allocation CSV files.
- `general_model_validation_sample.csv` — sample validation rows from the pretrained model.

## How to run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## How to deploy on Streamlit Cloud

1. Create a GitHub repo.
2. Upload all files in this folder to the repo root. Do not place them in subfolders.
3. In Streamlit Cloud, choose the repo and set the main file to `app.py`.
4. Deploy.

## Recommended workflow

### Predict/edit a new file

1. Open the **Predict / Edit File** tab.
2. Upload `allocation_ai_general_model.joblib` or a newer model you trained.
3. Upload the Daily Allocation CSV/XLSX to edit.
4. Click **Run predictions and build edited file**.
5. Download the edited XLSX or CSV.

### Train a new model

1. Open the **Train Model** tab.
2. Upload corrected historical Daily Allocation files.
3. Keep target mode as **Manual Adjustment from Alloc. Rec.** unless you want the model to ignore the allocation formula.
4. Download the trained `.joblib` model.

### Continue training

1. Open the **Continue Training** tab.
2. Upload the current `.joblib` model.
3. Upload newly corrected files.
4. Download the updated model.

## Important fixes in this version

- Fixed the `ValueError: Input contains NaN` issue by filling missing group labels before `GroupShuffleSplit`.
- Optimized parsing for the exact submitted CSV structure: blank first row and real headers on row 2.
- Correctly handles duplicate `MIL` and `FLM` columns by renaming the second copies to `MIL.1` and `FLM.1`.
- Treats `Z - No Alloc.` as no-allocation, not as an allocation row.
- Avoids leakage columns like `Final Supply`, `Left DC`, `Demand Check`, and `Final Cost` during training.
- Trains on a balanced subset of allocation-context rows plus sampled no-allocation rows instead of letting the model get overwhelmed by zeros.
- Adds audit columns explaining model outputs and safety-rule changes.

## Pretrained model notes

The included model was trained from the six submitted Daily Allocation CSVs using:

- Model: Extra Trees Regressor
- Target: Manual adjustment from `Alloc. Rec.` to `Final Alloc.`
- Rows used: 47,619
- Features: 89
- Validation MAE: about 1.28 allocation units

This should be treated as a strong starter model, not a final production model. It should improve as you upload more corrected outputs into the Continue Training tab.
