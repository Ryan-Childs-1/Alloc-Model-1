# Allocation AI Streamlit App

A flat-file Streamlit application that learns from corrected Daily Allocation spreadsheets and predicts values for the `Final Alloc.` column.

## Files

Keep all files in the same folder:

```text
app.py
requirements.txt
README.md
```

No nested app folders are required.

## Local Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud Deployment

1. Create a GitHub repo.
2. Upload `app.py`, `requirements.txt`, and `README.md` into the repo root.
3. In Streamlit Cloud, select the repo.
4. Set the main file path to `app.py`.
5. Deploy.

## Recommended Workflow

1. Open the **Train Model** tab.
2. Upload historical corrected Daily Allocation files where `Final Alloc.` is the correct/manual value.
3. Train the model and download the `.joblib` model bundle.
4. Open the **Predict Final Alloc.** tab.
5. Upload the saved model bundle and a new Daily Allocation file.
6. Download the edited XLSX/CSV with new `Final Alloc.` values.
7. Review the output and correct any rows needed.
8. Use the corrected output in **Continue Training** to create an improved model bundle.

## Model Design

The default model target is **Manual Adjustment from Alloc. Rec.**:

```text
Predicted Final Alloc. = Alloc. Rec. + ML-predicted adjustment
```

This is usually better than predicting `Final Alloc.` from scratch because the existing spreadsheet allocation recommendation already contains useful business logic.

## Safety Rules

After prediction, the app applies business constraints:

- Non-negative allocation values
- FLM rounding
- Optional blank-row/no-allocation protection
- Optional D60 demand cap
- Optional DC available cap
- Review flags for risky or low-confidence rows

## Notes

Streamlit Cloud local storage is not permanent. For a safe workflow, download the model bundle after training and upload it again in future sessions.
