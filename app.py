# Allocation AI - Keras Neural Network Streamlit App
# Flat-file app: place this file beside allocation_ai_keras_nn_model.keras and allocation_ai_keras_preprocessor.joblib.

import os
os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import io
import json
import hashlib
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:
    import keras
except Exception as exc:
    keras = None
    KERAS_IMPORT_ERROR = exc
else:
    KERAS_IMPORT_ERROR = None

APP_TITLE = "Allocation AI — Keras NN Final Alloc. Filler"
DEFAULT_MODEL_PATH = "allocation_ai_keras_nn_model.keras"
DEFAULT_PREPROCESSOR_PATH = "allocation_ai_keras_preprocessor.joblib"

st.set_page_config(page_title="Allocation AI Keras NN", layout="wide")

# -----------------------------
# File parsing helpers
# -----------------------------

def clean_col_name(c):
    return str(c).strip()


def find_header_row(uploaded_file, max_scan_rows=25):
    """Finds the row containing the real allocation headers in your Daily Allocation CSV exports."""
    uploaded_file.seek(0)
    raw = pd.read_csv(uploaded_file, header=None, nrows=max_scan_rows, dtype=str, low_memory=False)
    uploaded_file.seek(0)
    best_idx, best_score = 0, -1
    required_terms = ["Vendor", "Product ID", "Item", "Alloc. Rec.", "Flag", "Final Alloc."]
    for idx, row in raw.iterrows():
        vals = set(str(x).strip() for x in row.dropna().tolist())
        score = sum(term in vals for term in required_terms)
        if score > best_score:
            best_idx, best_score = idx, score
    return int(best_idx)


def read_allocation_file(uploaded_file):
    """Reads CSV/XLSX files formatted like the submitted Daily Allocation sheets."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        header_row = find_header_row(uploaded_file)
        df = pd.read_csv(uploaded_file, header=header_row, dtype=str, low_memory=False)
    elif name.endswith((".xlsx", ".xls")):
        # For Excel, scan the first sheet similarly.
        raw = pd.read_excel(uploaded_file, header=None, nrows=25, dtype=str)
        best_idx, best_score = 0, -1
        required_terms = ["Vendor", "Product ID", "Item", "Alloc. Rec.", "Flag", "Final Alloc."]
        for idx, row in raw.iterrows():
            vals = set(str(x).strip() for x in row.dropna().tolist())
            score = sum(term in vals for term in required_terms)
            if score > best_score:
                best_idx, best_score = idx, score
        uploaded_file.seek(0)
        df = pd.read_excel(uploaded_file, header=best_idx, dtype=str)
    else:
        raise ValueError("Please upload a CSV or XLSX allocation file.")

    df = df.dropna(how="all").copy()
    df.columns = [clean_col_name(c) for c in df.columns]
    # Remove fully unnamed trailing columns if they are completely empty.
    empty_unnamed = [c for c in df.columns if c.startswith("Unnamed") and df[c].isna().all()]
    if empty_unnamed:
        df = df.drop(columns=empty_unnamed)
    return df


def to_num(series):
    if series is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("%", "", regex=False)
        .replace({"nan": np.nan, "None": np.nan, "": np.nan, " ": np.nan}),
        errors="coerce",
    )


def get_col(df, name, default=""):
    if name in df.columns:
        return df[name]
    return pd.Series(default, index=df.index)

# -----------------------------
# Feature engineering; must match training
# -----------------------------

def derive_numeric_features(df, bundle):
    numeric_base = bundle["numeric_base"]
    out = pd.DataFrame(index=df.index)
    for c in numeric_base:
        out[c] = to_num(df[c]) if c in df.columns else np.nan

    flm = out.get("FLM.1", pd.Series(np.nan, index=df.index)).fillna(out.get("FLM", np.nan)).replace(0, np.nan)
    d60 = out.get("D60", pd.Series(np.nan, index=df.index))
    supply = out.get("Supply", pd.Series(np.nan, index=df.index))
    alloc_rec = out.get("Alloc. Rec.", pd.Series(np.nan, index=df.index))
    qoh = out.get("Qoh", pd.Series(np.nan, index=df.index))
    l30 = out.get("L30", pd.Series(np.nan, index=df.index))

    out["d60_minus_supply"] = d60 - supply
    out["proj_minus_supply"] = out.get("Proj. Demand", pd.Series(np.nan, index=df.index)) - supply
    out["alloc_rec_over_flm"] = alloc_rec / flm
    out["supply_over_d60"] = supply / d60.clip(lower=1)
    out["qoh_over_l30"] = qoh / l30.clip(lower=1)
    out["ttm_monthly"] = out.get("TTM", pd.Series(np.nan, index=df.index)) / 12.0
    out["recent_velocity"] = (
        l30.fillna(0) * 0.35
        + out.get("D30", pd.Series(0, index=df.index)).fillna(0) * 0.20
        + (d60.fillna(0) / 2) * 0.25
        + out.get("LW", pd.Series(0, index=df.index)).fillna(0) * 4.29 * 0.20
    )
    out["dc_avail_over_alloc_rec"] = out.get("Dc Avail", pd.Series(np.nan, index=df.index)) / alloc_rec.clip(lower=1)

    flag_upper = get_col(df, "Flag", "").fillna("").astype(str).str.upper()
    out["is_allocate"] = flag_upper.str.contains("ALLOC").astype(float)
    out["is_no_alloc_flag"] = flag_upper.str.contains("NO ALLOC|Z - NO", regex=True).astype(float)
    out["is_review"] = flag_upper.str.contains("REVIEW").astype(float)
    return out.replace([np.inf, -np.inf], np.nan)


def stable_bucket(text, hash_dim):
    digest = hashlib.blake2b(str(text).encode("utf-8", errors="ignore"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % hash_dim


def hash_categoricals(df, bundle):
    cat_cols = [c for c in bundle["categorical_base"] if c in df.columns]
    hash_dim = int(bundle["hash_dim"])
    n = len(df)
    mat = np.zeros((n, hash_dim), dtype=np.float32)
    rows = np.arange(n)
    for c in cat_cols:
        vals = get_col(df, c, "__MISSING__").fillna("__MISSING__").astype(str).str.strip().replace("", "__BLANK__").to_numpy()
        uniques, inv = np.unique(vals, return_inverse=True)
        buckets = np.array([stable_bucket(c + "=" + u, hash_dim) for u in uniques], dtype=np.int32)
        np.add.at(mat, (rows, buckets[inv]), 1.0)
    if cat_cols:
        mat /= np.sqrt(len(cat_cols))
    return mat


def build_model_matrix(df, bundle):
    prep = bundle["preprocessor"]
    numeric = derive_numeric_features(df, bundle).reindex(columns=prep["numeric_columns"])
    numeric_filled = numeric.fillna(prep["medians"])
    x_num = prep["scaler"].transform(numeric_filled).astype(np.float32)
    x_cat = hash_categoricals(df, bundle).astype(np.float32)
    return np.hstack([x_num, x_cat]).astype(np.float32)

# -----------------------------
# Model loading and prediction
# -----------------------------

@st.cache_resource(show_spinner=False)
def load_default_bundle_and_model():
    if KERAS_IMPORT_ERROR is not None:
        raise RuntimeError(f"Keras could not be imported: {KERAS_IMPORT_ERROR}")
    if not os.path.exists(DEFAULT_MODEL_PATH):
        raise FileNotFoundError(f"Missing {DEFAULT_MODEL_PATH}. Keep the .keras model in the same folder as app.py.")
    if not os.path.exists(DEFAULT_PREPROCESSOR_PATH):
        raise FileNotFoundError(f"Missing {DEFAULT_PREPROCESSOR_PATH}. Keep the preprocessor in the same folder as app.py.")
    bundle = joblib.load(DEFAULT_PREPROCESSOR_PATH)
    model = keras.models.load_model(DEFAULT_MODEL_PATH, compile=False)
    return bundle, model


def round_floor_bias(values, flm):
    values = np.asarray(values, dtype=float)
    flm = np.asarray(flm, dtype=float)
    flm = np.where(np.isfinite(flm) & (flm > 0), flm, 1.0)
    rounded = np.floor((values + 0.25 * flm) / flm) * flm
    return np.maximum(rounded, 0)


def postprocess_predictions(df, raw_pred):
    n = len(df)
    raw = np.nan_to_num(np.asarray(raw_pred, dtype=float).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    raw = np.clip(raw, 0, None)

    flm = to_num(get_col(df, "FLM.1", np.nan)).fillna(to_num(get_col(df, "FLM", np.nan))).fillna(1).replace(0, 1).to_numpy(float)
    clean = round_floor_bias(raw, flm)

    flag = get_col(df, "Flag", "").fillna("").astype(str).str.upper()
    alloc_rec = to_num(get_col(df, "Alloc. Rec.", 0)).fillna(0).to_numpy(float)
    supply = to_num(get_col(df, "Supply", 0)).fillna(0).to_numpy(float)
    d60 = to_num(get_col(df, "D60", np.nan)).to_numpy(float)
    dc_avail = to_num(get_col(df, "Dc Avail", np.nan)).to_numpy(float)

    no_alloc_flag = flag.str.contains("NO ALLOC|Z - NO", regex=True).to_numpy()
    blank_non_signal = ((flag.str.strip() == "") & (alloc_rec <= 0)).to_numpy()
    clean[no_alloc_flag | blank_non_signal] = 0

    # Demand safety cap: final supply should not exceed D60 by more than one FLM unless demand is missing.
    demand_cap_exists = np.isfinite(d60)
    max_by_demand = np.maximum(0, d60 + flm - supply)
    clean = np.where(demand_cap_exists, np.minimum(clean, round_floor_bias(max_by_demand, flm)), clean)

    # DC availability cap.
    dc_cap_exists = np.isfinite(dc_avail)
    clean = np.where(dc_cap_exists, np.minimum(clean, round_floor_bias(dc_avail, flm)), clean)

    # Very small allocations become zero.
    clean = np.where(clean < 0.5, 0, clean)

    reasons = []
    for i in range(n):
        r = []
        if no_alloc_flag[i]:
            r.append("No allocation flag")
        if blank_non_signal[i]:
            r.append("Blank/non-signal row")
        if demand_cap_exists[i] and raw[i] > max_by_demand[i]:
            r.append("Capped by D60 + one FLM")
        if dc_cap_exists[i] and raw[i] > dc_avail[i]:
            r.append("Capped by DC available")
        if abs(raw[i] - clean[i]) >= max(flm[i], 1):
            r.append("NN prediction adjusted by rules")
        reasons.append("; ".join(r) if r else "OK")

    confidence = np.where(np.abs(raw - clean) <= np.maximum(flm, 1), "High", "Medium")
    confidence = np.where(np.array([reason != "OK" for reason in reasons]), "Review", confidence)
    return raw, clean, confidence, reasons


def predict_file(df, bundle, model):
    X = build_model_matrix(df, bundle)
    pred = model.predict(X, batch_size=8192, verbose=0)
    raw, clean, confidence, reasons = postprocess_predictions(df, pred)

    out = df.copy()
    old_final = to_num(get_col(out, "Final Alloc.", np.nan))
    if "Final Alloc." not in out.columns:
        out["Final Alloc."] = ""

    # Write blanks for zero allocations because your corrected files generally use blank Final Alloc. for no allocation.
    final_as_object = pd.Series(clean, index=out.index).round(0).astype("Int64").astype(str)
    final_as_object = final_as_object.mask(clean <= 0, "")
    out["Final Alloc."] = final_as_object

    out["AI Raw NN Prediction"] = np.round(raw, 3)
    out["AI Clean Final Alloc."] = np.round(clean, 0).astype(int)
    out["AI Confidence"] = confidence
    out["AI Review Reason"] = reasons
    out["AI Change From Original"] = np.round(clean - old_final.fillna(0).to_numpy(float), 0).astype(int)
    out["AI Model Type"] = bundle.get("model_type", "keras_nn")
    out["AI Model Trained At"] = bundle.get("trained_at", "unknown")
    return out


def dataframe_to_xlsx_bytes(df):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Allocation AI Output")
        ws = writer.sheets["Allocation AI Output"]
        ws.freeze_panes(1, 0)
        for i, col in enumerate(df.columns):
            width = min(max(10, int(df[col].astype(str).str.len().quantile(0.90)) + 2), 40)
            ws.set_column(i, i, width)
    return buffer.getvalue()

# -----------------------------
# Optional fixed retraining, no UI settings
# -----------------------------

def assemble_training_from_uploads(files):
    frames = []
    for f in files:
        df = read_allocation_file(f)
        if "Final Alloc." not in df.columns or "Alloc. Rec." not in df.columns:
            continue
        y = to_num(df["Final Alloc."]).fillna(0).clip(lower=0)
        ar = to_num(df["Alloc. Rec."]).fillna(0).clip(lower=0)
        df = df.copy()
        df["_target_final_alloc"] = y
        signal = df[(y > 0) | ((y == 0) & (ar > 0))].copy()
        zero = df[(y == 0) & (ar == 0)].copy()
        if len(zero) > 5000:
            zero = zero.sample(n=5000, random_state=42)
        frames.append(pd.concat([signal, zero], ignore_index=True))
    if not frames:
        raise ValueError("No usable corrected training rows found. Make sure Final Alloc. and Alloc. Rec. columns are present.")
    return pd.concat(frames, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)


def retrain_from_uploads(files, bundle, base_model):
    train_df = assemble_training_from_uploads(files)
    y = train_df["_target_final_alloc"].astype(float).to_numpy(np.float32)
    X = build_model_matrix(train_df, bundle)
    weights = np.ones(len(y), dtype=np.float32)
    ar = to_num(get_col(train_df, "Alloc. Rec.", 0)).fillna(0).to_numpy(np.float32)
    weights[y > 0] = 8.0
    weights[(y == 0) & (ar > 0)] = 4.0

    # Fixed, no-hyperparameter continued training.
    base_model.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-4), loss=keras.losses.Huber(delta=2.0), metrics=["mae"])
    history = base_model.fit(X, y, sample_weight=weights, epochs=8, batch_size=2048, validation_split=0.15, verbose=0)
    bundle = dict(bundle)
    bundle["trained_at"] = datetime.now().isoformat(timespec="seconds")
    bundle["continued_training_rows"] = int(len(train_df))
    bundle["last_continue_training_mae"] = float(history.history.get("mae", [np.nan])[-1])
    return base_model, bundle


def keras_model_to_bytes(model):
    path = "updated_allocation_ai_keras_nn_model.keras"
    model.save(path)
    with open(path, "rb") as f:
        data = f.read()
    try:
        os.remove(path)
    except OSError:
        pass
    return data

# -----------------------------
# UI
# -----------------------------

st.title(APP_TITLE)
st.caption("Upload a Daily Allocation CSV/XLSX formatted like your submitted files. The app fills Final Alloc. using the included trained Keras neural network.")

try:
    bundle, model = load_default_bundle_and_model()
    model_ready = True
except Exception as exc:
    model_ready = False
    st.error(f"Model load failed: {exc}")

if model_ready:
    metrics = bundle.get("metrics", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", "Keras NN")
    c2.metric("Training rows", f"{metrics.get('training_rows', 0):,}")
    c3.metric("Validation MAE", f"{metrics.get('mae_raw', 0):.3f}")
    c4.metric("Source rows scanned", f"{metrics.get('all_source_rows_scanned', 0):,}")

predict_tab, train_tab, info_tab = st.tabs(["Fill Final Alloc.", "Continue Training", "Model Info"])

with predict_tab:
    st.subheader("Fill Final Alloc. from a Daily Allocation file")
    uploaded = st.file_uploader("Upload allocation CSV or XLSX", type=["csv", "xlsx", "xls"], key="predict_file")
    if uploaded and model_ready:
        try:
            df = read_allocation_file(uploaded)
            st.success(f"Parsed {len(df):,} rows and {len(df.columns):,} columns from {uploaded.name}.")
            missing = [c for c in ["Alloc. Rec.", "Flag", "Final Alloc."] if c not in df.columns]
            if missing:
                st.warning(f"Missing expected columns: {', '.join(missing)}. The app will still run if enough model features are present.")
            result = predict_file(df, bundle, model)

            changed = (result["AI Change From Original"] != 0).sum()
            review = (result["AI Confidence"] == "Review").sum()
            m1, m2, m3 = st.columns(3)
            m1.metric("Rows processed", f"{len(result):,}")
            m2.metric("Rows changed", f"{changed:,}")
            m3.metric("Review rows", f"{review:,}")

            st.dataframe(result.head(250), use_container_width=True)

            xlsx_bytes = dataframe_to_xlsx_bytes(result)
            csv_bytes = result.to_csv(index=False).encode("utf-8")
            base_name = uploaded.name.rsplit(".", 1)[0]
            st.download_button("Download edited XLSX", xlsx_bytes, file_name=f"{base_name}_AI_Final_Alloc.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            st.download_button("Download edited CSV", csv_bytes, file_name=f"{base_name}_AI_Final_Alloc.csv", mime="text/csv")
        except Exception as exc:
            st.exception(exc)

with train_tab:
    st.subheader("Continue training the included Keras NN")
    st.write("Upload one or more corrected allocation files. The app uses fixed internal training rules and does not expose model settings.")
    train_files = st.file_uploader("Upload corrected CSV/XLSX files", type=["csv", "xlsx", "xls"], accept_multiple_files=True, key="train_files")
    if train_files and model_ready:
        if st.button("Continue train neural network"):
            try:
                updated_model, updated_bundle = retrain_from_uploads(train_files, bundle, model)
                model_bytes = keras_model_to_bytes(updated_model)
                prep_bytes = io.BytesIO()
                joblib.dump(updated_bundle, prep_bytes, compress=3)
                st.success("Continued training completed. Download both files and replace the old model/preprocessor in your GitHub repo.")
                st.download_button("Download updated .keras model", model_bytes, file_name="allocation_ai_keras_nn_model.keras")
                st.download_button("Download updated preprocessor", prep_bytes.getvalue(), file_name="allocation_ai_keras_preprocessor.joblib")
            except Exception as exc:
                st.exception(exc)

with info_tab:
    st.subheader("Included neural network model")
    if model_ready:
        st.json({
            "model_type": bundle.get("model_type"),
            "trained_at": bundle.get("trained_at"),
            "metrics": bundle.get("metrics"),
            "source_files": bundle.get("source_files"),
            "postprocess": bundle.get("postprocess"),
        })
        st.write("The model avoids leakage columns such as Final Supply, Left DC, Final Cost, % UA, Left DC %, Final In Stock, and Demand Check.")
