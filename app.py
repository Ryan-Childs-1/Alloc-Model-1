# Allocation AI - Keras Neural Network Final Alloc. Filler
# Flat-file Streamlit app. Keep this file in the same folder as:
# allocation_ai_keras_nn_model.keras and allocation_ai_keras_preprocessor.joblib

import os
os.environ.setdefault("KERAS_BACKEND", "torch")
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
except Exception as exc:  # pragma: no cover
    keras = None
    KERAS_IMPORT_ERROR = exc
else:
    KERAS_IMPORT_ERROR = None

APP_TITLE = "Allocation AI — Keras NN Final Alloc. Filler"
DEFAULT_MODEL_PATH = "allocation_ai_keras_nn_model.keras"
DEFAULT_PREPROCESSOR_PATH = "allocation_ai_keras_preprocessor.joblib"
DEFAULT_METRICS_PATH = "training_metrics.json"
REQUIRED_HEADERS = ["Vendor", "Product ID", "Item", "Alloc. Rec.", "Flag", "Final Alloc."]
TRAILING_NOTE_COLS = {"3,039"}

st.set_page_config(page_title="Allocation AI Keras NN", layout="wide")

# -----------------------------
# Robust file parsing for submitted Daily Allocation CSV/XLSX format
# -----------------------------

def clean_col_name(c):
    s = str(c).strip()
    if s.lower() in {"nan", "none"}:
        return "Unnamed"
    return s


def dedupe_columns(cols):
    seen = {}
    out = []
    for col in cols:
        c = clean_col_name(col)
        if c == "":
            c = "Unnamed"
        if c in seen:
            seen[c] += 1
            out.append(f"{c}.{seen[c]}")
        else:
            seen[c] = 0
            out.append(c)
    return out


def find_header_row_from_csv(file_obj, max_scan_rows=35):
    file_obj.seek(0)
    raw = pd.read_csv(file_obj, header=None, nrows=max_scan_rows, dtype=str, low_memory=False)
    file_obj.seek(0)
    best_idx, best_score = 0, -1
    for idx, row in raw.iterrows():
        vals = set(str(x).strip() for x in row.dropna().tolist())
        score = sum(h in vals for h in REQUIRED_HEADERS)
        if "Vendor" in vals and "Vendor Site Id" in vals:
            score += 2
        if score > best_score:
            best_idx, best_score = int(idx), int(score)
    if best_score < 4:
        raise ValueError(
            "Could not find the Daily Allocation header row. Expected row with headers like "
            "Vendor, Product ID, Item, Alloc. Rec., Flag, and Final Alloc."
        )
    return best_idx


def find_header_row_from_excel(file_obj, max_scan_rows=35):
    file_obj.seek(0)
    raw = pd.read_excel(file_obj, header=None, nrows=max_scan_rows, dtype=str)
    file_obj.seek(0)
    best_idx, best_score = 0, -1
    for idx, row in raw.iterrows():
        vals = set(str(x).strip() for x in row.dropna().tolist())
        score = sum(h in vals for h in REQUIRED_HEADERS)
        if "Vendor" in vals and "Vendor Site Id" in vals:
            score += 2
        if score > best_score:
            best_idx, best_score = int(idx), int(score)
    if best_score < 4:
        raise ValueError(
            "Could not find the Daily Allocation header row. Expected row with headers like "
            "Vendor, Product ID, Item, Alloc. Rec., Flag, and Final Alloc."
        )
    return best_idx


def drop_non_table_tail_columns(df):
    """Drop trailing exported-note/helper columns while preserving the real allocation table.

    In the submitted files, the actual table ends at Demand Check. After that, some exports
    include columns like Helper, Unnamed: 78, 581, or 3,039. Those are worksheet notes or
    helper artifacts, not allocation-model inputs. Cutting at Demand Check makes the parser
    deterministic across every submitted CSV variant.
    """
    cols = list(df.columns)
    if "Demand Check" in cols:
        return df.loc[:, cols[: cols.index("Demand Check") + 1]].copy()

    drop = []
    for c in cols:
        cs = str(c).strip()
        if cs.startswith("Unnamed") or cs in TRAILING_NOTE_COLS or cs in {"Helper", "581"}:
            drop.append(c)
    return df.drop(columns=drop, errors="ignore")


def read_allocation_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        header_row = find_header_row_from_csv(uploaded_file)
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, header=header_row, dtype=str, low_memory=False)
    elif name.endswith((".xlsx", ".xls")):
        header_row = find_header_row_from_excel(uploaded_file)
        uploaded_file.seek(0)
        df = pd.read_excel(uploaded_file, header=header_row, dtype=str)
    else:
        raise ValueError("Please upload a CSV or XLSX Daily Allocation file.")

    df = df.dropna(how="all").copy()
    df.columns = dedupe_columns(df.columns)
    df = drop_non_table_tail_columns(df)

    missing = [h for h in ["Alloc. Rec.", "Flag"] if h not in df.columns]
    if missing:
        raise ValueError(f"The file is missing required column(s): {', '.join(missing)}")
    if "Final Alloc." not in df.columns:
        df["Final Alloc."] = ""
    return df


def to_num(series):
    if series is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace("(", "-", regex=False)
        .str.replace(")", "", regex=False)
        .replace({"nan": np.nan, "None": np.nan, "": np.nan, " ": np.nan, "-": np.nan}),
        errors="coerce",
    )


def get_col(df, name, default=""):
    if name in df.columns:
        return df[name]
    return pd.Series(default, index=df.index)

# -----------------------------
# Feature engineering; exactly aligned to the trained preprocessor
# -----------------------------

def derive_numeric_features(df, bundle):
    numeric_base = bundle["numeric_base"]
    out = pd.DataFrame(index=df.index)
    for c in numeric_base:
        out[c] = to_num(df[c]) if c in df.columns else np.nan

    flm = out.get("FLM.1", pd.Series(np.nan, index=df.index)).fillna(out.get("FLM", np.nan)).replace(0, np.nan)
    mil = out.get("MIL.1", pd.Series(np.nan, index=df.index)).fillna(out.get("MIL", np.nan))
    d60 = out.get("D60", pd.Series(np.nan, index=df.index))
    d30 = out.get("D30", pd.Series(np.nan, index=df.index))
    l30 = out.get("L30", pd.Series(np.nan, index=df.index))
    lw = out.get("LW", pd.Series(np.nan, index=df.index))
    ttm = out.get("TTM", pd.Series(np.nan, index=df.index))
    supply = out.get("Supply", pd.Series(np.nan, index=df.index))
    qoh = out.get("Qoh", pd.Series(np.nan, index=df.index))
    alloc_rec = out.get("Alloc. Rec.", pd.Series(np.nan, index=df.index))

    out["effective_flm"] = flm
    out["effective_mil"] = mil
    out["d60_minus_supply"] = d60 - supply
    out["d30_minus_supply"] = d30 - supply
    out["l30_minus_supply"] = l30 - supply
    out["proj_minus_supply"] = out.get("Proj. Demand", pd.Series(np.nan, index=df.index)) - supply
    out["alloc_rec_over_flm"] = alloc_rec / flm
    out["alloc_rec_minus_gap"] = alloc_rec - (d60 - supply)
    out["supply_over_d60"] = supply / d60.clip(lower=1)
    out["qoh_over_l30"] = qoh / l30.clip(lower=1)
    out["ttm_monthly"] = ttm / 12.0
    out["lw_monthly_runrate"] = lw * 4.29
    out["recent_velocity"] = (
        l30.fillna(0) * 0.35
        + d30.fillna(0) * 0.20
        + (d60.fillna(0) / 2) * 0.25
        + (lw.fillna(0) * 4.29) * 0.20
    )
    out["velocity_minus_supply"] = out["recent_velocity"] - supply
    out["dc_avail_over_alloc_rec"] = out.get("Dc Avail", pd.Series(np.nan, index=df.index)) / alloc_rec.clip(lower=1)
    out["dc_avail_over_flm"] = out.get("Dc Avail", pd.Series(np.nan, index=df.index)) / flm.clip(lower=1)
    pipeline_cols = ["Allocated", "Intrans", "Store Transfer", "QTY Reserve", "Store PO Qty"]
    out["store_pipeline"] = out[[c for c in pipeline_cols if c in out.columns]].sum(axis=1, min_count=1)
    out["available_store_supply"] = qoh + out["store_pipeline"].fillna(0)
    out["d60_gap_after_pipeline"] = d60 - out["available_store_supply"]

    flag = get_col(df, "Flag", "").fillna("").astype(str).str.upper()
    out["is_allocate"] = flag.str.fullmatch("ALLOCATE").fillna(False).astype(float)
    out["is_review"] = flag.str.contains("REVIEW", regex=False).fillna(False).astype(float)
    out["is_z_no_alloc"] = flag.str.contains("NO ALLOC", regex=False).fillna(False).astype(float)
    return out.replace([np.inf, -np.inf], np.nan)


def stable_bucket(text, hash_dim):
    digest = hashlib.blake2b(str(text).encode("utf-8", errors="ignore"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % int(hash_dim)


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
    prep_numeric_cols = bundle["numeric_columns"]
    numeric = derive_numeric_features(df, bundle).reindex(columns=prep_numeric_cols)
    medians = bundle["medians"]
    numeric_filled = numeric.fillna(medians).fillna(0)
    x_num = bundle["scaler"].transform(numeric_filled).astype(np.float32)
    x_cat = hash_categoricals(df, bundle).astype(np.float32)
    return np.hstack([x_num, x_cat]).astype(np.float32)

# -----------------------------
# Model loading and post-processing
# -----------------------------

@st.cache_resource(show_spinner=False)
def load_default_model_and_preprocessor():
    if KERAS_IMPORT_ERROR is not None:
        raise RuntimeError(f"Keras could not be imported: {KERAS_IMPORT_ERROR}")
    if not os.path.exists(DEFAULT_MODEL_PATH):
        raise FileNotFoundError(f"Missing {DEFAULT_MODEL_PATH}. Put it in the same folder as app.py.")
    if not os.path.exists(DEFAULT_PREPROCESSOR_PATH):
        raise FileNotFoundError(f"Missing {DEFAULT_PREPROCESSOR_PATH}. Put it in the same folder as app.py.")
    bundle = joblib.load(DEFAULT_PREPROCESSOR_PATH)
    model = keras.models.load_model(DEFAULT_MODEL_PATH, compile=False)
    return bundle, model


def floor_to_flm(values, flm):
    values = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    flm = np.nan_to_num(np.asarray(flm, dtype=float), nan=1.0, posinf=1.0, neginf=1.0)
    flm = np.where(flm > 0, flm, 1.0)
    return np.maximum(np.floor((values + 0.25 * flm) / flm) * flm, 0)


def postprocess_predictions(df, raw_pred):
    raw = np.nan_to_num(np.asarray(raw_pred, dtype=float).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    raw = np.clip(raw, 0, None)

    flm = to_num(get_col(df, "FLM.1", np.nan)).fillna(to_num(get_col(df, "FLM", np.nan))).fillna(1).replace(0, 1).to_numpy(float)
    alloc_rec = to_num(get_col(df, "Alloc. Rec.", 0)).fillna(0).to_numpy(float)
    supply = to_num(get_col(df, "Supply", 0)).fillna(0).to_numpy(float)
    d60 = to_num(get_col(df, "D60", np.nan)).to_numpy(float)
    dc_avail = to_num(get_col(df, "Dc Avail", np.nan)).to_numpy(float)
    flag = get_col(df, "Flag", "").fillna("").astype(str).str.upper()

    clean = floor_to_flm(raw, flm)

    no_alloc = flag.str.contains("NO ALLOC|Z - NO", regex=True).to_numpy()
    signal = flag.str.contains("ALLOCATE|REVIEW", regex=True).to_numpy() | (alloc_rec > 0)
    clean[no_alloc | ~signal] = 0

    # Demand cap: do not allow final supply to exceed D60 by more than 1 FLM when D60 exists.
    demand_cap_exists = np.isfinite(d60)
    max_by_demand = np.maximum(0, d60 + flm - supply)
    clean = np.where(demand_cap_exists, np.minimum(clean, floor_to_flm(max_by_demand, flm)), clean)

    # DC cap: do not recommend more than available DC by row when Dc Avail exists.
    dc_cap_exists = np.isfinite(dc_avail)
    clean = np.where(dc_cap_exists, np.minimum(clean, floor_to_flm(dc_avail, flm)), clean)

    reasons = []
    for i in range(len(df)):
        parts = []
        if no_alloc[i]:
            parts.append("No-allocation flag")
        if not signal[i]:
            parts.append("No allocation signal")
        if demand_cap_exists[i] and raw[i] > max_by_demand[i]:
            parts.append("Reduced by D60 + one FLM cap")
        if dc_cap_exists[i] and raw[i] > dc_avail[i]:
            parts.append("Reduced by DC availability cap")
        if clean[i] == 0 and signal[i] and not no_alloc[i]:
            parts.append("Model/rules recommend zero")
        reasons.append("; ".join(parts) if parts else "OK")

    return raw, clean, reasons


def predict_final_alloc(df, bundle, model):
    X = build_model_matrix(df, bundle)
    log_pred = model.predict(X, batch_size=4096, verbose=0)
    if isinstance(log_pred, dict):
        log_pred = list(log_pred.values())[0]
    if isinstance(log_pred, (list, tuple)):
        log_pred = log_pred[0]
    raw_pred = np.expm1(np.maximum(np.asarray(log_pred).reshape(-1), 0))
    raw, clean, reasons = postprocess_predictions(df, raw_pred)

    result = df.copy()
    result["AI Original Final Alloc."] = result["Final Alloc."] if "Final Alloc." in result.columns else ""
    result["AI Raw NN Prediction"] = np.round(raw, 3)
    result["AI Clean Final Alloc."] = clean.astype(int)
    result["AI Review Reason"] = reasons
    result["AI Model Version"] = bundle.get("version", "unknown")

    # Put final answer into the actual business column. Blank non-signal rows instead of filling zeros everywhere.
    flag = get_col(result, "Flag", "").fillna("").astype(str).str.upper()
    alloc_rec = to_num(get_col(result, "Alloc. Rec.", 0)).fillna(0)
    signal = flag.str.contains("ALLOCATE|REVIEW", regex=True) | (alloc_rec > 0)
    final_values = pd.Series(clean.astype(int), index=result.index).astype(object)
    final_values[~signal] = ""
    result["Final Alloc."] = final_values
    return result

# -----------------------------
# Export helpers with NaN-safe width logic
# -----------------------------

def dataframe_to_xlsx_bytes(df, sheet_name="Allocation AI Output"):
    output = io.BytesIO()
    safe_df = df.copy()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        safe_df.to_excel(writer, index=False, sheet_name=sheet_name)
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]
        header_fmt = workbook.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1})
        green_fmt = workbook.add_format({"bg_color": "#E2F0D9"})
        yellow_fmt = workbook.add_format({"bg_color": "#FFF2CC"})
        for col_idx, col in enumerate(safe_df.columns):
            worksheet.write(0, col_idx, col, header_fmt)
            # Fixed: all-blank columns can produce NaN quantiles. Convert to string, fill blanks, and guard.
            lengths = safe_df[col].astype("string").fillna("").str.len()
            q = lengths.quantile(0.90) if len(lengths) else np.nan
            if pd.isna(q) or not np.isfinite(float(q)):
                width = 12
            else:
                width = min(max(10, int(float(q)) + 2), 42)
            worksheet.set_column(col_idx, col_idx, width)
            if col == "Final Alloc.":
                worksheet.set_column(col_idx, col_idx, max(width, 14), green_fmt)
            if str(col).startswith("AI "):
                worksheet.set_column(col_idx, col_idx, max(width, 18), yellow_fmt)
        worksheet.freeze_panes(1, 0)
        worksheet.autofilter(0, 0, len(safe_df), max(0, len(safe_df.columns)-1))
    return output.getvalue()


def dataframe_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")

# -----------------------------
# UI
# -----------------------------

st.title(APP_TITLE)
st.caption("Upload a Daily Allocation CSV/XLSX. The included Keras neural net fills the Final Alloc. column and adds audit columns.")

try:
    bundle, model = load_default_model_and_preprocessor()
    model_ready = True
except Exception as exc:
    model_ready = False
    st.error(str(exc))
    bundle, model = None, None

if model_ready:
    m = bundle.get("metrics", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Source rows seen", f"{m.get('full_source_rows_seen', 0):,}")
    c2.metric("Signal training rows", f"{m.get('signal_training_rows', 0):,}")
    c3.metric("Validation MAE", f"{m.get('validation_mae_raw_units', 0):.2f}")
    c4.metric("Model features", f"{m.get('feature_count', 0):,}")

    tab_predict, tab_info = st.tabs(["Fill Final Alloc.", "Model Info"])

    with tab_predict:
        uploaded = st.file_uploader("Upload Daily Allocation CSV or XLSX", type=["csv", "xlsx", "xls"])
        if uploaded is not None:
            try:
                df = read_allocation_file(uploaded)
                st.success(f"Parsed {len(df):,} rows and {len(df.columns):,} columns.")
                st.dataframe(df.head(25), use_container_width=True)

                if st.button("Run Keras Neural Network and Fill Final Alloc.", type="primary"):
                    with st.spinner("Running neural network predictions and applying allocation safety rules..."):
                        result = predict_final_alloc(df, bundle, model)
                        xlsx_bytes = dataframe_to_xlsx_bytes(result)
                        csv_bytes = dataframe_to_csv_bytes(result)

                    st.success("Final Alloc. has been filled. Download the edited file below.")
                    st.dataframe(result.head(50), use_container_width=True)

                    base = os.path.splitext(uploaded.name)[0].replace(" ", "_")
                    st.download_button(
                        "Download edited XLSX",
                        data=xlsx_bytes,
                        file_name=f"{base}_AI_Final_Alloc.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                    st.download_button(
                        "Download edited CSV",
                        data=csv_bytes,
                        file_name=f"{base}_AI_Final_Alloc.csv",
                        mime="text/csv",
                    )

                    reason_counts = result["AI Review Reason"].value_counts(dropna=False).head(20)
                    st.subheader("Audit Summary")
                    st.dataframe(reason_counts.rename_axis("Reason").reset_index(name="Rows"), use_container_width=True)
            except Exception as exc:
                st.exception(exc)

    with tab_info:
        st.subheader("Included Model")
        st.json(bundle.get("metrics", {}))
        st.markdown(
            """
            **Design:** The neural network is trained on the decision-signal rows from all submitted files: rows marked
            `Allocate`, `Review`, rows with a positive `Alloc. Rec.`, and rows with a positive historical `Final Alloc.`.
            It then applies strict business-rule cleanup so blank/no-allocation rows are not filled accidentally.
            """
        )
