"""
Allocation AI — Streamlit App
Creators: Ryan Childs / Allocation AI workflow

A flat-file Streamlit app that:
1. Uploads Daily Allocation spreadsheets formatted like the provided CSV/XLSX files.
2. Trains a machine learning model to learn corrected Final Alloc. values.
3. Saves/loads model bundles with preprocessing, metadata, and optional training memory.
4. Predicts Final Alloc. on new sheets and exports an edited file plus audit columns.

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import io
import json
import math
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

try:
    import plotly.express as px
except Exception:  # pragma: no cover
    px = None

APP_NAME = "Allocation AI"
MODEL_VERSION = "allocation_ai_v1.0"
RANDOM_STATE = 42

# -----------------------------
# Column configuration
# -----------------------------

TARGET_COL = "Final Alloc."
ALLOC_REC_COL = "Alloc. Rec."
FLAG_COL = "Flag"
FINAL_SUPPLY_COL = "Final Supply"
DEMAND_CHECK_COL = "Demand Check"

REQUIRED_HEADER_HINTS = [
    "Vendor", "Item", "Site", "L30", "D60", "FLM", "Alloc. Rec.", "Flag", "Final Alloc."
]

LEAKAGE_COLUMNS = {
    "Final Alloc.",
    "Left DC",
    "Final Supply",
    "Final Cost",
    "% UA",
    "Left DC %",
    "Final In Stock",
    "Demand Check",
    "AI Predicted Final Alloc.",
    "AI Raw ML Prediction",
    "AI Clean Final Alloc.",
    "AI Confidence",
    "AI Review Reason",
    "AI Prediction Mode",
    "AI Change From Original",
}

PREFERRED_BASE_FEATURES = [
    # Product/store identity
    "Vendor", "Vendor Site Id", "Brand", "Dcl", "Department Id", "Class Id", "Class Name",
    "Line Id", "Line Name", "Product ID", "Pcode Description", "Color", "Size", "Item",
    "Description", "Mfg Code", "UPC", "Status", "Status 300", "Site", "Site Name", "State",
    "Square Footage", "Region", "Zone", "Buyer Name", "Planner Code", "Private Label",
    "Season Code", "Store Size", "Rank",
    # Demand and velocity
    "L30", "D30", "D60", "LW", "TTM", "Avg. WOC", "Proj. Demand", "Days",
    # Inventory and allocation drivers
    "MIL", "FLM", "DC FLM", "Orgs", "Cost", "Retail", "ATG Retail", "GM Pct",
    "Qoh", "Supply", "Allocated", "Intrans", "Store Transfer", "QTY Reserve", "Store PO Qty",
    "Dc Qoh", "Dc Avail", "DC Staged", "DC RV", "DC PO QTY", "Supply In Stock",
    "MIL.1", "FLM.1", "Alloc. Rec.", "New", "Store Flag", "SKU Flag", "Flag",
]

NUMERIC_HINTS = {
    "Department Id", "Class Id", "Line Id", "Site", "Square Footage", "MIL", "FLM", "DC FLM", "Orgs",
    "Cost", "Retail", "ATG Retail", "GM Pct", "L30", "D30", "D60", "LW", "TTM", "Qoh",
    "Supply", "Allocated", "Intrans", "Store Transfer", "QTY Reserve", "Store PO Qty", "Dc Qoh",
    "Dc Avail", "DC Staged", "DC RV", "DC PO QTY", "Rank", "Supply In Stock", "Avg. WOC",
    "MIL.1", "FLM.1", "Days", "Proj. Demand", "Alloc. Rec.", "Final Alloc."
}

CATEGORICAL_HINTS = {
    "Vendor", "Vendor Site Id", "Brand", "Dcl", "Class Name", "Line Name", "Product ID", "Pcode Description",
    "Color", "Size", "Item", "Description", "Mfg Code", "UPC", "Status", "Status 300", "Site Name",
    "State", "Region", "Zone", "Buyer Name", "Planner Code", "Private Label", "Season Code", "Store Size",
    "New", "Store Flag", "SKU Flag", "Flag"
}


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class ParsedSheet:
    df: pd.DataFrame
    original_raw: pd.DataFrame
    header_row_idx: int
    source_name: str
    file_type: str


@dataclass
class TrainResult:
    bundle: Dict[str, Any]
    metrics: Dict[str, Any]
    test_predictions: pd.DataFrame


# -----------------------------
# Generic helpers
# -----------------------------

def normalize_col_name(col: Any) -> str:
    """Normalize spreadsheet column names while preserving business-readable names."""
    if col is None or (isinstance(col, float) and pd.isna(col)):
        return ""
    s = str(col).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def make_unique_columns(cols: Sequence[Any]) -> List[str]:
    seen: Dict[str, int] = {}
    out: List[str] = []
    for i, col in enumerate(cols):
        base = normalize_col_name(col)
        if not base or base.lower().startswith("unnamed"):
            base = f"Unnamed: {i}"
        if base in seen:
            seen[base] += 1
            out.append(f"{base}.{seen[base]}")
        else:
            seen[base] = 0
            out.append(base)
    return out


def clean_numeric_series(s: pd.Series) -> pd.Series:
    """Convert strings like '$1,234', '15%', blanks, and Excel-ish text to numeric."""
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    txt = s.astype(str).str.strip()
    txt = txt.replace({"": np.nan, "nan": np.nan, "None": np.nan, "NULL": np.nan})
    txt = txt.str.replace(r"[,$]", "", regex=True)
    pct = txt.str.endswith("%", na=False)
    txt = txt.str.replace("%", "", regex=False)
    out = pd.to_numeric(txt, errors="coerce")
    out.loc[pct] = out.loc[pct] / 100.0
    return out


def upper_clean(s: Any) -> str:
    if pd.isna(s):
        return ""
    return str(s).strip().upper()


def safe_divide(a: pd.Series, b: pd.Series, default: float = 0.0) -> pd.Series:
    b2 = b.replace(0, np.nan)
    result = a / b2
    return result.replace([np.inf, -np.inf], np.nan).fillna(default)


def round_to_flm(value: float, flm: float, mode: str = "floor") -> float:
    """Round a value to a valid FLM multiple."""
    if pd.isna(value):
        return 0.0
    try:
        value = float(value)
    except Exception:
        return 0.0
    if value <= 0:
        return 0.0
    try:
        flm = float(flm)
    except Exception:
        flm = 1.0
    if pd.isna(flm) or flm <= 0:
        flm = 1.0
    if mode == "ceil":
        return float(math.ceil(value / flm) * flm)
    if mode == "nearest":
        return float(round(value / flm) * flm)
    return float(math.floor(value / flm) * flm)


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y_%m_%d_%H%M")


# -----------------------------
# File loading and header detection
# -----------------------------

def read_uploaded_file(uploaded_file: Any, max_preview_rows: Optional[int] = None) -> Tuple[pd.DataFrame, str]:
    """Read CSV/XLSX into raw DataFrame with no header assumption."""
    name = getattr(uploaded_file, "name", "uploaded_file")
    suffix = name.lower().split(".")[-1]
    data = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
    bio = io.BytesIO(data)

    if suffix in {"xlsx", "xlsm", "xls"}:
        raw = pd.read_excel(bio, header=None, dtype=object, nrows=max_preview_rows, engine=None)
        return raw, suffix

    # Robust CSV read: try utf-8-sig, then latin1.
    for enc in ["utf-8-sig", "utf-8", "latin1"]:
        try:
            raw = pd.read_csv(
                io.BytesIO(data),
                header=None,
                dtype=object,
                nrows=max_preview_rows,
                low_memory=False,
                encoding=enc,
            )
            return raw, "csv"
        except UnicodeDecodeError:
            continue
    raw = pd.read_csv(io.BytesIO(data), header=None, dtype=object, nrows=max_preview_rows, low_memory=False)
    return raw, "csv"


def detect_header_row(raw: pd.DataFrame) -> int:
    """Find the row that contains allocation headers. Works for your provided Daily Allocation CSV exports."""
    best_idx = 0
    best_score = -1
    scan_rows = min(len(raw), 50)
    hints = {h.upper() for h in REQUIRED_HEADER_HINTS}

    for idx in range(scan_rows):
        values = [upper_clean(v) for v in raw.iloc[idx].tolist()]
        value_set = set(values)
        score = 0
        for h in hints:
            if h in value_set:
                score += 4
            else:
                # Partial match for headers like FLM.1, spacing differences, etc.
                score += sum(1 for v in values if v == h or v.startswith(h + "."))
        # Bonus for dense row of text headers.
        score += sum(1 for v in values if v and not re.fullmatch(r"[-+]?\d+(\.\d+)?", v)) / 20
        if score > best_score:
            best_score = score
            best_idx = idx

    return int(best_idx)


def parse_allocation_file(uploaded_file: Any) -> ParsedSheet:
    raw, file_type = read_uploaded_file(uploaded_file)
    header_idx = detect_header_row(raw)
    cols = make_unique_columns(raw.iloc[header_idx].tolist())
    df = raw.iloc[header_idx + 1 :].copy()
    df.columns = cols
    df = df.dropna(how="all").reset_index(drop=True)

    # Remove obvious subtotal/footer rows with almost no real row content.
    if "Vendor" in df.columns:
        df = df[~df["Vendor"].astype(str).str.contains("grand total|total result|subtotal", case=False, na=False)].copy()

    # Normalize empty strings.
    df = df.replace(r"^\s*$", np.nan, regex=True)
    return ParsedSheet(df=df, original_raw=raw, header_row_idx=header_idx, source_name=uploaded_file.name, file_type=file_type)


def parse_many(files: Sequence[Any]) -> List[ParsedSheet]:
    parsed = []
    for f in files:
        try:
            parsed.append(parse_allocation_file(f))
        except Exception as exc:
            st.error(f"Could not parse {getattr(f, 'name', 'file')}: {exc}")
    return parsed


# -----------------------------
# Feature engineering
# -----------------------------

def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = make_unique_columns(df.columns)
    for c in df.columns:
        if c in NUMERIC_HINTS or c in {TARGET_COL, ALLOC_REC_COL}:
            df[c] = clean_numeric_series(df[c])
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_dataframe(df)

    def num(col: str, default: float = 0.0) -> pd.Series:
        if col in df.columns:
            return clean_numeric_series(df[col]).fillna(default)
        return pd.Series(default, index=df.index, dtype=float)

    def cat(col: str) -> pd.Series:
        if col in df.columns:
            return df[col].astype(str).fillna("").str.strip()
        return pd.Series("", index=df.index)

    l30 = num("L30")
    d30 = num("D30")
    d60 = num("D60")
    lw = num("LW")
    ttm = num("TTM")
    qoh = num("Qoh")
    supply = num("Supply")
    alloc_rec = num(ALLOC_REC_COL)
    flm = num("FLM", 1).replace(0, 1)
    dc_avail = num("Dc Avail")
    dc_qoh = num("Dc Qoh")
    proj_demand = num("Proj. Demand")
    days = num("Days", 30).replace(0, 30)

    df["FE_flag_allocate"] = cat(FLAG_COL).str.upper().str.contains("ALLOC", na=False).astype(int)
    df["FE_flag_review"] = cat(FLAG_COL).str.upper().str.contains("REVIEW", na=False).astype(int)
    df["FE_flag_blank"] = (cat(FLAG_COL).str.strip() == "").astype(int)

    df["FE_recent_velocity"] = (l30 * 0.40) + ((d30 / 1.0) * 0.20) + ((d60 / 2.0) * 0.25) + (lw * 4.29 * 0.15)
    df["FE_monthly_ttm_rate"] = ttm / 12.0
    df["FE_projected_demand_gap"] = proj_demand - supply
    df["FE_d60_gap"] = d60 - supply
    df["FE_l30_gap"] = l30 - qoh
    df["FE_alloc_rec_units"] = safe_divide(alloc_rec, flm)
    df["FE_supply_to_d60"] = safe_divide(supply, d60.clip(lower=1))
    df["FE_qoh_to_l30"] = safe_divide(qoh, l30.clip(lower=1))
    df["FE_alloc_rec_to_d60_gap"] = safe_divide(alloc_rec, (d60 - supply).clip(lower=1))
    df["FE_dc_avail_to_rec"] = safe_divide(dc_avail, alloc_rec.clip(lower=1))
    df["FE_dc_avail_to_qoh"] = safe_divide(dc_avail, dc_qoh.clip(lower=1))
    df["FE_days_scaled_demand"] = df["FE_recent_velocity"] * safe_divide(days, pd.Series(30.0, index=df.index), 1.0)
    df["FE_is_new_item"] = cat("New").str.upper().isin(["Y", "YES", "TRUE", "1", "NEW"]).astype(int)
    df["FE_has_store_flag"] = (cat("Store Flag").str.len() > 0).astype(int)
    df["FE_has_sku_flag"] = (cat("SKU Flag").str.len() > 0).astype(int)

    # Adjustment target helper is created only when target exists.
    if TARGET_COL in df.columns:
        final_alloc = clean_numeric_series(df[TARGET_COL])
        df["FE_manual_adjustment"] = final_alloc - alloc_rec

    return df.replace([np.inf, -np.inf], np.nan)


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    engineered = [c for c in df.columns if c.startswith("FE_") and c != "FE_manual_adjustment"]
    base = [c for c in PREFERRED_BASE_FEATURES if c in df.columns and c not in LEAKAGE_COLUMNS]

    # Keep useful unknown columns if they are not obvious leakage/helper leftovers.
    unknown = []
    for c in df.columns:
        if c in base or c in engineered or c in LEAKAGE_COLUMNS or c == "FE_manual_adjustment":
            continue
        if str(c).startswith("Unnamed"):
            continue
        if str(c).strip() == "":
            continue
        unknown.append(c)

    # Limit unknowns to avoid accidental noisy exports, but retain meaningful uploaded columns.
    return base + engineered + unknown[:20]


def split_feature_types(df: pd.DataFrame, feature_cols: Sequence[str]) -> Tuple[List[str], List[str]]:
    numeric_cols, categorical_cols = [], []
    for c in feature_cols:
        if c in NUMERIC_HINTS or c.startswith("FE_"):
            numeric_cols.append(c)
        else:
            # If mostly numeric, treat as numeric.
            numeric_version = clean_numeric_series(df[c]) if c in df.columns else pd.Series(dtype=float)
            valid = numeric_version.notna().mean() if len(numeric_version) else 0
            if valid > 0.85:
                numeric_cols.append(c)
            else:
                categorical_cols.append(c)
    return numeric_cols, categorical_cols


def make_preprocessor(df: pd.DataFrame, feature_cols: Sequence[str]) -> Tuple[ColumnTransformer, List[str], List[str]]:
    numeric_cols, categorical_cols = split_feature_types(df, feature_cols)

    # FunctionTransformer ensures numeric-ish object columns are converted during predict too.
    numeric_pipe = Pipeline(
        steps=[
            ("to_numeric", FunctionTransformer(lambda x: pd.DataFrame(x).apply(pd.to_numeric, errors="coerce"), validate=False)),
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor, numeric_cols, categorical_cols


def build_model(model_choice: str) -> Any:
    if model_choice == "Random Forest":
        return RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=3,
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    if model_choice == "Extra Trees":
        return ExtraTreesRegressor(
            n_estimators=350,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    return HistGradientBoostingRegressor(
        max_iter=250,
        learning_rate=0.055,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        random_state=RANDOM_STATE,
    )


def prepare_training_data(parsed_files: Sequence[ParsedSheet], target_mode: str) -> Tuple[pd.DataFrame, pd.Series, List[str], pd.DataFrame]:
    frames = []
    for p in parsed_files:
        df = add_engineered_features(p.df)
        df["FE_source_file"] = p.source_name
        frames.append(df)
    if not frames:
        raise ValueError("No parseable training files were uploaded.")

    full = pd.concat(frames, ignore_index=True, sort=False)
    if TARGET_COL not in full.columns:
        raise ValueError(f"Training files must contain '{TARGET_COL}'.")

    full[TARGET_COL] = clean_numeric_series(full[TARGET_COL])
    full[ALLOC_REC_COL] = clean_numeric_series(full[ALLOC_REC_COL]) if ALLOC_REC_COL in full.columns else 0

    # Train only rows that have a known final allocation or where flag/alloc rec gives meaningful no-allocation signal.
    has_target = full[TARGET_COL].notna()
    has_alloc_context = full.get(FLAG_COL, pd.Series("", index=full.index)).astype(str).str.len().gt(0) | full[ALLOC_REC_COL].fillna(0).ne(0)
    train_df = full[has_target | has_alloc_context].copy()
    train_df[TARGET_COL] = train_df[TARGET_COL].fillna(0)

    if target_mode == "Manual Adjustment from Alloc. Rec.":
        y = train_df[TARGET_COL] - train_df[ALLOC_REC_COL].fillna(0)
    else:
        y = train_df[TARGET_COL]

    feature_cols = select_feature_columns(train_df)
    # Remove target helper columns after selection.
    feature_cols = [c for c in feature_cols if c not in {TARGET_COL, "FE_manual_adjustment"} and c in train_df.columns]
    if len(feature_cols) < 5:
        raise ValueError("Not enough usable feature columns were found after parsing the files.")

    return train_df, y.astype(float), feature_cols, full


def train_model(
    parsed_files: Sequence[ParsedSheet],
    model_choice: str,
    target_mode: str,
    embed_training_memory: bool = True,
) -> TrainResult:
    train_df, y, feature_cols, full_memory = prepare_training_data(parsed_files, target_mode)
    X = train_df[feature_cols].copy()

    preprocessor, numeric_cols, categorical_cols = make_preprocessor(train_df, feature_cols)
    model = build_model(model_choice)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    # Prefer group split by Product ID/Item when possible to reduce over-optimistic tests.
    test_size = 0.20 if len(train_df) >= 50 else 0.30
    group_col = "Product ID" if "Product ID" in train_df.columns else ("Item" if "Item" in train_df.columns else None)
    if group_col and train_df[group_col].nunique(dropna=True) > 5 and len(train_df) >= 100:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
        train_idx, test_idx = next(splitter.split(X, y, groups=train_df[group_col].astype(str)))
    else:
        train_idx, test_idx = train_test_split(np.arange(len(train_df)), test_size=test_size, random_state=RANDOM_STATE)

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    # Fit final model on all data after validation.
    final_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", build_model(model_choice))])
    final_pipeline.fit(X, y)

    mae = float(mean_absolute_error(y_test, pred)) if len(y_test) else np.nan
    rmse = float(mean_squared_error(y_test, pred) ** 0.5) if len(y_test) else np.nan
    r2 = float(r2_score(y_test, pred)) if len(y_test) > 1 else np.nan

    metrics = {
        "rows_used": int(len(train_df)),
        "files_used": int(len(parsed_files)),
        "feature_count": int(len(feature_cols)),
        "numeric_feature_count": int(len(numeric_cols)),
        "categorical_feature_count": int(len(categorical_cols)),
        "target_mode": target_mode,
        "model_choice": model_choice,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }

    test_out = train_df.iloc[test_idx].copy()
    test_out["AI_test_actual_target"] = y_test.values
    test_out["AI_test_predicted_target"] = pred
    test_out["AI_test_abs_error"] = np.abs(y_test.values - pred)

    memory_payload = None
    if embed_training_memory:
        # Store compact standardized training rows in the model bundle so the user can retrain with new files later.
        # This makes the app portable on Streamlit Cloud where local disk is ephemeral.
        keep_cols = sorted(set(feature_cols + [TARGET_COL, ALLOC_REC_COL, FLAG_COL, "FE_source_file"]).intersection(full_memory.columns))
        memory_df = full_memory[keep_cols].copy()
        # Avoid enormous model bundles. If huge, sample while keeping all corrected/nonzero rows.
        max_memory_rows = 75000
        if len(memory_df) > max_memory_rows:
            important = memory_df[TARGET_COL].notna() | clean_numeric_series(memory_df.get(ALLOC_REC_COL, pd.Series(0, index=memory_df.index))).fillna(0).ne(0)
            memory_df = pd.concat([
                memory_df[important],
                memory_df[~important].sample(min(max_memory_rows // 5, (~important).sum()), random_state=RANDOM_STATE),
            ], ignore_index=True).head(max_memory_rows)
        memory_payload = memory_df.to_json(orient="split", date_format="iso")

    bundle = {
        "model_version": MODEL_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "pipeline": final_pipeline,
        "feature_columns": feature_cols,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "target_column": TARGET_COL,
        "target_mode": target_mode,
        "model_choice": model_choice,
        "metrics": metrics,
        "training_files": [p.source_name for p in parsed_files],
        "postprocessing_defaults": {
            "prediction_mode": "Balanced",
            "rounding_mode": "floor",
            "max_final_supply_over_d60_flm": 1.0,
            "respect_blank_flags": True,
            "cap_by_dc_avail": True,
        },
        "training_memory_json": memory_payload,
    }

    return TrainResult(bundle=bundle, metrics=metrics, test_predictions=test_out)


# -----------------------------
# Prediction and postprocessing
# -----------------------------

def load_model_bundle(uploaded_model: Any) -> Dict[str, Any]:
    data = uploaded_model.getvalue() if hasattr(uploaded_model, "getvalue") else uploaded_model.read()
    return joblib.load(io.BytesIO(data))


def model_to_bytes(bundle: Dict[str, Any]) -> bytes:
    bio = io.BytesIO()
    joblib.dump(bundle, bio, compress=3)
    bio.seek(0)
    return bio.getvalue()


def apply_postprocessing(
    df: pd.DataFrame,
    raw_prediction: np.ndarray,
    bundle: Dict[str, Any],
    prediction_mode: str,
    rounding_mode: str,
    max_over_d60_flm: float,
    respect_blank_flags: bool,
    cap_by_dc_avail: bool,
) -> pd.DataFrame:
    out = df.copy()

    alloc_rec = clean_numeric_series(out[ALLOC_REC_COL]) if ALLOC_REC_COL in out.columns else pd.Series(0, index=out.index)
    flm = clean_numeric_series(out["FLM"]) if "FLM" in out.columns else pd.Series(1, index=out.index)
    flm = flm.fillna(1).replace(0, 1)
    supply = clean_numeric_series(out["Supply"]) if "Supply" in out.columns else pd.Series(0, index=out.index)
    d60 = clean_numeric_series(out["D60"]) if "D60" in out.columns else pd.Series(np.nan, index=out.index)
    dc_avail = clean_numeric_series(out["Dc Avail"]) if "Dc Avail" in out.columns else pd.Series(np.nan, index=out.index)
    flags = out[FLAG_COL].astype(str).fillna("") if FLAG_COL in out.columns else pd.Series("", index=out.index)

    target_mode = bundle.get("target_mode", "Manual Adjustment from Alloc. Rec.")
    if target_mode == "Manual Adjustment from Alloc. Rec.":
        proposed = alloc_rec.fillna(0).values + raw_prediction
    else:
        proposed = raw_prediction

    # Mode tuning: conservative pulls toward lower of ML and Alloc Rec.; aggressive gives more room.
    if prediction_mode == "Conservative":
        proposed = np.minimum(proposed, alloc_rec.fillna(0).values)
        max_over_d60_flm = min(max_over_d60_flm, 0.75)
        rounding_mode_effective = "floor"
    elif prediction_mode == "Aggressive":
        proposed = np.maximum(proposed, raw_prediction if target_mode != "Manual Adjustment from Alloc. Rec." else proposed)
        max_over_d60_flm = max(max_over_d60_flm, 1.5)
        rounding_mode_effective = rounding_mode
    else:
        rounding_mode_effective = rounding_mode

    clean_vals: List[float] = []
    reasons: List[str] = []
    confidence: List[str] = []

    for i, val in enumerate(proposed):
        row_reasons = []
        row_flag = upper_clean(flags.iloc[i])
        row_flm = flm.iloc[i]
        row_supply = supply.iloc[i] if not pd.isna(supply.iloc[i]) else 0
        row_d60 = d60.iloc[i]
        row_dc = dc_avail.iloc[i]
        row_alloc_rec = alloc_rec.iloc[i] if not pd.isna(alloc_rec.iloc[i]) else 0

        value = max(float(val) if not pd.isna(val) else 0.0, 0.0)

        if respect_blank_flags and ("ALLOC" not in row_flag and "REVIEW" not in row_flag) and row_alloc_rec <= 0:
            value = 0.0
            row_reasons.append("Blank/non-allocation flag with no allocation recommendation")

        # Demand cap: final supply should not exceed D60 by more than N FLMs unless no D60 is available.
        if not pd.isna(row_d60) and row_d60 >= 0:
            cap = max(0.0, float(row_d60) + (float(max_over_d60_flm) * float(row_flm)) - float(row_supply))
            if value > cap:
                value = cap
                row_reasons.append(f"Capped so final supply stays near D60 + {max_over_d60_flm:g} FLM")

        if cap_by_dc_avail and not pd.isna(row_dc) and row_dc >= 0 and value > row_dc:
            value = float(row_dc)
            row_reasons.append("Capped by DC available")

        rounded = round_to_flm(value, row_flm, rounding_mode_effective)
        if abs(rounded - value) > 1e-9:
            row_reasons.append(f"Rounded by FLM using {rounding_mode_effective}")
        value = rounded

        # Risk/review tags.
        if row_alloc_rec > 0 and abs(value - row_alloc_rec) > 2 * max(float(row_flm), 1.0):
            row_reasons.append("Large change from Alloc. Rec.; review")
        if not pd.isna(row_d60) and (row_supply + value) > row_d60 + row_flm:
            row_reasons.append("Final supply still exceeds D60 by more than one FLM")
        if "REVIEW" in row_flag:
            row_reasons.append("Original row marked Review")

        # Basic confidence proxy using rule interventions and seen feature availability.
        if any("review" in r.lower() or "exceeds" in r.lower() for r in row_reasons):
            conf = "Low"
        elif len(row_reasons) >= 2:
            conf = "Medium"
        else:
            conf = "High"

        clean_vals.append(value)
        reasons.append("; ".join(dict.fromkeys(row_reasons)) if row_reasons else "Model prediction accepted")
        confidence.append(conf)

    out["AI Raw ML Prediction"] = raw_prediction
    out["AI Predicted Final Alloc."] = proposed
    out["AI Clean Final Alloc."] = clean_vals
    old_final = clean_numeric_series(out[TARGET_COL]) if TARGET_COL in out.columns else pd.Series(np.nan, index=out.index)
    out["AI Change From Original"] = pd.Series(clean_vals, index=out.index) - old_final.fillna(0)
    out["AI Confidence"] = confidence
    out["AI Review Reason"] = reasons
    out["AI Prediction Mode"] = prediction_mode

    # Overwrite/insert Final Alloc. with cleaned value. Keep true blanks for non-alloc rows if zero and no context.
    out[TARGET_COL] = clean_vals
    if respect_blank_flags and FLAG_COL in out.columns:
        no_alloc_mask = (~out[FLAG_COL].astype(str).str.upper().str.contains("ALLOC|REVIEW", na=False)) & (alloc_rec.fillna(0) <= 0) & (out[TARGET_COL] == 0)
        out.loc[no_alloc_mask, TARGET_COL] = np.nan

    return out


def predict_with_bundle(
    parsed: ParsedSheet,
    bundle: Dict[str, Any],
    prediction_mode: str,
    rounding_mode: str,
    max_over_d60_flm: float,
    respect_blank_flags: bool,
    cap_by_dc_avail: bool,
) -> pd.DataFrame:
    df = add_engineered_features(parsed.df)
    feature_cols = bundle["feature_columns"]
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan
    X = df[feature_cols]
    raw_pred = bundle["pipeline"].predict(X)
    return apply_postprocessing(
        df=parsed.df.copy(),
        raw_prediction=raw_pred,
        bundle=bundle,
        prediction_mode=prediction_mode,
        rounding_mode=rounding_mode,
        max_over_d60_flm=max_over_d60_flm,
        respect_blank_flags=respect_blank_flags,
        cap_by_dc_avail=cap_by_dc_avail,
    )


# -----------------------------
# Export helpers
# -----------------------------

def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def dataframe_to_xlsx_bytes(df: pd.DataFrame, sheet_name: str = "Allocation AI Output") -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name[:31])
        wb = writer.book
        ws = writer.sheets[sheet_name[:31]]
        header_fmt = wb.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1})
        final_fmt = wb.add_format({"bg_color": "#E2F0D9", "border": 1})
        review_fmt = wb.add_format({"bg_color": "#FFF2CC", "border": 1})
        num_fmt = wb.add_format({"num_format": "#,##0.00"})
        int_fmt = wb.add_format({"num_format": "#,##0"})
        for col_idx, col in enumerate(df.columns):
            ws.write(0, col_idx, col, header_fmt)
            width = min(max(10, int(df[col].astype(str).str.len().quantile(0.90) if len(df) else len(col)) + 2), 38)
            if col in {"Description", "Pcode Description", "AI Review Reason"}:
                width = 34
            ws.set_column(col_idx, col_idx, width)
            if col == TARGET_COL:
                ws.set_column(col_idx, col_idx, 14, final_fmt)
            elif col.startswith("AI "):
                ws.set_column(col_idx, col_idx, max(width, 16), review_fmt)
            elif pd.api.types.is_numeric_dtype(df[col]):
                ws.set_column(col_idx, col_idx, width, int_fmt if col in {TARGET_COL, "AI Clean Final Alloc."} else num_fmt)
        ws.freeze_panes(1, 0)
        ws.autofilter(0, 0, max(len(df), 1), max(len(df.columns) - 1, 0))
    bio.seek(0)
    return bio.getvalue()


def create_zip(files: Dict[str, bytes]) -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            # Force flat archive: strip folders if any.
            z.writestr(name.split("/")[-1], data)
    bio.seek(0)
    return bio.getvalue()


def metrics_frame(metrics: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for k, v in metrics.items():
        rows.append({"Metric": k, "Value": v})
    return pd.DataFrame(rows)


def feature_importance_frame(bundle: Dict[str, Any]) -> pd.DataFrame:
    pipeline = bundle.get("pipeline")
    if pipeline is None:
        return pd.DataFrame()
    model = pipeline.named_steps.get("model")
    pre = pipeline.named_steps.get("preprocessor")
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame({"Note": ["This model type does not expose built-in feature importances. Use Random Forest or Extra Trees for importances."]})
    try:
        names = pre.get_feature_names_out()
        vals = model.feature_importances_
        return pd.DataFrame({"Feature": names, "Importance": vals}).sort_values("Importance", ascending=False).head(50)
    except Exception as exc:
        return pd.DataFrame({"Note": [f"Could not extract feature importances: {exc}"]})


def restore_training_memory(bundle: Dict[str, Any]) -> Optional[pd.DataFrame]:
    payload = bundle.get("training_memory_json")
    if not payload:
        return None
    try:
        return pd.read_json(io.StringIO(payload), orient="split")
    except Exception:
        return None


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title=APP_NAME, page_icon="📦", layout="wide")

st.title("📦 Allocation AI")
st.caption("Train, save, reuse, and continually improve a model that fills the `Final Alloc.` column from Daily Allocation spreadsheets.")

with st.sidebar:
    st.header("Model Safety Settings")
    prediction_mode = st.selectbox("Prediction mode", ["Conservative", "Balanced", "Aggressive"], index=1)
    rounding_mode = st.selectbox("FLM rounding", ["floor", "nearest", "ceil"], index=0)
    max_over_d60_flm = st.slider("Max final supply over D60, in FLMs", 0.0, 3.0, 1.0, 0.25)
    respect_blank_flags = st.checkbox("Respect blank / non-allocation rows", value=True)
    cap_by_dc_avail = st.checkbox("Cap by DC available", value=True)
    st.divider()
    st.markdown("**Recommended daily mode:** Balanced + floor rounding + D60 cap of 1 FLM.")

main_tabs = st.tabs([
    "1) Predict Final Alloc.",
    "2) Train Model",
    "3) Continue Training",
    "4) Diagnostics / Model Info",
    "5) Help",
])

with main_tabs[0]:
    st.subheader("Predict `Final Alloc.` on a New Daily Allocation File")
    c1, c2 = st.columns(2)
    with c1:
        model_file = st.file_uploader("Upload saved model bundle (`.joblib`)", type=["joblib"], key="predict_model")
    with c2:
        predict_file = st.file_uploader("Upload allocation file to edit", type=["csv", "xlsx", "xlsm", "xls"], key="predict_sheet")

    if model_file and predict_file:
        try:
            bundle = load_model_bundle(model_file)
            parsed = parse_allocation_file(predict_file)
            st.success(f"Parsed `{predict_file.name}` with {len(parsed.df):,} rows. Header row detected at raw row {parsed.header_row_idx + 1}.")

            with st.spinner("Running allocation model and applying business rules..."):
                output_df = predict_with_bundle(
                    parsed=parsed,
                    bundle=bundle,
                    prediction_mode=prediction_mode,
                    rounding_mode=rounding_mode,
                    max_over_d60_flm=max_over_d60_flm,
                    respect_blank_flags=respect_blank_flags,
                    cap_by_dc_avail=cap_by_dc_avail,
                )

            k1, k2, k3, k4 = st.columns(4)
            final_numeric = clean_numeric_series(output_df[TARGET_COL]) if TARGET_COL in output_df.columns else pd.Series(dtype=float)
            k1.metric("Rows", f"{len(output_df):,}")
            k2.metric("Total AI Final Alloc.", f"{final_numeric.fillna(0).sum():,.0f}")
            k3.metric("Low Confidence Rows", f"{(output_df['AI Confidence'] == 'Low').sum():,}")
            changed = clean_numeric_series(output_df["AI Change From Original"]).fillna(0).ne(0).sum()
            k4.metric("Changed Rows", f"{changed:,}")

            st.markdown("#### Preview of AI allocation output")
            preview_cols = [c for c in ["Vendor", "Brand", "Class Name", "Item", "Site", "D60", "Supply", "FLM", "Alloc. Rec.", TARGET_COL, "AI Confidence", "AI Review Reason"] if c in output_df.columns]
            st.dataframe(output_df[preview_cols].head(500), use_container_width=True)

            if px is not None and "AI Confidence" in output_df.columns:
                chart_df = output_df["AI Confidence"].value_counts().reset_index()
                chart_df.columns = ["Confidence", "Rows"]
                st.plotly_chart(px.bar(chart_df, x="Confidence", y="Rows", title="Prediction Confidence Distribution"), use_container_width=True)

            base = re.sub(r"\.[^.]+$", "", predict_file.name)
            xlsx_bytes = dataframe_to_xlsx_bytes(output_df)
            csv_bytes = dataframe_to_csv_bytes(output_df)
            audit_cols = [c for c in output_df.columns if c.startswith("AI ") or c in ["Vendor", "Brand", "Class Name", "Item", "Site", "D60", "Supply", "FLM", "Alloc. Rec.", TARGET_COL, FLAG_COL]]
            audit_bytes = dataframe_to_xlsx_bytes(output_df[audit_cols], sheet_name="AI Audit")

            dl1, dl2, dl3 = st.columns(3)
            dl1.download_button("⬇️ Download edited XLSX", xlsx_bytes, file_name=f"{base}_allocation_ai_output.xlsx")
            dl2.download_button("⬇️ Download edited CSV", csv_bytes, file_name=f"{base}_allocation_ai_output.csv")
            dl3.download_button("⬇️ Download audit XLSX", audit_bytes, file_name=f"{base}_allocation_ai_audit.xlsx")
        except Exception as exc:
            st.exception(exc)
    else:
        st.info("Upload a saved model and a daily allocation file to produce a completed `Final Alloc.` output.")

with main_tabs[1]:
    st.subheader("Train a New Model from Corrected Historical Files")
    train_files = st.file_uploader(
        "Upload corrected Daily Allocation files where `Final Alloc.` is already the desired answer",
        type=["csv", "xlsx", "xlsm", "xls"],
        accept_multiple_files=True,
        key="train_files",
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        model_choice = st.selectbox("Model type", ["HistGradientBoosting", "Random Forest", "Extra Trees"], index=0, key="train_model_type")
    with c2:
        target_mode = st.selectbox("Training target", ["Manual Adjustment from Alloc. Rec.", "Final Alloc. Directly"], index=0, key="target_mode")
    with c3:
        embed_memory = st.checkbox("Embed compact training memory in model", value=True)

    if train_files and st.button("🚀 Train Allocation Model", type="primary"):
        parsed_files = parse_many(train_files)
        if parsed_files:
            try:
                with st.spinner("Training model and validating performance..."):
                    result = train_model(parsed_files, model_choice, target_mode, embed_memory)
                st.success("Model trained successfully.")
                st.dataframe(metrics_frame(result.metrics), use_container_width=True)

                st.markdown("#### Validation sample with largest errors")
                err_cols = [c for c in ["FE_source_file", "Vendor", "Brand", "Class Name", "Item", "Site", "D60", "Supply", "Alloc. Rec.", TARGET_COL, "AI_test_actual_target", "AI_test_predicted_target", "AI_test_abs_error"] if c in result.test_predictions.columns]
                st.dataframe(result.test_predictions.sort_values("AI_test_abs_error", ascending=False)[err_cols].head(100), use_container_width=True)

                st.markdown("#### Feature importance")
                st.dataframe(feature_importance_frame(result.bundle), use_container_width=True)

                model_bytes = model_to_bytes(result.bundle)
                st.download_button(
                    "⬇️ Download trained model bundle",
                    model_bytes,
                    file_name=f"allocation_ai_model_{timestamp_slug()}.joblib",
                    mime="application/octet-stream",
                )
            except Exception as exc:
                st.exception(exc)
    else:
        st.info("Upload several corrected historical allocation files, then train. The more reviewed/corrected examples you provide, the better the model gets.")

with main_tabs[2]:
    st.subheader("Continue Training / Retrain with New Corrected Files")
    st.write("This app retrains from the uploaded model's embedded training memory plus any newly corrected files you upload. This is safer than pretending tree models can incrementally learn row-by-row.")
    c1, c2 = st.columns(2)
    with c1:
        existing_model = st.file_uploader("Upload existing model bundle", type=["joblib"], key="continue_model")
    with c2:
        new_corrected_files = st.file_uploader("Upload newly corrected files", type=["csv", "xlsx", "xlsm", "xls"], accept_multiple_files=True, key="continue_files")

    cont_model_choice = st.selectbox("Updated model type", ["HistGradientBoosting", "Random Forest", "Extra Trees"], index=0, key="continue_model_type")
    cont_target_mode = st.selectbox("Updated target", ["Manual Adjustment from Alloc. Rec.", "Final Alloc. Directly"], index=0, key="continue_target")

    if existing_model and new_corrected_files and st.button("🔁 Retrain Updated Model", type="primary"):
        try:
            old_bundle = load_model_bundle(existing_model)
            memory_df = restore_training_memory(old_bundle)
            parsed_new = parse_many(new_corrected_files)

            pseudo_files: List[ParsedSheet] = []
            if memory_df is not None and len(memory_df):
                pseudo_files.append(ParsedSheet(df=memory_df, original_raw=memory_df, header_row_idx=0, source_name="embedded_training_memory", file_type="memory"))
            pseudo_files.extend(parsed_new)

            if not pseudo_files:
                st.error("The uploaded model did not contain training memory, and no new corrected files could be parsed.")
            else:
                with st.spinner("Retraining updated model..."):
                    updated = train_model(pseudo_files, cont_model_choice, cont_target_mode, embed_training_memory=True)
                st.success("Updated model trained successfully.")
                st.markdown("#### New model metrics")
                st.dataframe(metrics_frame(updated.metrics), use_container_width=True)

                if old_bundle.get("metrics"):
                    st.markdown("#### Old vs New Metrics")
                    old_m = old_bundle.get("metrics", {})
                    compare = pd.DataFrame([
                        {"Metric": "MAE", "Old": old_m.get("mae"), "New": updated.metrics.get("mae")},
                        {"Metric": "RMSE", "Old": old_m.get("rmse"), "New": updated.metrics.get("rmse")},
                        {"Metric": "Rows Used", "Old": old_m.get("rows_used"), "New": updated.metrics.get("rows_used")},
                        {"Metric": "Feature Count", "Old": old_m.get("feature_count"), "New": updated.metrics.get("feature_count")},
                    ])
                    st.dataframe(compare, use_container_width=True)

                st.download_button(
                    "⬇️ Download updated model bundle",
                    model_to_bytes(updated.bundle),
                    file_name=f"allocation_ai_model_updated_{timestamp_slug()}.joblib",
                    mime="application/octet-stream",
                )
        except Exception as exc:
            st.exception(exc)
    else:
        st.info("For best continual learning, train your first model with 'Embed compact training memory' enabled.")

with main_tabs[3]:
    st.subheader("Diagnostics / Model Info")
    inspect_model = st.file_uploader("Upload a model bundle to inspect", type=["joblib"], key="inspect_model")
    if inspect_model:
        try:
            b = load_model_bundle(inspect_model)
            c1, c2, c3, c4 = st.columns(4)
            m = b.get("metrics", {})
            c1.metric("Version", b.get("model_version", "Unknown"))
            c2.metric("Rows Used", f"{m.get('rows_used', 0):,}")
            c3.metric("MAE", f"{m.get('mae', np.nan):.3f}" if m.get("mae") is not None else "n/a")
            c4.metric("Features", f"{len(b.get('feature_columns', [])):,}")

            st.markdown("#### Model metadata")
            st.json({k: v for k, v in b.items() if k not in {"pipeline", "training_memory_json"}}, expanded=False)

            st.markdown("#### Feature columns")
            st.dataframe(pd.DataFrame({"Feature": b.get("feature_columns", [])}), use_container_width=True)

            st.markdown("#### Feature importance")
            st.dataframe(feature_importance_frame(b), use_container_width=True)
        except Exception as exc:
            st.exception(exc)
    else:
        st.info("Upload a `.joblib` model bundle to inspect training metrics, model configuration, and feature columns.")

with main_tabs[4]:
    st.subheader("How to Use This App")
    st.markdown(
        """
### Recommended workflow

1. **Train** using several old Daily Allocation files where you already corrected `Final Alloc.`.
2. Download the `.joblib` model bundle.
3. Go to **Predict Final Alloc.** and upload the model plus a new allocation file.
4. Download the edited file and review the `AI Confidence` and `AI Review Reason` columns.
5. After you manually correct the output, upload that corrected file to **Continue Training**.
6. Download the updated model and use it next time.

### Best modeling strategy

The default target is **Manual Adjustment from Alloc. Rec.**. That means the model learns how your manual decisions differ from the spreadsheet's existing allocation recommendation. This is usually better than predicting `Final Alloc.` from scratch because your existing formula already contains valuable business logic.

### Built-in safety rules

The model output is cleaned after prediction:

- Negative allocations become zero.
- Allocations are rounded to valid `FLM` multiples.
- Rows with blank/non-allocation flags can remain blank.
- Final supply can be capped near `D60 + N × FLM`.
- Allocations can be capped by `Dc Avail`.
- Large changes and risky rows are marked for review.

### GitHub / Streamlit Cloud setup

Put these flat files in one GitHub folder:

```text
app.py
requirements.txt
README.md
```

Then deploy the repo through Streamlit Cloud and set the entry point to:

```text
app.py
```
        """
    )
