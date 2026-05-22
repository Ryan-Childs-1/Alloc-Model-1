"""
Allocation AI — Streamlit flat-file app

A production-style, single-file Streamlit app for Daily Allocation CSV/XLSX files
formatted like Ryan's submitted Sportsman's Warehouse allocation exports.

Main workflow:
1) Train from corrected historical Daily Allocation files.
2) Download a portable .joblib model bundle.
3) Upload the model + a new Daily Allocation CSV/XLSX.
4) Populate/overwrite the "Final Alloc." column and download the edited file.
5) Continue training with new corrected files.
"""

from __future__ import annotations

import io
import math
import os
import re
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

try:
    import plotly.express as px
except Exception:
    px = None

APP_NAME = "Allocation AI"
MODEL_VERSION = "allocation_ai_v2_exact_daily_allocation_csv"
RANDOM_STATE = 42

TARGET_COL = "Final Alloc."
ALLOC_REC_COL = "Alloc. Rec."
FLAG_COL = "Flag"

# Headers from the submitted CSVs. The files have a blank first row and the true
# headers on row index 1. The detector below still scans to avoid hard failure.
HEADER_HINTS = [
    "Vendor", "Vendor Site Id", "Brand", "Dcl", "Department Id", "Class Id", "Class Name",
    "Line Id", "Line Name", "Product ID", "Item", "Site", "L30", "D30", "D60", "Qoh",
    "Supply", "Dc Avail", "FLM", "Days", "Proj. Demand", "Alloc. Rec.", "Flag", "Final Alloc."
]

NUMERIC_HINTS = {
    "Department Id", "Class Id", "Line Id", "Site", "Square Footage", "MIL", "MIL.1", "FLM", "FLM.1",
    "DC FLM", "Orgs", "Cost", "Retail", "ATG Retail", "GM Pct", "L30", "D30", "D60", "LW", "TTM",
    "Qoh", "Supply", "Allocated", "Intrans", "Store Transfer", "QTY Reserve", "Store PO Qty", "Dc Qoh",
    "Dc Avail", "DC Staged", "DC RV", "DC PO QTY", "Rank", "Supply In Stock", "Avg. WOC", "Days",
    "Proj. Demand", "Alloc. Rec.", "Final Alloc.", "Left DC", "Final Supply", "Final Cost", "% UA",
    "Left DC %", "Final In Stock", "Demand Check"
}

PREFERRED_FEATURES = [
    # Identity / grouping
    "Vendor", "Vendor Site Id", "Brand", "Dcl", "Department Id", "Class Id", "Class Name", "Line Id", "Line Name",
    "Product ID", "Pcode Description", "Color", "Size", "Item", "Description", "Mfg Code", "UPC", "Status", "Status 300",
    "Site", "Site Name", "State", "Square Footage", "Region", "Zone", "Buyer Name", "Planner Code", "Private Label",
    "Season Code", "Store Size", "Rank",
    # Demand / inventory / allocation inputs
    "MIL", "FLM", "DC FLM", "Orgs", "Cost", "Retail", "ATG Retail", "GM Pct", "L30", "D30", "D60", "LW", "TTM",
    "Qoh", "Supply", "Allocated", "Intrans", "Store Transfer", "QTY Reserve", "Store PO Qty", "Dc Qoh", "Dc Avail",
    "DC Staged", "DC RV", "DC PO QTY", "Supply In Stock", "Avg. WOC", "MIL.1", "FLM.1", "Days", "Proj. Demand",
    "Alloc. Rec.", "New", "Store Flag", "SKU Flag", "Flag",
]

# Columns that occur after or because of Final Alloc.; never train on them.
LEAKAGE_COLS = {
    "Final Alloc.", "Left DC", "Final Supply", "Final Cost", "% UA", "Left DC %", "Final In Stock", "Demand Check",
    "AI Raw ML Prediction", "AI Proposed Final Alloc.", "AI Clean Final Alloc.", "AI Confidence", "AI Review Reason",
    "AI Change From Original", "AI Prediction Mode", "AI Flag Class", "AI Model Version"
}

@dataclass
class ParsedSheet:
    df: pd.DataFrame
    raw: pd.DataFrame
    header_row_idx: int
    source_name: str
    file_type: str

@dataclass
class TrainResult:
    bundle: Dict[str, Any]
    metrics: Dict[str, Any]
    validation_rows: pd.DataFrame

# -----------------------------
# Parsing helpers
# -----------------------------

def normalize_col_name(x: Any) -> str:
    if x is None or pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def make_unique_columns(cols: Sequence[Any]) -> List[str]:
    seen: Dict[str, int] = {}
    out: List[str] = []
    for i, col in enumerate(cols):
        base = normalize_col_name(col)
        if not base or base.lower().startswith("unnamed") or base.lower() == "nan":
            base = f"Unnamed: {i}"
        if base in seen:
            seen[base] += 1
            out.append(f"{base}.{seen[base]}")
        else:
            seen[base] = 0
            out.append(base)
    return out


def read_any_file(uploaded_file: Any, nrows: Optional[int] = None) -> Tuple[pd.DataFrame, str]:
    name = getattr(uploaded_file, "name", "uploaded_file")
    suffix = name.lower().split(".")[-1]
    data = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
    bio = io.BytesIO(data)

    if suffix in {"xlsx", "xlsm", "xls"}:
        return pd.read_excel(bio, header=None, dtype=object, nrows=nrows), suffix

    for enc in ("utf-8-sig", "utf-8", "latin1"):
        try:
            raw = pd.read_csv(io.BytesIO(data), header=None, dtype=object, nrows=nrows, low_memory=False, encoding=enc)
            return raw, "csv"
        except UnicodeDecodeError:
            continue
    return pd.read_csv(io.BytesIO(data), header=None, dtype=object, nrows=nrows, low_memory=False), "csv"


def upper_clean(x: Any) -> str:
    if x is None or pd.isna(x):
        return ""
    return str(x).strip().upper()


def detect_header_row(raw: pd.DataFrame) -> int:
    # Ryan's submitted files: row 0 blank, row 1 actual headers.
    # Still score the first 50 rows for safety.
    best_idx, best_score = 0, -1
    hint_set = {h.upper() for h in HEADER_HINTS}
    for idx in range(min(len(raw), 50)):
        values = [upper_clean(v) for v in raw.iloc[idx].tolist()]
        value_set = set(values)
        exact = sum(1 for h in hint_set if h in value_set)
        partial = 0
        for h in hint_set:
            if h not in value_set:
                partial += sum(1 for v in values if v.startswith(h + "."))
        density = sum(1 for v in values if v and not re.fullmatch(r"[-+]?\d+(\.\d+)?", v)) / 50.0
        score = exact * 5 + partial + density
        if score > best_score:
            best_idx, best_score = idx, score
    return int(best_idx)


def parse_allocation_file(uploaded_file: Any) -> ParsedSheet:
    """Parse Ryan's Daily Allocation exports efficiently.

    The submitted CSVs are large (~214k rows) and have a blank first row with
    true headers on the second row. We detect the header from a 50-row preview,
    then reread with that row as the header. This is much faster than loading
    the full file as headerless object data.
    """
    preview, file_type = read_any_file(uploaded_file, nrows=50)
    header_idx = detect_header_row(preview)

    name = getattr(uploaded_file, "name", "uploaded_file")
    suffix = name.lower().split(".")[-1]
    data = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()

    if suffix in {"xlsx", "xlsm", "xls"}:
        df = pd.read_excel(io.BytesIO(data), header=header_idx, dtype=object)
        file_type = suffix
    else:
        last_err = None
        for enc in ("utf-8-sig", "utf-8", "latin1"):
            try:
                df = pd.read_csv(io.BytesIO(data), header=header_idx, dtype=object, low_memory=False, encoding=enc)
                file_type = "csv"
                break
            except UnicodeDecodeError as exc:
                last_err = exc
        else:
            if last_err:
                raise last_err
            df = pd.read_csv(io.BytesIO(data), header=header_idx, dtype=object, low_memory=False)
            file_type = "csv"

    df.columns = make_unique_columns(df.columns)
    # Avoid whole-dataframe regex replacement on 200k+ row exports;
    # downstream numeric/category cleaners handle blanks safely.
    df = df.dropna(how="all").reset_index(drop=True)

    if "Vendor" in df.columns:
        vendor_txt = df["Vendor"].astype(str)
        df = df[~vendor_txt.str.contains(r"grand total|total result|subtotal", case=False, na=False)].copy()

    return ParsedSheet(df=df, raw=preview, header_row_idx=header_idx, source_name=name, file_type=file_type)

def parse_many(files: Sequence[Any]) -> List[ParsedSheet]:
    parsed: List[ParsedSheet] = []
    for f in files:
        try:
            parsed.append(parse_allocation_file(f))
        except Exception as exc:
            st.error(f"Could not parse {getattr(f, 'name', 'file')}: {exc}")
    return parsed

# -----------------------------
# Cleaning / features
# -----------------------------

def clean_numeric_series(s: Any) -> pd.Series:
    if not isinstance(s, pd.Series):
        s = pd.Series(s)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    txt = s.astype(str).str.strip()
    txt = txt.replace({"": np.nan, "nan": np.nan, "None": np.nan, "NULL": np.nan, "NaN": np.nan})
    txt = txt.str.replace(r"[,$]", "", regex=True)
    pct_mask = txt.str.endswith("%", na=False)
    txt = txt.str.replace("%", "", regex=False)
    out = pd.to_numeric(txt, errors="coerce").astype(float)
    out.loc[pct_mask] = out.loc[pct_mask] / 100.0
    return out


def safe_num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return clean_numeric_series(df[col]).replace([np.inf, -np.inf], np.nan).fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def safe_cat(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return df[col].astype(str).replace({"nan": "", "None": ""}).fillna("").str.strip()
    return pd.Series("", index=df.index, dtype=object)


def safe_div(a: pd.Series, b: pd.Series, default: float = 0.0) -> pd.Series:
    b = b.replace(0, np.nan)
    out = a / b
    return out.replace([np.inf, -np.inf], np.nan).fillna(default)


def flag_class_series(flag: pd.Series) -> pd.Series:
    txt = flag.astype(str).fillna("").str.strip().str.upper()
    no_alloc = txt.str.contains("NO ALLOC", na=False) | txt.str.startswith("Z - NO", na=False)
    review = txt.str.contains("REVIEW", na=False)
    allocate = txt.str.contains("ALLOC", na=False) & ~no_alloc
    blank = txt.isin(["", "NAN", "NONE"])
    return pd.Series(np.select([no_alloc, review, allocate, blank], ["No Alloc", "Review", "Allocate", "Blank"], default="Other"), index=flag.index)


def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = make_unique_columns(out.columns)
    for c in out.columns:
        if c in NUMERIC_HINTS or c.startswith("AI "):
            out[c] = clean_numeric_series(out[c])
    return out


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_dataframe(df)

    l30 = safe_num(df, "L30")
    d30 = safe_num(df, "D30")
    d60 = safe_num(df, "D60")
    lw = safe_num(df, "LW")
    ttm = safe_num(df, "TTM")
    qoh = safe_num(df, "Qoh")
    supply = safe_num(df, "Supply")
    alloc_rec = safe_num(df, ALLOC_REC_COL)
    flm = safe_num(df, "FLM", 1).replace(0, 1)
    dc_avail = safe_num(df, "Dc Avail")
    dc_qoh = safe_num(df, "Dc Qoh")
    proj = safe_num(df, "Proj. Demand")
    days = safe_num(df, "Days", 30).replace(0, 30)
    avg_woc = safe_num(df, "Avg. WOC")
    intrans = safe_num(df, "Intrans")
    store_po = safe_num(df, "Store PO Qty")
    reserve = safe_num(df, "QTY Reserve")
    supply_in_stock = safe_num(df, "Supply In Stock")

    flags = flag_class_series(safe_cat(df, FLAG_COL))
    df["FE_flag_class"] = flags
    df["FE_is_allocate"] = (flags == "Allocate").astype(int)
    df["FE_is_review"] = (flags == "Review").astype(int)
    df["FE_is_no_alloc"] = (flags == "No Alloc").astype(int)
    df["FE_is_blank_flag"] = (flags == "Blank").astype(int)

    df["FE_recent_velocity"] = (l30 * 0.40) + (d30 * 0.20) + ((d60 / 2.0) * 0.25) + (lw * 4.29 * 0.15)
    df["FE_monthly_ttm_rate"] = ttm / 12.0
    df["FE_projected_demand_gap"] = proj - supply
    df["FE_d60_gap"] = d60 - supply
    df["FE_l30_gap"] = l30 - qoh
    df["FE_pipeline_units"] = intrans + store_po + reserve
    df["FE_total_near_supply"] = supply + intrans + store_po + reserve
    df["FE_alloc_rec_units"] = safe_div(alloc_rec, flm)
    df["FE_supply_to_d60"] = safe_div(supply, d60.clip(lower=1))
    df["FE_qoh_to_l30"] = safe_div(qoh, l30.clip(lower=1))
    df["FE_alloc_rec_to_d60_gap"] = safe_div(alloc_rec, (d60 - supply).clip(lower=1))
    df["FE_dc_avail_to_rec"] = safe_div(dc_avail, alloc_rec.clip(lower=1))
    df["FE_dc_avail_to_qoh"] = safe_div(dc_avail, dc_qoh.clip(lower=1))
    df["FE_days_scaled_velocity"] = df["FE_recent_velocity"] * safe_div(days, pd.Series(30.0, index=df.index), 1.0)
    df["FE_woc_pressure"] = safe_div(pd.Series(4.0, index=df.index), avg_woc.clip(lower=0.25))
    df["FE_supply_in_stock_gap"] = d60 - supply_in_stock
    df["FE_is_new_item"] = safe_cat(df, "New").str.upper().isin(["Y", "YES", "TRUE", "1", "NEW"]).astype(int)
    df["FE_has_store_flag"] = safe_cat(df, "Store Flag").str.len().gt(0).astype(int)
    df["FE_has_sku_flag"] = safe_cat(df, "SKU Flag").str.len().gt(0).astype(int)

    if TARGET_COL in df.columns:
        df["FE_manual_adjustment"] = clean_numeric_series(df[TARGET_COL]) - alloc_rec

    return df.replace([np.inf, -np.inf], np.nan)


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    def usable(c: str) -> bool:
        return c in df.columns and c not in LEAKAGE_COLS and df[c].notna().any()
    base = [c for c in PREFERRED_FEATURES if usable(c)]
    engineered = [c for c in df.columns if c.startswith("FE_") and c not in {"FE_manual_adjustment", "FE_source_file"} and df[c].notna().any()]
    return base + engineered


def split_feature_types(df: pd.DataFrame, feature_cols: Sequence[str]) -> Tuple[List[str], List[str]]:
    numeric_cols: List[str] = []
    cat_cols: List[str] = []
    for c in feature_cols:
        if c in NUMERIC_HINTS or (c.startswith("FE_") and c != "FE_flag_class"):
            numeric_cols.append(c)
        elif c in df.columns:
            numeric_like = clean_numeric_series(df[c]).notna().mean() if len(df) else 0
            if numeric_like > 0.90:
                numeric_cols.append(c)
            else:
                cat_cols.append(c)
    return numeric_cols, cat_cols


def coerce_feature_frame(df: pd.DataFrame, feature_cols: Sequence[str], numeric_cols: Sequence[str], categorical_cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in feature_cols:
        if c not in out.columns:
            out[c] = np.nan
    for c in numeric_cols:
        out[c] = clean_numeric_series(out[c]) if c in out.columns else np.nan
    for c in categorical_cols:
        out[c] = out[c].astype(str).replace({"nan": "__MISSING__", "None": "__MISSING__"}).fillna("__MISSING__") if c in out.columns else "__MISSING__"
    return out[list(feature_cols)]


def make_preprocessor(df: pd.DataFrame, feature_cols: Sequence[str]) -> Tuple[ColumnTransformer, List[str], List[str]]:
    numeric_cols, categorical_cols = split_feature_types(df, feature_cols)
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols),
    ], remainder="drop", verbose_feature_names_out=False)
    return pre, numeric_cols, categorical_cols


def build_model(choice: str) -> Any:
    if choice == "Random Forest":
        return RandomForestRegressor(n_estimators=80, min_samples_leaf=3, max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1)
    if choice == "Gradient Boosting":
        return GradientBoostingRegressor(n_estimators=150, learning_rate=0.055, max_depth=3, random_state=RANDOM_STATE)
    return ExtraTreesRegressor(n_estimators=90, min_samples_leaf=2, max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1)

# -----------------------------
# Training
# -----------------------------

def balance_raw_rows_for_training(df: pd.DataFrame, max_zero_rows_per_file: int = 6000) -> pd.DataFrame:
    """Select trainable rows before expensive feature engineering.

    The submitted Daily Allocation exports are ~214k rows each. Most rows are
    explicit no-allocation rows, so engineering every row is slow and creates a
    model dominated by zeros. This function keeps all meaningful allocation
    rows plus a controlled sample of no-allocation rows.
    """
    raw = df.copy()
    final = clean_numeric_series(raw[TARGET_COL]) if TARGET_COL in raw.columns else pd.Series(np.nan, index=raw.index)
    alloc_rec = clean_numeric_series(raw[ALLOC_REC_COL]) if ALLOC_REC_COL in raw.columns else pd.Series(0, index=raw.index)
    flags = flag_class_series(raw[FLAG_COL]) if FLAG_COL in raw.columns else pd.Series("Blank", index=raw.index)

    corrected_or_positive = final.notna() & final.fillna(0).ne(0)
    explicit_zero = final.notna() & final.fillna(0).eq(0)
    allocation_context = alloc_rec.fillna(0).ne(0) | flags.isin(["Allocate", "Review"])
    no_alloc_pool = flags.eq("No Alloc") & ~(corrected_or_positive | explicit_zero | allocation_context)

    keep_mask = corrected_or_positive | explicit_zero | allocation_context
    keep = raw[keep_mask].copy()
    if no_alloc_pool.sum() > 0:
        sample_n = min(max_zero_rows_per_file, int(no_alloc_pool.sum()))
        keep = pd.concat([keep, raw[no_alloc_pool].sample(sample_n, random_state=RANDOM_STATE)], ignore_index=True, sort=False)
    return keep.reset_index(drop=True)


def make_training_frame(parsed_files: Sequence[ParsedSheet], prior_memory_json: Optional[str] = None) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    if prior_memory_json:
        try:
            frames.append(pd.read_json(io.StringIO(prior_memory_json), orient="split"))
        except Exception:
            pass
    for p in parsed_files:
        selected = balance_raw_rows_for_training(p.df)
        df = add_engineered_features(selected)
        df["FE_source_file"] = p.source_name
        frames.append(df)
    if not frames:
        raise ValueError("No trainable files were parsed.")
    full = pd.concat(frames, ignore_index=True, sort=False)
    return full


def balance_training_rows(full: pd.DataFrame, max_zero_rows: int = 35000) -> pd.DataFrame:
    """Final balancing after old embedded memory and new files are combined."""
    final = clean_numeric_series(full[TARGET_COL]) if TARGET_COL in full.columns else pd.Series(np.nan, index=full.index)
    alloc_rec = safe_num(full, ALLOC_REC_COL)
    flag_class = flag_class_series(safe_cat(full, FLAG_COL))

    corrected_or_positive = final.notna() & final.fillna(0).ne(0)
    final_explicit_zero = final.notna() & final.fillna(0).eq(0)
    allocation_context = alloc_rec.fillna(0).ne(0) | flag_class.isin(["Allocate", "Review"])
    no_alloc_context = flag_class.eq("No Alloc")

    must_keep = corrected_or_positive | final_explicit_zero | allocation_context
    zero_pool = (~must_keep) & no_alloc_context

    keep = full[must_keep].copy()
    if zero_pool.sum() > 0:
        sample_n = min(max_zero_rows, int(zero_pool.sum()))
        zero_sample = full[zero_pool].sample(sample_n, random_state=RANDOM_STATE)
        keep = pd.concat([keep, zero_sample], ignore_index=True, sort=False)

    keep[TARGET_COL] = clean_numeric_series(keep.get(TARGET_COL, pd.Series(np.nan, index=keep.index))).fillna(0)
    return keep.reset_index(drop=True)

def prepare_xy(parsed_files: Sequence[ParsedSheet], target_mode: str, prior_memory_json: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str], List[str], pd.DataFrame]:
    full = make_training_frame(parsed_files, prior_memory_json=prior_memory_json)
    train_df = balance_training_rows(full)
    if TARGET_COL not in train_df.columns:
        raise ValueError(f"Training files must contain '{TARGET_COL}'.")

    alloc_rec = safe_num(train_df, ALLOC_REC_COL)
    final = clean_numeric_series(train_df[TARGET_COL]).fillna(0)
    if target_mode == "Manual Adjustment from Alloc. Rec.":
        y = (final - alloc_rec).astype(float)
    else:
        y = final.astype(float)
    y = y.replace([np.inf, -np.inf], np.nan).fillna(0)

    feature_cols = select_feature_columns(train_df)
    if len(feature_cols) < 5:
        raise ValueError("Not enough usable feature columns were found. Check that the Daily Allocation headers parsed correctly.")
    pre, numeric_cols, cat_cols = make_preprocessor(train_df, feature_cols)
    X = coerce_feature_frame(train_df, feature_cols, numeric_cols, cat_cols)
    return X, y, feature_cols, numeric_cols, cat_cols, train_df


def train_model(parsed_files: Sequence[ParsedSheet], model_choice: str, target_mode: str, embed_memory: bool, prior_bundle: Optional[Dict[str, Any]] = None) -> TrainResult:
    prior_memory = prior_bundle.get("training_memory_json") if prior_bundle else None
    X, y, feature_cols, numeric_cols, cat_cols, train_df = prepare_xy(parsed_files, target_mode, prior_memory_json=prior_memory)

    preprocessor, _, _ = make_preprocessor(train_df, feature_cols)
    # Re-coerce after final type lists are known.
    X = coerce_feature_frame(train_df, feature_cols, numeric_cols, cat_cols)

    pipeline = Pipeline([("preprocessor", preprocessor), ("model", build_model(model_choice))])

    test_size = 0.20 if len(train_df) >= 100 else 0.30
    group_col = "Product ID" if "Product ID" in train_df.columns else ("Item" if "Item" in train_df.columns else None)
    idx = np.arange(len(train_df))
    if group_col and train_df[group_col].nunique(dropna=True) > 5 and len(train_df) >= 100:
        groups = train_df[group_col].fillna("__MISSING_GROUP__").astype(str).replace({"nan": "__MISSING_GROUP__"})
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    else:
        train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=RANDOM_STATE)

    pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
    pred = pipeline.predict(X.iloc[test_idx])

    final_pipeline = Pipeline([("preprocessor", preprocessor), ("model", build_model(model_choice))])
    final_pipeline.fit(X, y)

    mae = float(mean_absolute_error(y.iloc[test_idx], pred)) if len(test_idx) else float("nan")
    rmse = float(mean_squared_error(y.iloc[test_idx], pred) ** 0.5) if len(test_idx) else float("nan")
    r2 = float(r2_score(y.iloc[test_idx], pred)) if len(test_idx) > 1 else float("nan")

    validation = train_df.iloc[test_idx].copy()
    validation["AI Validation Actual Target"] = y.iloc[test_idx].values
    validation["AI Validation Predicted Target"] = pred
    validation["AI Validation Abs Error"] = np.abs(y.iloc[test_idx].values - pred)

    metrics = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "rows_used": int(len(train_df)),
        "files_used_this_run": int(len(parsed_files)),
        "feature_count": int(len(feature_cols)),
        "numeric_feature_count": int(len(numeric_cols)),
        "categorical_feature_count": int(len(cat_cols)),
        "target_mode": target_mode,
        "model_choice": model_choice,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }

    memory_json = None
    if embed_memory:
        keep_cols = sorted(set(feature_cols + [TARGET_COL, ALLOC_REC_COL, FLAG_COL, "FE_source_file"]).intersection(train_df.columns))
        mem = train_df[keep_cols].copy()
        max_rows = 70000
        if len(mem) > max_rows:
            final = clean_numeric_series(mem[TARGET_COL]) if TARGET_COL in mem.columns else pd.Series(0, index=mem.index)
            alloc = clean_numeric_series(mem[ALLOC_REC_COL]) if ALLOC_REC_COL in mem.columns else pd.Series(0, index=mem.index)
            important = final.fillna(0).ne(0) | alloc.fillna(0).ne(0)
            imp = mem[important]
            rest = mem[~important]
            mem = pd.concat([imp, rest.sample(min(max_rows - min(len(imp), max_rows), len(rest)), random_state=RANDOM_STATE)], ignore_index=True).head(max_rows)
        memory_json = mem.to_json(orient="split")

    bundle = {
        "model_version": MODEL_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "pipeline": final_pipeline,
        "feature_columns": feature_cols,
        "numeric_columns": numeric_cols,
        "categorical_columns": cat_cols,
        "target_column": TARGET_COL,
        "target_mode": target_mode,
        "model_choice": model_choice,
        "metrics": metrics,
        "training_files": [p.source_name for p in parsed_files] + (prior_bundle.get("training_files", []) if prior_bundle else []),
        "training_memory_json": memory_json,
        "postprocessing_defaults": {
            "prediction_mode": "Balanced",
            "rounding_mode": "floor",
            "max_final_supply_over_d60_flm": 1.0,
            "respect_no_alloc_flags": True,
            "cap_by_dc_avail": True,
        },
    }
    return TrainResult(bundle=bundle, metrics=metrics, validation_rows=validation)

# -----------------------------
# Prediction / postprocessing
# -----------------------------

def round_to_flm(value: float, flm: float, mode: str) -> float:
    try:
        value = float(value)
    except Exception:
        value = 0.0
    try:
        flm = float(flm)
    except Exception:
        flm = 1.0
    if not np.isfinite(value) or value <= 0:
        return 0.0
    if not np.isfinite(flm) or flm <= 0:
        flm = 1.0
    if mode == "ceil":
        return float(math.ceil(value / flm) * flm)
    if mode == "nearest":
        return float(round(value / flm) * flm)
    return float(math.floor(value / flm) * flm)


def apply_postprocessing(df: pd.DataFrame, raw_pred: np.ndarray, bundle: Dict[str, Any], prediction_mode: str, rounding_mode: str,
                         max_over_d60_flm: float, respect_no_alloc_flags: bool, cap_by_dc_avail: bool) -> pd.DataFrame:
    out = df.copy()
    alloc_rec = safe_num(out, ALLOC_REC_COL)
    flm = safe_num(out, "FLM", 1).replace(0, 1)
    supply = safe_num(out, "Supply")
    d60 = clean_numeric_series(out["D60"]) if "D60" in out.columns else pd.Series(np.nan, index=out.index)
    dc_avail = clean_numeric_series(out["Dc Avail"]) if "Dc Avail" in out.columns else pd.Series(np.nan, index=out.index)
    flag_class = flag_class_series(safe_cat(out, FLAG_COL))

    if bundle.get("target_mode") == "Manual Adjustment from Alloc. Rec.":
        proposed = alloc_rec.fillna(0).to_numpy(dtype=float) + np.asarray(raw_pred, dtype=float)
    else:
        proposed = np.asarray(raw_pred, dtype=float)

    if prediction_mode == "Conservative":
        proposed = np.minimum(proposed, alloc_rec.fillna(0).to_numpy(dtype=float))
        rounding_effective = "floor"
        max_over_d60_flm = min(max_over_d60_flm, 0.75)
    elif prediction_mode == "Aggressive":
        rounding_effective = rounding_mode
        max_over_d60_flm = max(max_over_d60_flm, 1.5)
    else:
        rounding_effective = rounding_mode

    clean_vals: List[float] = []
    reasons: List[str] = []
    confidence: List[str] = []

    for pos, idx in enumerate(out.index):
        val = proposed[pos]
        if not np.isfinite(val):
            val = 0.0
        val = max(float(val), 0.0)
        row_reasons: List[str] = []
        row_flm = float(flm.loc[idx]) if np.isfinite(flm.loc[idx]) and flm.loc[idx] > 0 else 1.0
        row_supply = float(supply.loc[idx]) if np.isfinite(supply.loc[idx]) else 0.0
        row_d60 = d60.loc[idx]
        row_dc = dc_avail.loc[idx]
        row_rec = float(alloc_rec.loc[idx]) if np.isfinite(alloc_rec.loc[idx]) else 0.0
        row_flag = flag_class.loc[idx]

        if respect_no_alloc_flags and row_flag in {"No Alloc", "Blank", "Other"} and row_rec <= 0:
            val = 0.0
            row_reasons.append("No-allocation/blank flag and no Alloc. Rec.")

        if pd.notna(row_d60) and np.isfinite(float(row_d60)) and float(row_d60) >= 0:
            cap = max(0.0, float(row_d60) + float(max_over_d60_flm) * row_flm - row_supply)
            if val > cap:
                val = cap
                row_reasons.append(f"Capped to keep Final Supply near D60 + {max_over_d60_flm:g} FLM")

        if cap_by_dc_avail and pd.notna(row_dc) and np.isfinite(float(row_dc)) and float(row_dc) >= 0 and val > float(row_dc):
            val = float(row_dc)
            row_reasons.append("Capped by Dc Avail")

        rounded = round_to_flm(val, row_flm, rounding_effective)
        if abs(rounded - val) > 1e-9:
            row_reasons.append(f"Rounded to FLM by {rounding_effective}")
        val = rounded

        if row_rec > 0 and abs(val - row_rec) > 2 * max(row_flm, 1.0):
            row_reasons.append("Large change from Alloc. Rec.; review")
        if pd.notna(row_d60) and (row_supply + val) > float(row_d60) + row_flm:
            row_reasons.append("Final Supply still exceeds D60 by more than one FLM")
        if row_flag == "Review":
            row_reasons.append("Original row marked Review")

        if any("review" in r.lower() or "exceeds" in r.lower() for r in row_reasons):
            conf = "Low"
        elif len(row_reasons) >= 2:
            conf = "Medium"
        else:
            conf = "High"

        clean_vals.append(val)
        reasons.append("; ".join(dict.fromkeys(row_reasons)) if row_reasons else "Model prediction accepted")
        confidence.append(conf)

    old_final = clean_numeric_series(out[TARGET_COL]) if TARGET_COL in out.columns else pd.Series(np.nan, index=out.index)
    out["AI Raw ML Prediction"] = raw_pred
    out["AI Proposed Final Alloc."] = proposed
    out["AI Clean Final Alloc."] = clean_vals
    out["AI Change From Original"] = pd.Series(clean_vals, index=out.index) - old_final.fillna(0)
    out["AI Confidence"] = confidence
    out["AI Review Reason"] = reasons
    out["AI Prediction Mode"] = prediction_mode
    out["AI Flag Class"] = flag_class.values
    out["AI Model Version"] = bundle.get("model_version", "unknown")

    out[TARGET_COL] = clean_vals
    blank_mask = respect_no_alloc_flags & flag_class.isin(["No Alloc", "Blank", "Other"]) & alloc_rec.fillna(0).le(0) & pd.Series(clean_vals, index=out.index).eq(0)
    out.loc[blank_mask, TARGET_COL] = np.nan
    return out


def predict_with_bundle(parsed: ParsedSheet, bundle: Dict[str, Any], prediction_mode: str, rounding_mode: str,
                        max_over_d60_flm: float, respect_no_alloc_flags: bool, cap_by_dc_avail: bool) -> pd.DataFrame:
    df = add_engineered_features(parsed.df)
    feature_cols = bundle["feature_columns"]
    numeric_cols = bundle.get("numeric_columns", [])
    cat_cols = bundle.get("categorical_columns", [])
    X = coerce_feature_frame(df, feature_cols, numeric_cols, cat_cols)
    raw_pred = bundle["pipeline"].predict(X)
    return apply_postprocessing(parsed.df.copy(), raw_pred, bundle, prediction_mode, rounding_mode, max_over_d60_flm, respect_no_alloc_flags, cap_by_dc_avail)

# -----------------------------
# Serialization / exports
# -----------------------------

def model_to_bytes(bundle: Dict[str, Any]) -> bytes:
    bio = io.BytesIO()
    joblib.dump(bundle, bio, compress=3)
    bio.seek(0)
    return bio.getvalue()


def load_model_bundle(uploaded_file: Any) -> Dict[str, Any]:
    data = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
    return joblib.load(io.BytesIO(data))


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def df_to_xlsx_bytes(df: pd.DataFrame, sheet_name: str = "Allocation AI Output") -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name[:31])
        ws = writer.sheets[sheet_name[:31]]
        ws.freeze_panes(1, 0)
        for i, c in enumerate(df.columns):
            width = min(max(len(str(c)) + 2, 10), 34)
            ws.set_column(i, i, width)
    bio.seek(0)
    return bio.getvalue()


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y_%m_%d_%H%M")

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title=APP_NAME, page_icon="📦", layout="wide")
st.title("📦 Allocation AI")
st.caption("Train, save, reload, and apply a machine-learning model for Daily Allocation files formatted like your submitted CSV exports.")

with st.sidebar:
    st.header("Model Settings")
    model_choice = st.selectbox("Model type", ["Extra Trees", "Random Forest", "Gradient Boosting"], index=0)
    target_mode = st.selectbox("Training target", ["Manual Adjustment from Alloc. Rec.", "Final Alloc. Directly"], index=0)
    embed_memory = st.checkbox("Embed compact training memory in model", value=True)
    st.divider()
    st.header("Prediction Safety")
    prediction_mode = st.selectbox("Prediction mode", ["Conservative", "Balanced", "Aggressive"], index=1)
    rounding_mode = st.selectbox("FLM rounding", ["floor", "nearest", "ceil"], index=0)
    max_over = st.number_input("Max Final Supply over D60, in FLMs", min_value=0.0, max_value=5.0, value=1.0, step=0.25)
    respect_no_alloc = st.checkbox("Respect No Alloc / blank rows", value=True)
    cap_dc = st.checkbox("Cap by Dc Avail", value=True)

train_tab, predict_tab, continue_tab, inspect_tab = st.tabs(["Train Model", "Predict / Edit File", "Continue Training", "Inspect Files"])

with train_tab:
    st.subheader("Train a model from corrected Daily Allocation files")
    st.write("Upload corrected historical CSV/XLSX files where `Final Alloc.` reflects the desired human-reviewed outcome.")
    train_files = st.file_uploader("Historical corrected allocation files", type=["csv", "xlsx", "xlsm", "xls"], accept_multiple_files=True, key="train_files")
    if st.button("Train model", type="primary", disabled=not train_files):
        parsed = parse_many(train_files)
        if parsed:
            with st.spinner("Training model..."):
                result = train_model(parsed, model_choice, target_mode, embed_memory)
            st.success("Model trained successfully.")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows used", f"{result.metrics['rows_used']:,}")
            c2.metric("MAE", f"{result.metrics['mae']:.3f}")
            c3.metric("RMSE", f"{result.metrics['rmse']:.3f}")
            c4.metric("R²", f"{result.metrics['r2']:.3f}")
            st.download_button("Download trained model", model_to_bytes(result.bundle), file_name=f"allocation_ai_model_{timestamp_slug()}.joblib", mime="application/octet-stream")
            st.dataframe(result.validation_rows.head(500), use_container_width=True)
            if px is not None and not result.validation_rows.empty:
                fig = px.histogram(result.validation_rows, x="AI Validation Abs Error", nbins=60, title="Validation Absolute Error Distribution")
                st.plotly_chart(fig, use_container_width=True)

with predict_tab:
    st.subheader("Upload a model and a new allocation file to populate Final Alloc.")
    model_file = st.file_uploader("Pretrained Allocation AI model (.joblib)", type=["joblib", "pkl"], key="predict_model")
    pred_file = st.file_uploader("Daily Allocation file to edit", type=["csv", "xlsx", "xlsm", "xls"], key="predict_file")
    out_type = st.radio("Output type", ["XLSX", "CSV"], horizontal=True)
    if st.button("Run predictions and build edited file", type="primary", disabled=not (model_file and pred_file)):
        bundle = load_model_bundle(model_file)
        parsed = parse_allocation_file(pred_file)
        with st.spinner("Predicting Final Alloc. and applying business rules..."):
            out_df = predict_with_bundle(parsed, bundle, prediction_mode, rounding_mode, max_over, respect_no_alloc, cap_dc)
        st.success("Edited file is ready.")
        st.dataframe(out_df.head(1000), use_container_width=True)
        base = os.path.splitext(pred_file.name)[0]
        if out_type == "XLSX":
            st.download_button("Download edited XLSX", df_to_xlsx_bytes(out_df), file_name=f"{base}_allocation_ai_output.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.download_button("Download edited CSV", df_to_csv_bytes(out_df), file_name=f"{base}_allocation_ai_output.csv", mime="text/csv")
        if "AI Confidence" in out_df.columns:
            st.write("Confidence summary")
            st.dataframe(out_df["AI Confidence"].value_counts(dropna=False).rename_axis("Confidence").reset_index(name="Rows"), use_container_width=True)

with continue_tab:
    st.subheader("Continue training from an existing model")
    st.write("Upload the current model plus newly corrected files. If the old model contains embedded memory, this retrains on old + new data.")
    old_model = st.file_uploader("Existing Allocation AI model", type=["joblib", "pkl"], key="continue_model")
    new_files = st.file_uploader("New corrected allocation files", type=["csv", "xlsx", "xlsm", "xls"], accept_multiple_files=True, key="continue_files")
    if st.button("Retrain / update model", type="primary", disabled=not (old_model and new_files)):
        prior = load_model_bundle(old_model)
        parsed = parse_many(new_files)
        with st.spinner("Retraining with prior model memory plus new corrected files..."):
            result = train_model(parsed, model_choice, target_mode, embed_memory=True, prior_bundle=prior)
        st.success("Updated model trained successfully.")
        st.json(result.metrics)
        st.download_button("Download updated model", model_to_bytes(result.bundle), file_name=f"allocation_ai_updated_{timestamp_slug()}.joblib", mime="application/octet-stream")
        st.dataframe(result.validation_rows.head(500), use_container_width=True)

with inspect_tab:
    st.subheader("Inspect file parsing")
    st.write("Use this to confirm the app detects your Daily Allocation headers correctly.")
    inspect_files = st.file_uploader("Files to inspect", type=["csv", "xlsx", "xlsm", "xls"], accept_multiple_files=True, key="inspect_files")
    if inspect_files:
        parsed = parse_many(inspect_files)
        for p in parsed:
            st.markdown(f"### {p.source_name}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Detected header row", p.header_row_idx + 1)
            c2.metric("Rows", f"{len(p.df):,}")
            c3.metric("Columns", f"{len(p.df.columns):,}")
            st.write("Detected columns")
            st.code(", ".join(map(str, p.df.columns.tolist())))
            st.dataframe(p.df.head(20), use_container_width=True)
