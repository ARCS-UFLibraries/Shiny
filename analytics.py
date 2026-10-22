"""Reusable, testable data-quality utilities for the Political Science Data Lab."""
from __future__ import annotations

import pandas as pd

REQUIRED_DICTIONARY_COLUMNS = {"column_name", "description", "data_type"}


def profile_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per column with missingness, cardinality, and IQR outliers."""
    numeric_columns = set(df.select_dtypes(include="number").columns)
    rows: list[dict[str, object]] = []
    for column in df.columns:
        series = df[column]
        outliers: int | None = None
        if column in numeric_columns:
            values = series.dropna()
            if values.empty:
                outliers = 0
            else:
                q1, q3 = values.quantile([0.25, 0.75])
                iqr = q3 - q1
                outliers = int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum())
        rows.append({
            "column_name": str(column),
            "data_type": str(series.dtype),
            "missing_count": int(series.isna().sum()),
            "missing_pct": round(float(series.isna().mean() * 100), 2),
            "unique_values": int(series.nunique(dropna=True)),
            "IQR_outliers": outliers,
        })
    return pd.DataFrame(rows)


def validate_data_dictionary(df: pd.DataFrame, dictionary: pd.DataFrame) -> dict[str, object]:
    """Validate dictionary structure and enforce required/allowed-value rules."""
    missing_fields = sorted(REQUIRED_DICTIONARY_COLUMNS - set(dictionary.columns))
    if missing_fields:
        return {"ok": False, "message": "Dictionary is missing required columns: " + ", ".join(missing_fields), "dictionary": dictionary}

    dictionary = dictionary.copy()
    if dictionary["column_name"].isna().any() or dictionary["column_name"].astype(str).str.strip().eq("").any():
        return {"ok": False, "message": "Every dictionary row must have a non-empty column_name.", "dictionary": dictionary}
    dictionary["column_name"] = dictionary["column_name"].astype(str).str.strip()
    if dictionary["column_name"].duplicated().any():
        return {"ok": False, "message": "The dictionary contains duplicate column_name entries.", "dictionary": dictionary}

    listed = set(dictionary["column_name"])
    actual = set(map(str, df.columns))
    missing_entries = sorted(actual - listed)
    extra_entries = sorted(listed - actual)
    if missing_entries:
        return {"ok": False, "message": "Add dictionary entries for dataset columns: " + ", ".join(missing_entries), "dictionary": dictionary}

    issues: list[str] = []
    for _, spec in dictionary.iterrows():
        column = str(spec["column_name"])
        if column not in df.columns:
            continue
        series = df[column]
        expected = str(spec.get("data_type", "")).strip().lower()
        if expected in {"numeric", "number", "integer", "int", "float", "decimal"}:
            converted = pd.to_numeric(series, errors="coerce")
            invalid = int((series.notna() & converted.isna()).sum())
            if expected in {"integer", "int"}:
                invalid += int((converted.notna() & (converted % 1 != 0)).sum())
            if invalid:
                issues.append(f"{column}: {invalid} value(s) do not match expected type {expected}.")
        elif expected in {"date", "datetime", "timestamp"}:
            converted = pd.to_datetime(series, errors="coerce")
            invalid = int((series.notna() & converted.isna()).sum())
            if invalid:
                issues.append(f"{column}: {invalid} invalid date value(s).")
        required_value = str(spec.get("required", "false")).strip().lower()
        if required_value in {"true", "yes", "1", "required"}:
            missing_count = int(series.isna().sum())
            if missing_count:
                issues.append(f"{column}: {missing_count} missing value(s), but the dictionary marks it required.")
        allowed = spec.get("allowed_values", "")
        if pd.notna(allowed) and str(allowed).strip():
            allowed_values = {value.strip() for value in str(allowed).split("|") if value.strip()}
            unexpected = sorted(set(series.dropna().astype(str).unique()) - allowed_values)
            if unexpected:
                preview = ", ".join(unexpected[:5])
                suffix = "…" if len(unexpected) > 5 else ""
                issues.append(f"{column}: values outside allowed_values ({preview}{suffix}).")

    message = "Dictionary matches all dataset columns."
    if issues:
        message += " Validation findings: " + " ".join(issues[:6])
        if len(issues) > 6:
            message += f" And {len(issues) - 6} more finding(s)."
    if extra_entries:
        message += " Extra dictionary entries not found in the dataset: " + ", ".join(extra_entries) + "."
    return {"ok": not issues, "message": message, "dictionary": dictionary}
