import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def infer_column_types(df):
    types = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            types[col] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            types[col] = "datetime"
        else:
            types[col] = "categorical"
    return types

def missing_summary(df):
    total = len(df)
    miss = df.isna().sum()
    miss_pct = (miss / total * 100).round(2)
    summary = pd.DataFrame({"missing_count": miss, "missing_percent": miss_pct})
    return summary[summary["missing_count"] > 0].sort_values("missing_percent", ascending=False)

def iqr_outlier_summary(df, numeric_cols):
    outlier_info = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)][col]
        outlier_info[col] = {"n_outliers": int(outliers.count()), "lower_bound": float(lower), "upper_bound": float(upper)}
    return outlier_info

def top_correlations(df, numeric_cols, n=10):
    if not numeric_cols:
        return pd.DataFrame()
    corr = df[numeric_cols].corr().abs().unstack().reset_index()
    corr.columns = ["feature_1","feature_2","abs_corr"]
    corr = corr[corr["feature_1"] != corr["feature_2"]]
    corr["pair"] = corr.apply(lambda r: tuple(sorted([r["feature_1"], r["feature_2"]])), axis=1)
    corr = corr.drop_duplicates(subset=["pair"]).sort_values("abs_corr", ascending=False)
    return corr.head(n)[["feature_1","feature_2","abs_corr"]].reset_index(drop=True)

def simple_impute(df):
    df2 = df.copy()
    numeric_cols = df2.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        imputer = SimpleImputer(strategy="median")
        df2[numeric_cols] = imputer.fit_transform(df2[numeric_cols])
    cat_cols = df2.select_dtypes(exclude=[np.number, "datetime64[ns]"]).columns.tolist()
    for c in cat_cols:
        if df2[c].isna().any():
            mode = df2[c].mode().iloc[0]
            df2[c] = df2[c].fillna(mode)
    return df2

def data_health_score(df):
    score = 100
    total_rows = len(df)

    if total_rows == 0:
        return 0, {"Missing%": 0, "Duplicates%": 0, "Outliers%": 0}

    # Missing %
    missing_pct = (df.isna().sum().sum() / (total_rows * len(df.columns))) * 100
    # Duplicates %
    dup_pct = (df.duplicated().sum() / total_rows) * 100
    # Outliers %
    outlier_pct = 0
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        outliers_total = 0
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers_total += df[(df[col] < lower) | (df[col] > upper)].shape[0]
        outlier_pct = (outliers_total / total_rows) * 100

    score -= (missing_pct * 0.5 + dup_pct * 0.3 + outlier_pct * 0.2)
    score = max(0, min(100, round(score, 2)))

    breakdown = {"Missing%": round(missing_pct, 2), "Duplicates%": round(dup_pct, 2), "Outliers%": round(outlier_pct, 2)}
    return score, breakdown
