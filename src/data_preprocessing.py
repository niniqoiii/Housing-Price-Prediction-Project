"""The California Housing dataset has no missing values. However, we still implement a robust cleaning pipeline for reproducibility and best practices."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.makedirs("reports/figures", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# ======================================================
# PREPROCESSING FUNCTION
# ======================================================

df = pd.read_csv("data/raw/california_housing_raw.csv")


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the housing DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        Raw input data

    Returns:
    --------
    pd.DataFrame
        Cleaned DataFrame
    """
    """
    Preprocessing pipeline:
    1) Remove duplicates
    2) Flag and impute missing values
    3) Convert data types
    4) Create derived features
    5) Detect and cap outliers using IQR
    """

    print("\n===== DATA QUALITY REPORT =====")

    df_clean = df.copy()

    # --------------------------------------------------
    # 1. Remove duplicate rows
    # --------------------------------------------------
    before = len(df_clean)
    df_clean.drop_duplicates(inplace=True)
    print(f"Removed {before - len(df_clean)} duplicate rows")

    # --------------------------------------------------
    # 2. Flag and impute missing values
    # --------------------------------------------------
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:

            # Flag column
            df_clean[f"{col}_missing"] = df_clean[col].isnull().astype(int)

            # Impute
            if df_clean[col].dtype in ["int64", "float64"]:
                median_value = df_clean[col].median()
                df_clean[col].fillna(median_value, inplace=True)
                print(f"Imputed missing values in '{col}' with median")
            else:
                mode_value = df_clean[col].mode()[0]
                df_clean[col].fillna(mode_value, inplace=True)
                print(f"Imputed missing values in '{col}' with mode")

    # --------------------------------------------------
    # 3. Data type conversion
    # --------------------------------------------------
    for col in df_clean.select_dtypes(include="object").columns:
        df_clean[col] = df_clean[col].astype("category")

    print("Converted object columns to category type")

    # --------------------------------------------------
    # 4. Derived feature
    # --------------------------------------------------
    numeric_cols = df_clean.select_dtypes(include=np.number).columns

    if len(numeric_cols) >= 2:
        df_clean["Derived_Feature"] = df_clean[numeric_cols[0]] / (df_clean[numeric_cols[1]] + 1)
    

    # --------------------------------------------------
    # 5. Outlier detection & capping (IQR)
    # --------------------------------------------------
    def cap_outliers(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return series.clip(lower, upper)

    numeric_cols = df_clean.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        df_clean[col] = cap_outliers(df_clean[col])


    print("Capped outliers in numeric columns using IQR")

    print("Final dataset shape:", df_clean.shape)
    print("==============================\n")

    return df_clean

# ======================================================
# RUN PIPELINE
# ======================================================
df_clean = preprocess_data(df)


# ======================================================
# BOXPLOT CHECK
# ======================================================

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
axes = axes.flatten()

numeric_cols = df_clean.select_dtypes(include='number').columns.tolist()
for i, col in enumerate(numeric_cols[:len(axes)]):
    axes[i].boxplot(df_clean[col].dropna(), patch_artist=True,
                    boxprops=dict(facecolor='#AED6F1', color='#2874A6'),
                    medianprops=dict(color='#C0392B', linewidth=2))
    axes[i].set_title(col)
    axes[i].set_xlabel('')

for j in range(len(numeric_cols), len(axes)):
    fig.delaxes(axes[j])

plt.suptitle('Box Plots — Outlier Detection Across All Features', fontsize=14, fontweight='bold', y=0.99)
plt.tight_layout()

plt.savefig("reports/figures/outlier_boxplots.png", dpi=300)
plt.close()

print("Outlier boxplot saved to reports/figures/outlier_boxplots.png")

df_clean.to_csv("data/processed/california_housing_clean.csv", index=False)
print("Cleaned data saved to data/processed/california_housing_clean.csv")
