import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing


# Plot styling
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['axes.titlesize'] = 13

# Load dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame.copy()

# Preview data
print(df.head(10))

# --- Initial data report ---

print("Shape:", df.shape)

def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a comprehensive data quality report.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe to analyze

    Returns:
    --------
    pd.DataFrame
        Quality report with missing values, dtypes, and uniqueness info
    """
    report = pd.DataFrame({
        'dtype':        df.dtypes,
        'missing':      df.isnull().sum(),
        'missing_%':    (df.isnull().sum() / len(df) * 100).round(2),
        'unique_vals':  df.nunique(),
        'min':          df.min(),
        'max':          df.max(),
    })
    return report

report = data_quality_report(df)
print('=== DATA QUALITY REPORT ===')
print(report.to_string())
print(f'\nTotal missing values: {df.isnull().sum().sum()}')
print(f'Duplicate rows: {df.duplicated().sum()}')

# Save raw dataset
df.to_csv("data/raw/california_housing_raw.csv", index=False)
print("Raw data saved to data/raw/california_housing_raw.csv")



