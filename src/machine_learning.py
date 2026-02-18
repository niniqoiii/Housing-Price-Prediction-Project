import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
os.makedirs("reports/figures", exist_ok=True)
os.makedirs("reports/results", exist_ok=True)

df_clean = pd.read_csv("data/processed/california_housing_clean.csv")

# ==============================
# Feature Engineering
# ==============================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from existing columns.

    Parameters:
    -----------
    df : pd.DataFrame
        Input cleaned DataFrame

    Returns:
    --------
    pd.DataFrame
        DataFrame with engineered features added
    """
    df_fe = df.copy()
    # Rooms and bedrooms per person
    df_fe['RoomsPerPerson']    = df_fe['AveRooms']    / df_fe['AveOccup'].clip(lower=0.1)
    df_fe['BedroomsPerRoom']   = df_fe['AveBedrms']   / df_fe['AveRooms'].clip(lower=0.1)
    df_fe['PopulationPerHH']   = df_fe['Population']  / df_fe['AveOccup'].clip(lower=0.1)
    # Age group: newer vs older homes
    df_fe['NewHome'] = (df_fe['HouseAge'] < 20).astype(int)
    return df_fe

df_fe = engineer_features(df_clean)

# Drop helper columns before modeling
if 'IncomeGroup' in df_fe.columns:
    df_fe.drop(columns=['IncomeGroup'], inplace=True)

# ==============================
# Train/Test Split
# ==============================
FEATURES = [c for c in df_fe.columns if c != 'MedHouseVal']
TARGET   = 'MedHouseVal'

X = df_fe[FEATURES]
y = df_fe[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f'Training set: {X_train.shape[0]} samples')
print(f'Test set:     {X_test.shape[0]} samples')
print(f'Features used: {FEATURES}')

# ==============================
# Feature Scaling (for LR)
# ==============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ==============================
# Model 1 — Linear Regression
# ==============================
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)

lr_r2   = r2_score(y_test, y_pred_lr)
lr_mse  = mean_squared_error(y_test, y_pred_lr)
lr_rmse = np.sqrt(lr_mse)
lr_mae  = mean_absolute_error(y_test, y_pred_lr)

print('=== Linear Regression Results ===')
print(f'R²   : {lr_r2:.4f}')
print(f'MSE  : {lr_mse:.4f}')
print(f'RMSE : {lr_rmse:.4f}')
print(f'MAE  : {lr_mae:.4f}')

# Cross-validation
cv_scores = cross_val_score(LinearRegression(), X_train_scaled, y_train, cv=5, scoring='r2')
print(f'Cross-Val R² (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')

# Feature coefficients
coef_df = pd.DataFrame({'Feature': FEATURES, 'Coefficient': lr_model.coef_}).sort_values('Coefficient', key=abs, ascending=True)
coef_df.to_csv("reports/results/linear_regression_coefficients.csv", index=False)

# Plot coefficients
plt.figure(figsize=(9,6))
colors = ['#E74C3C' if c < 0 else '#2ECC71' for c in coef_df['Coefficient']]
plt.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, edgecolor='white')
plt.axvline(0, color='black', linewidth=0.8)
plt.xlabel('Coefficient Value')
plt.title('Linear Regression — Feature Coefficients (Standardized)', fontweight='bold')
plt.tight_layout()
plt.savefig("reports/figures/linear_regression_coefficients.png", dpi=300)
plt.close()

# ==============================
# Model 2 — Decision Tree Regressor
# ==============================
dt_model = DecisionTreeRegressor(
    max_depth=8,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)
dt_model.fit(X_train, y_train)  
y_pred_dt = dt_model.predict(X_test)

dt_r2   = r2_score(y_test, y_pred_dt)
dt_mse  = mean_squared_error(y_test, y_pred_dt)
dt_rmse = np.sqrt(dt_mse)
dt_mae  = mean_absolute_error(y_test, y_pred_dt)

print('=== Decision Tree Regressor Results ===')
print(f'R²   : {dt_r2:.4f}')
print(f'MSE  : {dt_mse:.4f}')
print(f'RMSE : {dt_rmse:.4f}')
print(f'MAE  : {dt_mae:.4f}')

# Cross-validation
cv_dt = cross_val_score(
    DecisionTreeRegressor(max_depth=8, min_samples_split=20, min_samples_leaf=10, random_state=42),
    X_train, y_train, cv=5, scoring='r2'
)
print(f'Cross-Val R² (5-fold): {cv_dt.mean():.4f} ± {cv_dt.std():.4f}')

# Feature importance
importance_df = pd.DataFrame({'Feature': FEATURES, 'Importance': dt_model.feature_importances_}).sort_values('Importance', ascending=True)
importance_df.to_csv("reports/results/decision_tree_importances.csv", index=False)

# Plot feature importance
plt.figure(figsize=(9,6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='#9B59B6', edgecolor='white')
plt.xlabel('Importance Score')
plt.title('Decision Tree — Feature Importances', fontweight='bold')
plt.tight_layout()
plt.savefig("reports/figures/decision_tree_feature_importances.png", dpi=300)
plt.close()

# ==============================
# Save model metrics
# ==============================
metrics = pd.DataFrame({
    'Model': ['Linear Regression', 'Decision Tree Regressor'],
    'R2': [lr_r2, dt_r2],
    'MSE': [lr_mse, dt_mse],
    'RMSE': [lr_rmse, dt_rmse],
    'MAE': [lr_mae, dt_mae]
})
metrics.to_csv("reports/results/model_metrics.csv", index=False)
print("Model metrics saved to reports/results/model_metrics.csv")


# ==== Model Comparison & Conclusions ====

# -----------------------------
# Side-by-Side Metrics Comparison
# -----------------------------
results = pd.DataFrame({
    'Model':  ['Linear Regression', 'Decision Tree Regressor'],
    'R²':     [round(lr_r2, 4),   round(dt_r2, 4)],
    'MSE':    [round(lr_mse, 4),  round(dt_mse, 4)],
    'RMSE':   [round(lr_rmse, 4), round(dt_rmse, 4)],
    'MAE':    [round(lr_mae, 4),  round(dt_mae, 4)],
})

print('=== MODEL COMPARISON ===')
print(results.to_string(index=False))

# Save metrics to CSV for reports/results
results.to_csv("reports/results/model_comparison.csv", index=False)
print("\nModel comparison saved to reports/results/model_comparison.csv")

# -----------------------------
# Determine better model based on R²
# -----------------------------
winner = 'Decision Tree Regressor' if dt_r2 > lr_r2 else 'Linear Regression'
print(f'\nBetter R² Score: {winner}')

# -----------------------------
# Visualization 7: Metrics Bar Plot
# -----------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

metrics   = ['R²', 'RMSE', 'MAE']
lr_vals   = [lr_r2, lr_rmse, lr_mae]
dt_vals   = [dt_r2, dt_rmse, dt_mae]
x         = np.arange(1)
width     = 0.3

for i, (ax, metric, lv, dv) in enumerate(zip(axes, metrics, lr_vals, dt_vals)):
    bars1 = ax.bar(x - width/2, lv, width, label='Linear Regression', color='#3498DB', edgecolor='white')
    bars2 = ax.bar(x + width/2, dv, width, label='Decision Tree',      color='#E74C3C', edgecolor='white')
    ax.set_title(metric)
    ax.set_xticks([])
    ax.legend(fontsize=8)
    ax.bar_label(bars1, fmt='%.3f', padding=3, fontsize=9)
    ax.bar_label(bars2, fmt='%.3f', padding=3, fontsize=9)

plt.suptitle('Visualization 7: Model Performance Comparison', fontweight='bold')
plt.tight_layout()
plt.savefig("reports/figures/model_comparison_metrics.png", dpi=300)
plt.close()
print("Saved bar plot comparison to reports/figures/model_comparison_metrics.png")

# -----------------------------
# Visualization 8: Actual vs Predicted
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, name, preds, color in zip(
    axes,
    ['Linear Regression', 'Decision Tree Regressor'],
    [y_pred_lr, y_pred_dt],
    ['#3498DB', '#E74C3C']
):
    ax.scatter(y_test, preds, alpha=0.3, color=color, s=10)
    lim = [min(y_test.min(), preds.min()) - 0.1,
           max(y_test.max(), preds.max()) + 0.1]
    ax.plot(lim, lim, 'k--', linewidth=1.5, label='Perfect prediction')
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel('Actual Values ($100k)')
    ax.set_ylabel('Predicted Values ($100k)')
    r2 = r2_score(y_test, preds)
    ax.set_title(f'{name}\nR² = {r2:.4f}', fontweight='bold')
    ax.legend(fontsize=8)

plt.suptitle('Visualization 8: Actual vs Predicted House Values', fontweight='bold')
plt.tight_layout()
plt.savefig("reports/figures/actual_vs_predicted.png", dpi=300)
plt.close()
print("Saved actual vs predicted plot to reports/figures/actual_vs_predicted.png")

# -----------------------------
# Summary Report
# -----------------------------
print('=' * 55)
print('       FINAL PROJECT — SUMMARY REPORT')
print('=' * 55)
print(f'Dataset         : California Housing (sklearn)')
print(f'Total Records   : {len(df_fe):,}')
print(f'Features Used   : {len(FEATURES)}')
print(f'Train / Test    : {len(X_train):,} / {len(X_test):,}')
print()
print(f'Linear Regression   R² = {lr_r2:.4f}  RMSE = {lr_rmse:.4f}')
print(f'Decision Tree       R² = {dt_r2:.4f}  RMSE = {dt_rmse:.4f}')
print()
best = 'Decision Tree Regressor' if dt_r2 > lr_r2 else 'Linear Regression'
print(f'Best Model      : {best}')
print('=' * 55)
