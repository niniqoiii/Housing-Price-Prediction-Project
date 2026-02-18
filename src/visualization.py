import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ======================================================
# Visualization
# ======================================================

df_clean = pd.read_csv("data/processed/california_housing_clean.csv")

# Target Variable Distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Histogram with KDE
sns.histplot(df_clean['MedHouseVal'], bins=50, kde=True, ax=ax1, color='#2E86AB')
ax1.axvline(df_clean['MedHouseVal'].mean(), color='red', linestyle='--', label=f"Mean: {df_clean['MedHouseVal'].mean():.2f}")
ax1.axvline(df_clean['MedHouseVal'].median(), color='orange', linestyle='--', label=f"Median: {df_clean['MedHouseVal'].median():.2f}")
ax1.set_title('Distribution of Median House Values')
ax1.set_xlabel('Median House Value ($100k)')
ax1.set_ylabel('Count')
ax1.legend()

# Box plot
ax2.boxplot(df_clean['MedHouseVal'], patch_artist=True,
            boxprops=dict(facecolor='#AED6F1', color='#2874A6'),
            medianprops=dict(color='red', linewidth=2),
            flierprops=dict(marker='o', markerfacecolor='#E74C3C', markersize=3, alpha=0.5))
ax2.set_title('Box Plot — Median House Values')
ax2.set_ylabel('Median House Value ($100k)')
ax2.set_xticks([])

plt.suptitle('Visualization 1: Target Variable — Median House Value', fontweight='bold')
plt.tight_layout()
plt.savefig("reports/figures/vis1_target_distribution.png", dpi=300)
plt.close()

print(df_clean['MedHouseVal'].describe())

# Correlation Heatmap
plt.figure(figsize=(11, 8))
corr_matrix = df_clean.corr(numeric_only=True)

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # hide upper triangle for clarity
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, linewidths=0.5, square=True, annot_kws={'size': 9},
            cbar_kws={'shrink': 0.8})

plt.title('Visualization 2: Correlation Heatmap — All Features', fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig("reports/figures/vis2_correlation_heatmap.png", dpi=300)
plt.close()

# Print top correlations with target
print('Top correlations with MedHouseVal:')
print(corr_matrix['MedHouseVal'].sort_values(ascending=False).to_string())

# Scatter Plots — Price vs Key Features
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

scatter_features = ['MedInc', 'AveRooms', 'HouseAge']
colors = ['#3498DB', '#2ECC71', '#E74C3C']

for ax, feat, color in zip(axes, scatter_features, colors):
    # Sample 2000 points for clarity
    sample = df_clean.sample(2000, random_state=42)
    ax.scatter(sample[feat], sample['MedHouseVal'], alpha=0.3, color=color, s=10)

    # Trend line
    z = np.polyfit(df_clean[feat], df_clean['MedHouseVal'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df_clean[feat].min(), df_clean[feat].max(), 200)
    ax.plot(x_line, p(x_line), color='black', linewidth=2, linestyle='--', label='Trend line')

    ax.set_xlabel(feat)
    ax.set_ylabel('MedHouseVal ($100k)')
    ax.set_title(f'{feat} vs House Value')
    ax.legend(fontsize=8)

plt.suptitle('Visualization 3: Scatter Plots — Key Features vs House Price', fontweight='bold')
plt.tight_layout()
plt.savefig("reports/figures/vis3_scatter_key_features.png", dpi=300)
plt.close()

# Feature Distributions (Histogram Grid)
features = [c for c in df_clean.columns if c != 'MedHouseVal']
n_features = len(features)

n_cols = 4
n_rows = (n_features + n_cols - 1) // n_cols  # ceil division
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4))
axes = axes.flatten()

palette = sns.color_palette('muted', n_features)

for i, (feat, color) in enumerate(zip(features, palette)):
    sns.histplot(df_clean[feat], bins=40, ax=axes[i], color=color, kde=True, edgecolor='none')
    axes[i].set_title(feat)
    axes[i].set_xlabel('')

# Hide any extra axes
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle('Visualization 4: Feature Distributions (Histograms + KDE)', fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig("reports/figures/vis4_feature_distributions.png", dpi=300)
plt.close()


# Geographical Distribution — House Values by Location
plt.figure(figsize=(11, 7))
scatter = plt.scatter(
    df_clean['Longitude'], df_clean['Latitude'],
    c=df_clean['MedHouseVal'], cmap='RdYlGn',
    alpha=0.4, s=df_clean['Population'] / df_clean['Population'].max() * 40 + 1,
    edgecolors='none'
)
plt.colorbar(scatter, label='Median House Value ($100k)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Visualization 5: California House Values by Geographic Location\n(dot size = relative population)', fontweight='bold')
plt.tight_layout()
plt.savefig("reports/figures/vis5_geographical_distribution.png", dpi=300)
plt.close()

print('Insight: Higher-value properties cluster along the coast (LA, Bay Area).')

# Violin Plots — House Value by Income Group
# Bin median income into 4 groups
df_clean['IncomeGroup'] = pd.cut(
    df_clean['MedInc'],
    bins=[0, 2, 4, 6, df_clean['MedInc'].max()],
    labels=['Low (<2)', 'Lower-Mid (2-4)', 'Upper-Mid (4-6)', 'High (6+)']
)

plt.figure(figsize=(12, 6))
sns.violinplot(data=df_clean, x='IncomeGroup', y='MedHouseVal',
               palette='Set2', inner='quartile')
plt.xlabel('Median Income Group (in $10,000s)')
plt.ylabel('Median House Value ($100k)')
plt.title('Visualization 6: House Value Distribution by Income Group (Violin Plot)', fontweight='bold')
plt.tight_layout()
plt.savefig("reports/figures/vis6_incomegroup_violin.png", dpi=300)
plt.close()

print('Insight: Higher income groups show significantly higher and more variable house prices.')

# Statistical Summary
print('=== DESCRIPTIVE STATISTICS ===')
print(df_clean.describe().round(3).to_string())

print('\n=== SKEWNESS (measures distribution asymmetry) ===')
print(df_clean.select_dtypes(include=np.number).skew().round(3).to_string())

print('\n=== STRONG CORRELATIONS WITH TARGET ===')
corr = df_clean.corr(numeric_only=True)['MedHouseVal'].drop('MedHouseVal')
strong = corr[corr.abs() > 0.1].sort_values(ascending=False)
print(strong.round(3).to_string())

