# Title: Final Project
# Course: MSA8200
# Collaborators: SQL-RACIST CHAT 🚀🚀🚀 -- insert name here
# Purpose: to be good data science bois n get a JOB ദ്ദി(˵ •̀ ᴗ - ˵ ) ✧ -- figure out later

# Import dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# SECTION 1: EXPLORATORY DATA ANALYSIS
# =============================================================================

# --- 1.1  Load data ---
df = pd.read_csv('data.csv')

#  --- 1.2 Descriptive statistics ---
print(f"Shape: {df.shape[0]} rows and {df.shape[1]} columns.\n")

print("Column data types:")
print(df.dtypes)

print("\nStatistical summary:")
print(df.describe().round(0))
df.describe().round(2).to_csv("statistical_summary.csv")

missing = df.isnull().sum()
if missing.sum() > 0:
    print(f"Missing values: {missing.sum()}")
    print(missing[missing > 0])
else:
    print("\nNo missing values\n")

n_dupes = df.duplicated().sum()
if n_dupes > 0:
    print(f"Duplicate rows: {n_dupes}")
    df = df.drop_duplicates()
    print(f"Duplicates removed. New shape: {df.shape[0]} rows and {df.shape[1]} columns.")
else:
    print("No duplicate rows\n")

print("Unique values")
print(df.nunique())

# --- 1.3 Histograms for all columns ---
num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

# Numeric columns — histograms
if num_cols:
    n_cols = 3
    n_rows = -(-len(num_cols) // n_cols)  # ceiling division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()
    for i, col in enumerate(num_cols):
        sns.histplot(data=df, x=col, bins=20, kde=True, ax=axes[i])
        axes[i].set_title(col)
        axes[i].set_xlabel("")
    for j in range(i + 1, len(axes)):  # hide unused subplots
        axes[j].set_visible(False)
    fig.suptitle("Numeric Features — Distributions", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("numeric_features_histogram.png")
    plt.close()

# --- 1.4 Bar chart for categorical columns ---
if cat_cols:
    n_cols = 3
    n_rows = -(-len(cat_cols) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()
    for i, col in enumerate(cat_cols):
        order = df[col].value_counts().index
        sns.countplot(data=df, x=col, order=order, ax=axes[i])
        axes[i].set_title(col)
        axes[i].set_xlabel("")
        axes[i].tick_params(axis="x", rotation=30)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Categorical Features — Value Counts", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("categorical_features_bar.png")
    plt.close()


# --- 1.5 Violin plots — grouped by gender and occupation ---
col_map = {c.lower(): c for c in df.columns}
group_cols = [col_map[c] for c in ["gender", "occupation"] if c in col_map]

for group in group_cols:
    n_cols = 3
    n_rows = -(-len(num_cols) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()
    for i, col in enumerate(num_cols):
        sns.violinplot(data=df, x=group, y=col, ax=axes[i])
        axes[i].set_title(col)
        axes[i].set_xlabel("")
        axes[i].tick_params(axis="x", rotation=30)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"Numeric Features by {group}", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(f"violin_by_{group.lower()}.png")
    plt.close()

# --- 1.6 Drop user_id column (data leakage) ---
df = df.drop(columns="user_id")
num_cols = df.select_dtypes(include="number").columns.tolist()

# --- 1.7 IQR Outlier detection method ---
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    print(
    f"    {col}: {n_outliers} outlier(s) detected "
    f"[bounds {lower:.3f}, {upper:.3f}]")

# 1.8 --- Box plot for sleep_quality_score outliers ---
Q1 = df["sleep_quality_score"].quantile(0.25)
Q3 = df["sleep_quality_score"].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
outliers = df[(df["sleep_quality_score"] < lower) | (df["sleep_quality_score"] > upper)]

fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df, x="sleep_quality_score", ax=ax)
sns.stripplot(data=outliers, x="sleep_quality_score", ax=ax,
              color="red", size=6, jitter=True, alpha=0.8)
ax.set_title("sleep_quality_score — Box and Whisker Plot (red = outliers)")
plt.tight_layout()
plt.savefig("boxplot_sleep_quality_score.png")
plt.close()

# --- 1.9 Scatterplot / pairwise relationships --
pair_plot = sns.pairplot(df[num_cols], plot_kws={"alpha": 0.3, "s": 10}, diag_kind="kde")
pair_plot.fig.suptitle("Pairwise Scatterplots — Numeric Features", y=1.01, fontsize=14)
plt.savefig("pairplot_numeric.png")
plt.close()

# --- 1.10 Pearson Correlation Analysis
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, square=True, linewidths=0.5, ax=ax)
ax.set_title("Correlation Matrix — Numeric Features")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()
# corr.to_csv("correlation_matrix.csv")

corr_pairs = (corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
              .stack()
              .reset_index())
corr_pairs.columns = ["feature_1", "feature_2", "correlation"]
corr_pairs["abs_correlation"] = corr_pairs["correlation"].abs()
corr_pairs = corr_pairs.sort_values("abs_correlation", ascending=False)

print("\nTop 10 correlated pairs:")
print(corr_pairs.head(10).to_string(index=False))

# To do next: Up to this point, EDA has been very broad; all pairplots/correlations are done without a target variable
# Once target variable is picked (most likely Sleep Quality Score), run plots for numeric variables against target.
