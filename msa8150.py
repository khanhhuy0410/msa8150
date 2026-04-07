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

# Categorical columns — bar charts (count plots)
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


# --- 1.4 Violin plots — numeric features by gender and occupation ---
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


# =============================================================================
# SECTION 2: DATA PREPROCESSING & CLEANING
# =============================================================================

# 2.1 --- Drop user_id column (data leakage) ---
df = df.drop(columns="user_id")
num_cols = df.select_dtypes(include="number").columns.tolist()

# 2.2 --- Handle outliers using IQR Method ---

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

# 2.3 --- Box plot for sleep_quality_score outliers ---
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
