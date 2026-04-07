MSA8150 Final Project
=====================

Course: MSA8200
Team: add yur names here

------------------------------------------------------------------------
OVERVIEW
------------------------------------------------------------------------
Exploratory data analysis (EDA) and preprocessing pipeline for a sleep
and lifestyle dataset. The goal is to understand the data, identify
patterns, and prepare it for downstream modeling.

------------------------------------------------------------------------
PROJECT STRUCTURE
------------------------------------------------------------------------
  msa8150.py                   Main analysis script
  data.csv                     Raw dataset (not tracked in git)
  requirements.txt             Python dependencies
  statistical_summary.csv      Output: descriptive statistics
  numeric_features_histogram.png
  categorical_features_bar.png
  violin_by_gender.png
  violin_by_occupation.png
  boxplot_sleep_quality_score.png

------------------------------------------------------------------------
SETUP
------------------------------------------------------------------------
1. Clone the repository
     git clone <repo-url>
     cd <repo-folder>

2. Install dependencies
     pip install -r requirements.txt

3. Add the dataset
     Place data.csv in the project root. The file is not tracked in
     git — get it from a teammate.

4. Run the script
     python msa8150.py

------------------------------------------------------------------------
WHAT THE SCRIPT DOES
------------------------------------------------------------------------

Section 1 — Exploratory Data Analysis

  1.1 Load data
      Reads data.csv into a pandas DataFrame.

  1.2 Descriptive statistics
      Prints shape, column data types, statistical summary (also saved
      to statistical_summary.csv), missing value report, and unique
      value counts per column.

  1.3 Histograms
      Generates distribution plots for all numeric columns (with KDE
      overlay) and count bar charts for all categorical columns.
      Saved as PNG files.

  1.4 Violin plots
      Plots all numeric features grouped by Gender and Occupation to
      assess whether these categorical variables separate the
      distributions. Saved as PNG files.

Section 2 — Preprocessing & Cleaning

  2.1 Drop user_id
      Removed to prevent data leakage. num_cols is redefined after
      the drop.

  2.2 IQR outlier detection
      Flags outliers in each numeric column using the 1.5xIQR rule
      and prints a summary of detected outliers with bounds.

  2.3 Box plot for sleep_quality_score
      Visualizes the distribution with individual outlier points.
      Further analysis of the outliers.

------------------------------------------------------------------------
DEPENDENCIES
------------------------------------------------------------------------
  pandas        2.2.3
  numpy         2.1.3
  matplotlib    3.10.0
  seaborn       0.13.2

------------------------------------------------------------------------
NOTES
------------------------------------------------------------------------
- All plots are saved as PNG files in the project root and closed
  after saving (non-interactive mode).
