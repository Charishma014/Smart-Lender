"""
eda.py
------
Epic 2: Visualizing and Analysing the Data

Generates:
1. Univariate Analysis
2. Bivariate Analysis
3. Correlation Heatmap
4. Pair Plot

All plots are saved in:
static/eda/
"""

import os
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# -----------------------------
# Plot Style
# -----------------------------
sns.set_theme(style="whitegrid")

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "loan_data.csv")
OUT = os.path.join(BASE_DIR, "static", "eda")

os.makedirs(OUT, exist_ok=True)

# -----------------------------
# Read Dataset
# -----------------------------
df = pd.read_csv(DATA_PATH)

# ==========================================================
# UNIVARIATE ANALYSIS
# ==========================================================

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

sns.countplot(
    x="Loan_Status",
    hue="Loan_Status",
    data=df,
    ax=axes[0, 0],
    palette="viridis",
    legend=False,
)
axes[0, 0].set_title("Loan Status Distribution")

sns.countplot(
    x="Gender",
    hue="Gender",
    data=df,
    ax=axes[0, 1],
    palette="viridis",
    legend=False,
)
axes[0, 1].set_title("Gender Distribution")

sns.countplot(
    x="Education",
    hue="Education",
    data=df,
    ax=axes[0, 2],
    palette="viridis",
    legend=False,
)
axes[0, 2].set_title("Education Distribution")

sns.histplot(
    df["ApplicantIncome"],
    bins=30,
    ax=axes[1, 0],
    color="steelblue",
)

axes[1, 0].set_title("Applicant Income Distribution")

sns.histplot(
    df["LoanAmount"].dropna(),
    bins=30,
    ax=axes[1, 1],
    color="steelblue",
)

axes[1, 1].set_title("Loan Amount Distribution")

sns.countplot(
    x="Property_Area",
    hue="Property_Area",
    data=df,
    ax=axes[1, 2],
    palette="viridis",
    legend=False,
)

axes[1, 2].set_title("Property Area Distribution")

plt.tight_layout()
plt.savefig(os.path.join(OUT, "univariate.png"), dpi=120)
plt.close()

# ==========================================================
# BIVARIATE ANALYSIS
# ==========================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

sns.countplot(
    x="Credit_History",
    hue="Loan_Status",
    data=df,
    ax=axes[0, 0],
    palette="mako",
)

axes[0, 0].set_title("Credit History vs Loan Status")

sns.countplot(
    x="Married",
    hue="Loan_Status",
    data=df,
    ax=axes[0, 1],
    palette="mako",
)

axes[0, 1].set_title("Marital Status vs Loan Status")

sns.boxplot(
    x="Loan_Status",
    y="ApplicantIncome",
    hue="Loan_Status",
    data=df,
    ax=axes[1, 0],
    palette="mako",
    legend=False,
)

axes[1, 0].set_title("Applicant Income vs Loan Status")

sns.countplot(
    x="Property_Area",
    hue="Loan_Status",
    data=df,
    ax=axes[1, 1],
    palette="mako",
)

axes[1, 1].set_title("Property Area vs Loan Status")

plt.tight_layout()
plt.savefig(os.path.join(OUT, "bivariate.png"), dpi=120)
plt.close()

# ==========================================================
# MULTIVARIATE ANALYSIS
# ==========================================================

num_cols = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]

plt.figure(figsize=(7, 6))

sns.heatmap(
    df[num_cols].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
)

plt.title("Correlation Heatmap")
plt.tight_layout()

plt.savefig(os.path.join(OUT, "correlation_heatmap.png"), dpi=120)
plt.close()

# ==========================================================
# PAIRPLOT
# ==========================================================

pair_df = df[num_cols + ["Loan_Status"]].dropna()

pair_plot = sns.pairplot(
    pair_df,
    hue="Loan_Status",
    palette="viridis",
    corner=True,
)

pair_plot.savefig(os.path.join(OUT, "pairplot.png"), dpi=120)

plt.close("all")

print("\n====================================")
print("EDA Completed Successfully!")
print("Plots saved in:")
print(OUT)
print("====================================")