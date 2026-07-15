"""
train_model.py
--------------
Modernized ML Pipeline with:
1. ColumnTransformer for clean preprocessing (Imputation, OneHot Encoding, Scaling).
2. Proper Train-Test Split before balancing to prevent data leakage.
3. Oversampling of minority class applied only to the training set.
4. Hyperparameter tuning via GridSearchCV for Random Forest and XGBoost.
5. Saving of the best-performing end-to-end Pipeline to model/pipeline.pkl.
6. Exporting of evaluation charts (ROC Curve, Confusion Matrix) to static/model_metrics/.
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")

# Try importing XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

RANDOM_STATE = 42

# Project Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "loan_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
METRICS_DIR = os.path.join(BASE_DIR, "static", "model_metrics")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

PIPELINE_PATH = os.path.join(MODEL_DIR, "pipeline.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.json")

# ============================================================
# 1. Load and Clean Dataset
# ============================================================
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

if "Loan_ID" in df.columns:
    df = df.drop(columns=["Loan_ID"])

# Feature Groups
CATEGORICAL = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
NUMERIC = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]
TARGET = "Loan_Status"

# Standardize Dependents values in dataframe (e.g. "3+" to "3")
if "Dependents" in df.columns:
    df["Dependents"] = df["Dependents"].astype(str).str.replace("3+", "3", regex=False)
    df["Dependents"] = df["Dependents"].replace("nan", np.nan)

# Encode Target: Y -> 1, N -> 0
df[TARGET] = df[TARGET].map({"Y": 1, "N": 0})

X = df[CATEGORICAL + NUMERIC]
y = df[TARGET]

# ============================================================
# 2. Train-Test Split (BEFORE balancing to prevent leakage!)
# ============================================================
print("Splitting dataset into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

# ============================================================
# 3. Balance Training Set (Only upsample train data)
# ============================================================
print("Balancing training set...")
train_df = pd.concat([X_train, y_train], axis=1)
majority = train_df[train_df[TARGET] == 1]
minority = train_df[train_df[TARGET] == 0]

minority_upsampled = minority.sample(
    n=len(majority),
    replace=True,
    random_state=RANDOM_STATE
)

balanced_train = pd.concat([majority, minority_upsampled]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
X_train_bal = balanced_train[CATEGORICAL + NUMERIC]
y_train_bal = balanced_train[TARGET]

# ============================================================
# 4. Define Preprocessing Pipeline
# ============================================================
print("Defining preprocessing ColumnTransformer...")

# Numeric transformer: Impute with median, scale
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical transformer: Impute with mode, OneHot encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, NUMERIC),
        ('cat', categorical_transformer, CATEGORICAL)
    ]
)

# ============================================================
# 5. Define Models & Grid Search Parameters
# ============================================================
print("Setting up models for training & hyperparameter search...")

classifiers = {
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE) if HAS_XGBOOST else HistGradientBoostingClassifier(random_state=RANDOM_STATE)
}

# Grid Search parameters to get optimized models
param_grids = {
    "Decision Tree": {
        "classifier__max_depth": [4, 6, 8, 10],
        "classifier__min_samples_split": [2, 5, 10]
    },
    "Random Forest": {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [6, 8, 10],
        "classifier__min_samples_split": [2, 5]
    },
    "KNN": {
        "classifier__n_neighbors": [3, 5, 7, 9],
        "classifier__weights": ["uniform", "distance"]
    },
    "XGBoost": {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [3, 4, 5],
        "classifier__learning_rate": [0.01, 0.05, 0.1]
    }
}

# ============================================================
# 6. Train and Evaluate Models
# ============================================================
results = {}
trained_pipelines = {}
test_predictions = {}
test_probabilities = {}

for name, clf in classifiers.items():
    print("-" * 50)
    print(f"Training model: {name}")
    print("-" * 50)
    
    # Create the full pipeline: Preprocessor + Model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    
    # Run GridSearch
    grid_search = GridSearchCV(
        pipeline,
        param_grids[name],
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    grid_search.fit(X_train_bal, y_train_bal)
    best_pipe = grid_search.best_estimator_
    
    # Evaluate
    train_pred = best_pipe.predict(X_train_bal)
    test_pred = best_pipe.predict(X_test)
    
    train_acc = accuracy_score(y_train_bal, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"Best Params       : {grid_search.best_params_}")
    print(f"Training Accuracy : {train_acc * 100:.2f}%")
    print(f"Testing Accuracy  : {test_acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, test_pred, target_names=["Denied (N)", "Approved (Y)"]))
    
    results[name] = {
        "train_accuracy": round(train_acc * 100, 2),
        "test_accuracy": round(test_acc * 100, 2),
        "best_params": {k.replace("classifier__", ""): v for k, v in grid_search.best_params_.items()}
    }
    
    trained_pipelines[name] = best_pipe
    test_predictions[name] = test_pred
    
    # Get probability scores for ROC curve
    if hasattr(best_pipe, "predict_proba"):
        test_probabilities[name] = best_pipe.predict_proba(X_test)[:, 1]
    elif hasattr(best_pipe, "decision_function"):
        test_probabilities[name] = best_pipe.decision_function(X_test)
    else:
        test_probabilities[name] = test_pred

# ============================================================
# 7. Select and Save the Best Model
# ============================================================
best_model_name = max(results, key=lambda x: results[x]["test_accuracy"])
best_pipeline = trained_pipelines[best_model_name]

print("=" * 60)
print(f"BEST MODEL SELECTED: {best_model_name} with {results[best_model_name]['test_accuracy']}% Accuracy")
print("=" * 60)

# Save best pipeline
with open(PIPELINE_PATH, "wb") as f:
    pickle.dump(best_pipeline, f)

# Save metadata
metadata = {
    "features": CATEGORICAL + NUMERIC,
    "categorical_features": CATEGORICAL,
    "numeric_features": NUMERIC,
    "target": TARGET,
    "best_model": best_model_name,
    "used_real_xgboost": HAS_XGBOOST,
    "results": results
}
with open(METADATA_PATH, "w") as f:
    json.dump(metadata, f, indent=4)

# ============================================================
# 8. Generate and Save Performance Visualizations
# ============================================================
print("\nGenerating model metrics charts...")

# Chart 1: Confusion Matrix for the Best Model
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, test_predictions[best_model_name])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["Denied (N)", "Approved (Y)"],
            yticklabels=["Denied (N)", "Approved (Y)"])
plt.title(f"Confusion Matrix: {best_model_name} (Best Model)")
plt.ylabel("Actual Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig(os.path.join(METRICS_DIR, "confusion_matrix.png"), dpi=150)
plt.close()

# Chart 2: ROC Curve Comparison
plt.figure(figsize=(8, 6))
for name, probs in test_probabilities.items():
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Comparison")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(METRICS_DIR, "roc_curve.png"), dpi=150)
plt.close()

print(f"Pipeline saved to: {PIPELINE_PATH}")
print(f"Metadata saved to: {METADATA_PATH}")
print(f"Evaluation charts saved to: {METRICS_DIR}")
print("Training Pipeline Completed Successfully!")