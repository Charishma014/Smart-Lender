"""
generate_dataset.py
--------------------
Generates a synthetic loan-applicant dataset that mirrors the schema and
statistical shape of the well-known "Loan Prediction" dataset used in the
Smart Lender project (Gender, Married, Dependents, Education, Self_Employed,
ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term,
Credit_History, Property_Area, Loan_Status).

Run once to (re)build data/loan_data.csv. Replace this file with the real
Kaggle "Loan Prediction Problem Dataset" CSV in production -- the rest of
the pipeline (train_model.py / app.py) expects the same column names.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 614  # matches the size of the classic Loan Prediction dataset

gender = np.random.choice(["Male", "Female"], N, p=[0.81, 0.19])
married = np.random.choice(["Yes", "No"], N, p=[0.65, 0.35])
dependents = np.random.choice(["0", "1", "2", "3+"], N, p=[0.58, 0.17, 0.17, 0.08])
education = np.random.choice(["Graduate", "Not Graduate"], N, p=[0.78, 0.22])
self_employed = np.random.choice(["Yes", "No"], N, p=[0.14, 0.86])
property_area = np.random.choice(["Urban", "Semiurban", "Rural"], N, p=[0.38, 0.38, 0.24])

# Income distributions (right-skewed, like real income data)
applicant_income = np.round(np.random.lognormal(mean=8.35, sigma=0.55, size=N)).astype(int)
coapplicant_income = np.round(
    np.where(married == "Yes",
             np.random.lognormal(mean=7.2, sigma=1.0, size=N),
             0)
).astype(int)
coapplicant_income = np.where(np.random.rand(N) < 0.35, 0, coapplicant_income)

loan_amount_term = np.random.choice(
    [360, 180, 480, 300, 240, 120, 60, 84, 36, 12],
    N, p=[0.72, 0.09, 0.03, 0.03, 0.03, 0.03, 0.02, 0.02, 0.02, 0.01]
)

credit_history = np.random.choice([1.0, 0.0], N, p=[0.84, 0.16])

# Loan amount roughly correlated with combined income
combined_income = applicant_income + coapplicant_income
loan_amount = np.round(
    (combined_income / 45) + np.random.normal(0, 25, N)
).clip(9, 700).astype(int)

# Inject some missing values, like the real dataset
def inject_nan(arr, frac, dtype=object):
    arr = arr.astype(dtype)
    idx = np.random.choice(len(arr), int(len(arr) * frac), replace=False)
    arr[idx] = np.nan
    return arr

gender = inject_nan(gender, 0.02)
married = inject_nan(married, 0.005)
dependents = inject_nan(dependents, 0.025)
self_employed = inject_nan(self_employed, 0.05)
loan_amount = inject_nan(loan_amount.astype(float), 0.035, dtype=float)
loan_amount_term = inject_nan(loan_amount_term.astype(float), 0.02, dtype=float)
credit_history = inject_nan(credit_history.astype(float), 0.08, dtype=float)

# Build a "true" approval probability from a logistic-ish combination of
# features so the models have real signal to learn (credit history and
# income are the dominant drivers, matching real-world loan approval).
def approval_prob(row):
    score = -1.2
    if row["Credit_History"] == 1.0:
        score += 3.0
    elif pd.isna(row["Credit_History"]):
        score += 0.3
    score += 0.35 if row["Property_Area"] == "Semiurban" else (0.0 if row["Property_Area"] == "Urban" else -0.3)
    score += 0.25 if row["Education"] == "Graduate" else -0.25
    inc = (row["ApplicantIncome"] + row["CoapplicantIncome"])
    score += 0.4 if inc > 5000 else (0.0 if inc > 2500 else -0.5)
    if not pd.isna(row["LoanAmount"]) and row["LoanAmount"] > 0:
        ratio = inc / (row["LoanAmount"] + 1)
        score += 0.5 if ratio > 25 else (0.0 if ratio > 12 else -0.6)
    score += np.random.normal(0, 0.6)
    return 1 / (1 + np.exp(-score))

df = pd.DataFrame({
    "Loan_ID": [f"LP{100000 + i}" for i in range(N)],
    "Gender": gender,
    "Married": married,
    "Dependents": dependents,
    "Education": education,
    "Self_Employed": self_employed,
    "ApplicantIncome": applicant_income,
    "CoapplicantIncome": coapplicant_income,
    "LoanAmount": loan_amount,
    "Loan_Amount_Term": loan_amount_term,
    "Credit_History": credit_history,
    "Property_Area": property_area,
})

probs = df.apply(approval_prob, axis=1)
df["Loan_Status"] = np.where(np.random.rand(N) < probs, "Y", "N")

df.to_csv("data/loan_data.csv", index=False)
print(df.shape)
print(df.isna().sum())
print(df["Loan_Status"].value_counts())
