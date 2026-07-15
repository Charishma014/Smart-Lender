"""
app.py
------
Production-grade Flask application for Smart Lender.
Integrates:
1. Structured Logging (writes to console and app.log).
2. MongoDB integration for auditing and user-specific dashboard histories.
3. Dynamic authentication accepting any username & password combination.
4. Loading unified scikit-learn Pipeline (zero train-serving skew).
5. REST API Endpoint (/api/predict) with validation and API log tracking.
"""

import os
import json
import pickle
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from bson import ObjectId

import config
from database import db_instance

# ============================================================
# 1. Setup Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("smart_lender_app")

app = Flask(__name__)
app.secret_key = config.SECRET_KEY

# ============================================================
# 2. Database & Model Initialization
# ============================================================
db_instance.connect()

pipeline = None
metadata = {}
try:
    if os.path.exists(config.PIPELINE_PATH):
        with open(config.PIPELINE_PATH, "rb") as f:
            pipeline = pickle.load(f)
        logger.info(f"Loaded ML model pipeline from {config.PIPELINE_PATH}")
    else:
        logger.error(f"Pipeline file not found at {config.PIPELINE_PATH}. Run train_model.py first.")

    if os.path.exists(config.METADATA_PATH):
        with open(config.METADATA_PATH, "r") as f:
            metadata = json.load(f)
        logger.info("Loaded model metadata successfully.")
    else:
        logger.warning(f"Metadata file not found at {config.METADATA_PATH}.")
except Exception as e:
    logger.exception(f"Error loading model files: {e}")

FEATURES = metadata.get("features", [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History", "Property_Area"
])
BEST_MODEL_NAME = metadata.get("best_model", "Classifier")

# ============================================================
# Helpers
# ============================================================
class MongoJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        if isinstance(o, datetime):
            return o.isoformat()
        return super().default(o)

def prepare_input_dataframe(raw_data):
    mapped_data = {
        "Gender": str(raw_data.get("gender", "Male")),
        "Married": str(raw_data.get("married", "No")),
        "Dependents": str(raw_data.get("dependents", "0")),
        "Education": str(raw_data.get("education", "Graduate")),
        "Self_Employed": str(raw_data.get("self_employed", "No")),
        "Property_Area": str(raw_data.get("property_area", "Urban")),
        "ApplicantIncome": float(raw_data.get("applicant_income", 0)),
        "CoapplicantIncome": float(raw_data.get("coapplicant_income", 0)),
        "LoanAmount": float(raw_data.get("loan_amount", 0)),
        "Loan_Amount_Term": float(raw_data.get("loan_amount_term", 360)),
        "Credit_History": float(raw_data.get("credit_history", 1.0))
    }
    mapped_data["Dependents"] = mapped_data["Dependents"].replace("3+", "3")
    df = pd.DataFrame([mapped_data])
    return df[FEATURES], mapped_data

# ============================================================
# 3. Authentication Routes
# ============================================================
@app.route("/login", methods=["GET", "POST"])
def login():
    """Login page — verifies credentials against users collection."""
    if "username" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if not username or not password:
            return render_template("login.html", error="Please enter both username and password.")

        # Access users collection
        users_col = db_instance.get_users_collection()
        if users_col is None:
            return render_template("login.html", error="Database connection is currently unavailable. Please try again later.")

        try:
            # Look up the user
            user = users_col.find_one({"username": username})
            if user:
                if user.get("password") == password:
                    session["username"] = username
                    session["role"] = "customer"
                    logger.info(f"User '{username}' logged in.")
                    return redirect(url_for("dashboard"))
                else:
                    return render_template("login.html", error="Incorrect password. Please try again.")
            else:
                return render_template("login.html", error="Username not found. Please Sign Up if you don't have an account.")
        except Exception as e:
            logger.error(f"Error checking user login: {e}")
            return render_template("login.html", error="An error occurred during sign in. Please try again.")

    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    """Sign-up page — registers new users in the users collection."""
    if "username" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm  = request.form.get("confirm_password", "")

        if not username or not password:
            return render_template("signup.html", error="Please fill in all fields.")
        if password != confirm:
            return render_template("signup.html", error="Passwords do not match. Please try again.")
        if len(password) < 4:
            return render_template("signup.html", error="Password must be at least 4 characters long.")

        # Access users collection
        users_col = db_instance.get_users_collection()
        if users_col is None:
            return render_template("signup.html", error="Database connection is currently unavailable. Please try again later.")

        try:
            # Check if user already exists
            existing_user = users_col.find_one({"username": username})
            if existing_user:
                return render_template("signup.html", error="Username already exists. Please choose a different name.")

            # Insert new user document
            user_doc = {
                "username": username,
                "password": password,
                "created_at": datetime.utcnow()
            }
            users_col.insert_one(user_doc)
            logger.info(f"New user doc created for '{username}'.")

            # Log user in
            session["username"] = username
            session["role"] = "customer"
            return redirect(url_for("dashboard"))
        except Exception as e:
            logger.error(f"Error registering user: {e}")
            return render_template("signup.html", error="An error occurred during account creation. Please try again.")

    return render_template("signup.html")


@app.route("/logout")
def logout():
    username = session.get("username")
    session.clear()
    logger.info(f"User '{username}' logged out.")
    return redirect(url_for("home"))

# ============================================================
# 4. Public Pages
# ============================================================
@app.route("/")
def home():
    """Public landing page."""
    session_user = session.get("username")
    return render_template("home.html", session_user=session_user)


@app.route("/loan-info")
def loan_info():
    """Loan information & educational guide page."""
    logged_in = "username" in session
    return render_template("loan_info.html", logged_in=logged_in)

# ============================================================
# 5. Customer Portal Routes (Login Required)
# ============================================================
@app.route("/dashboard")
def dashboard():
    """Customer personal dashboard with application history."""
    if "username" not in session:
        return redirect(url_for("login"))

    username = session.get("username")
    collection = db_instance.get_predictions_collection()

    total_apps   = 0
    approved_count = 0
    denied_count   = 0
    max_eligible_loan = 0.0
    logs = []

    if collection is not None:
        try:
            logs = list(collection.find({"username": username}).sort("timestamp", -1))
            total_apps = len(logs)
            for log in logs:
                log["_id"] = str(log["_id"])
            if total_apps > 0:
                approved_count = sum(1 for r in logs if r.get("approved") is True)
                denied_count   = total_apps - approved_count
                approved_loans = [r.get("loan_amount", 0) for r in logs if r.get("approved") is True]
                max_eligible_loan = max(approved_loans) if approved_loans else 0.0
        except Exception as e:
            logger.error(f"Error fetching customer data from MongoDB: {e}")

    return render_template(
        "customer_dashboard.html",
        username=username,
        total_apps=total_apps,
        approved_count=approved_count,
        denied_count=denied_count,
        max_eligible_loan=max_eligible_loan,
        logs=logs
    )


@app.route("/predict")
def predict_form():
    """Loan application form — protected to logged-in users."""
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("predict.html")


@app.route("/submit", methods=["POST"])
def submit():
    """Handles loan application prediction submissions."""
    if "username" not in session:
        return redirect(url_for("login"))

    if pipeline is None:
        return "Model pipeline is currently unavailable. Please run training first.", 503

    try:
        form_data = request.form
        X_input, raw = prepare_input_dataframe(form_data)

        pred     = int(pipeline.predict(X_input)[0])
        approved = (pred == 1)

        confidence = 0.0
        if hasattr(pipeline, "predict_proba"):
            probs      = pipeline.predict_proba(X_input)[0]
            confidence = round(float(probs[pred]) * 100, 1)

        # Log to MongoDB linked to user session
        collection = db_instance.get_predictions_collection()
        username   = session["username"]

        record = {
            "username": username,
            "gender": raw["Gender"],
            "married": raw["Married"],
            "dependents": raw["Dependents"],
            "education": raw["Education"],
            "self_employed": raw["Self_Employed"],
            "property_area": raw["Property_Area"],
            "applicant_income": raw["ApplicantIncome"],
            "coapplicant_income": raw["CoapplicantIncome"],
            "loan_amount": raw["LoanAmount"],
            "loan_amount_term": raw["Loan_Amount_Term"],
            "credit_history": raw["Credit_History"],
            "approved": approved,
            "confidence": confidence,
            "model_used": BEST_MODEL_NAME,
            "timestamp": datetime.utcnow()
        }

        if collection is not None:
            collection.insert_one(record)
            logger.info(f"Saved loan decision to MongoDB for user '{username}'")
        else:
            logger.warning("MongoDB not connected. Prediction not archived.")

        # Display values for result page
        applicant_display = {
            "Gender": raw["Gender"],
            "Married": raw["Married"],
            "Dependents": raw["Dependents"],
            "Education": raw["Education"],
            "Self_Employed": raw["Self_Employed"],
            "Property_Area": raw["Property_Area"],
            "ApplicantIncome": raw["ApplicantIncome"],
            "CoapplicantIncome": raw["CoapplicantIncome"],
            "LoanAmount": raw["LoanAmount"],
            "Loan_Amount_Term": raw["Loan_Amount_Term"],
            "Credit_History": raw["Credit_History"]
        }

        # Risk factor analysis
        risk_factors = []
        if raw["Credit_History"] == 0.0:
            risk_factors.append({
                "factor": "No / Poor Credit History",
                "severity": "high",
                "desc": "Banks require a clean repayment record. Defaults or missed payments significantly reduce approval chances."
            })

        combined_inc = raw["ApplicantIncome"] + raw["CoapplicantIncome"]
        if combined_inc > 0:
            term_months    = raw["Loan_Amount_Term"]
            monthly_payment = (raw["LoanAmount"] * 1000) / term_months if term_months > 0 else 0.0
            dti             = (monthly_payment / combined_inc) * 100

            if dti > 40:
                risk_factors.append({
                    "factor": f"High Debt-to-Income Ratio ({round(dti, 1)}%)",
                    "severity": "high",
                    "desc": "Your estimated EMI exceeds 40% of monthly income. Banks prefer this below 40%."
                })
            elif dti > 25:
                risk_factors.append({
                    "factor": f"Moderate Debt-to-Income Ratio ({round(dti, 1)}%)",
                    "severity": "medium",
                    "desc": "EMI payments consume a substantial portion of income. Consider a larger repayment term."
                })
            else:
                risk_factors.append({
                    "factor": f"Healthy Debt-to-Income Ratio ({round(dti, 1)}%)",
                    "severity": "low",
                    "desc": "Your income comfortably supports the estimated EMI. This is a positive indicator."
                })
        else:
            risk_factors.append({
                "factor": "No Income Declared",
                "severity": "high",
                "desc": "No active income was specified. Banks require proof of repayment capacity."
            })

        if combined_inc > 0 and combined_inc < 15000:
            risk_factors.append({
                "factor": "Low Household Income",
                "severity": "medium",
                "desc": "Combined income below ₹15,000/month may make it difficult to meet EMI obligations consistently."
            })

        if approved and raw["Credit_History"] == 1.0:
            risk_factors.append({
                "factor": "Good Credit Standing",
                "severity": "low",
                "desc": "Your credit history meets standard underwriting guidelines. This is a strong positive factor."
            })

        return render_template(
            "submit.html",
            approved=approved,
            confidence=confidence,
            applicant=applicant_display,
            model_name=BEST_MODEL_NAME,
            risk_factors=risk_factors
        )

    except Exception as e:
        logger.exception(f"Prediction form process failed: {e}")
        return f"Internal error during prediction: {e}", 500


# ============================================================
# 6. Admin Dashboard
# ============================================================
@app.route("/admin")
def admin_dashboard():
    """Admin analytics dashboard."""
    if "username" not in session:
        return redirect(url_for("login"))

    collection = db_instance.get_predictions_collection()

    total_apps    = 0
    approved_count = 0
    denied_count   = 0
    approval_rate  = 0.0
    avg_income     = 0.0
    avg_loan       = 0.0
    recent_logs    = []

    approval_by_credit   = {"credit_1_approved": 0, "credit_1_denied": 0, "credit_0_approved": 0, "credit_0_denied": 0}
    approval_by_property = {"Rural": {"Y": 0, "N": 0}, "Semiurban": {"Y": 0, "N": 0}, "Urban": {"Y": 0, "N": 0}}

    if collection is not None:
        try:
            records    = list(collection.find().sort("timestamp", -1))
            total_apps = len(records)
            recent_logs = records[:15]

            if total_apps > 0:
                approved_count = sum(1 for r in records if r.get("approved") is True)
                denied_count   = total_apps - approved_count
                approval_rate  = round((approved_count / total_apps) * 100, 1)

                incomes    = [r.get("applicant_income", 0) + r.get("coapplicant_income", 0) for r in records]
                avg_income = round(np.mean(incomes), 0) if incomes else 0.0

                loans    = [r.get("loan_amount", 0) for r in records]
                avg_loan = round(np.mean(loans), 1) if loans else 0.0

                for r in records:
                    ch  = r.get("credit_history", 1.0)
                    app = "approved" if r.get("approved") is True else "denied"
                    if ch == 1.0:
                        approval_by_credit[f"credit_1_{app}"] += 1
                    else:
                        approval_by_credit[f"credit_0_{app}"] += 1

                    prop       = r.get("property_area", "Urban")
                    status_key = "Y" if r.get("approved") is True else "N"
                    if prop in approval_by_property:
                        approval_by_property[prop][status_key] += 1
        except Exception as e:
            logger.error(f"Error fetching admin stats: {e}")

    train_results = metadata.get("results", {})

    return render_template(
        "admin_dashboard.html",
        model_name=BEST_MODEL_NAME,
        train_results=train_results,
        total_apps=total_apps,
        approved_count=approved_count,
        denied_count=denied_count,
        approval_rate=approval_rate,
        avg_income=avg_income,
        avg_loan=avg_loan,
        recent_logs=recent_logs,
        approval_by_credit=approval_by_credit,
        approval_by_property=approval_by_property,
        db_connected=(collection is not None)
    )

# ============================================================
# 7. REST API Endpoints
# ============================================================
@app.route("/api/predict", methods=["POST"])
def api_predict():
    """REST API endpoint for real-time model prediction."""
    if pipeline is None:
        return jsonify({"error": "Model pipeline is currently unavailable."}), 503

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Missing JSON request body"}), 400

        required_fields = [
            "gender", "married", "dependents", "education", "self_employed",
            "property_area", "applicant_income", "coapplicant_income",
            "loan_amount", "loan_amount_term", "credit_history"
        ]
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({"error": f"Missing required parameter(s): {', '.join(missing)}"}), 400

        X_input, raw = prepare_input_dataframe(data)
        pred     = int(pipeline.predict(X_input)[0])
        approved = (pred == 1)

        confidence = 0.0
        if hasattr(pipeline, "predict_proba"):
            probs      = pipeline.predict_proba(X_input)[0]
            confidence = round(float(probs[pred]) * 100, 2)

        collection = db_instance.get_predictions_collection()
        record_id  = None
        username   = data.get("username", "api_client")

        record = {
            "username": username,
            "gender": raw["Gender"],
            "married": raw["Married"],
            "dependents": raw["Dependents"],
            "education": raw["Education"],
            "self_employed": raw["Self_Employed"],
            "property_area": raw["Property_Area"],
            "applicant_income": raw["ApplicantIncome"],
            "coapplicant_income": raw["CoapplicantIncome"],
            "loan_amount": raw["LoanAmount"],
            "loan_amount_term": raw["Loan_Amount_Term"],
            "credit_history": raw["Credit_History"],
            "approved": approved,
            "confidence": confidence,
            "model_used": BEST_MODEL_NAME,
            "source": "api",
            "timestamp": datetime.utcnow()
        }

        if collection is not None:
            res       = collection.insert_one(record)
            record_id = str(res.inserted_id)

        response_payload = {
            "status": "success",
            "applicant_id": record_id,
            "prediction": {
                "eligible": approved,
                "confidence_percentage": confidence,
                "model_name": BEST_MODEL_NAME
            },
            "timestamp": record["timestamp"].isoformat()
        }

        logger.info(f"API prediction for '{username}'. ID: {record_id}")
        return jsonify(response_payload), 200

    except Exception as e:
        logger.exception(f"API prediction request failed: {e}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route("/api/history", methods=["GET"])
def api_history():
    """Returns global prediction logs from MongoDB."""
    collection = db_instance.get_predictions_collection()
    if collection is None:
        return jsonify({"error": "Database connection is not available."}), 503

    try:
        limit   = int(request.args.get("limit", 50))
        records = list(collection.find().sort("timestamp", -1).limit(limit))
        records_json = json.loads(json.dumps(records, cls=MongoJSONEncoder))
        return jsonify({"count": len(records_json), "logs": records_json}), 200
    except Exception as e:
        logger.error(f"API history fetching failed: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    logger.info(f"Starting Smart Lender server on port {config.PORT}...")
    app.run(debug=config.DEBUG, port=config.PORT)
