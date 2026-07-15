# 🏦 SmartLender – AI-Powered Loan Eligibility Platform



SmartLender is a production-grade fintech web application that uses machine learning to instantly predict loan eligibility. Built with Flask, MongoDB, and a scikit-learn ML pipeline.



---



## 🚀 Features



- **Instant Loan Eligibility Check** — AI-powered decision in seconds

- **4-Step User-Friendly Application Form** — Step-by-step wizard with visual toggles

- **Loan Information Hub** — CIBIL score guide, top bank rates, loan types, approval tips

- **Personal Dashboard** — Track all your applications with history

- **Sign Up / Login** — Simple session-based authentication

- **MongoDB Integration** — All predictions logged to MongoDB

- **Risk Report** — Detailed breakdown of approval/denial factors

- **REST API** — `/api/predict` endpoint for programmatic access

- **Admin Dashboard** — Analytics on all submissions



---



## 📂 Project Structure



```

smart-lender/

├── app.py                 # Main Flask application & all routes

├── config.py              # Environment variable configuration

├── database.py            # MongoDB connection helper

├── train_model.py         # ML pipeline training script

├── eda.py                 # Exploratory Data Analysis script

├── requirements.txt       # Python dependencies

├── .env.example           # Example environment variables

├── data/

│   └── generate_dataset.py

├── model/

│   ├── pipeline.pkl       # Trained ML pipeline (generated)

│   └── metadata.json      # Model metrics & feature list

├── static/

│   └── style.css          # Premium fintech CSS theme

└── templates/

    ├── home.html          # Public landing page

    ├── login.html         # Login page

    ├── signup.html        # Sign-up page

    ├── loan_info.html     # Loan guide & information

    ├── customer_dashboard.html  # User dashboard

    ├── predict.html       # Loan application form (4-step wizard)

    ├── submit.html        # Eligibility result page

    └── admin_dashboard.html     # Admin analytics

```



---



## ⚙️ Setup & Installation



### 1. Clone the repository

```bash

git clone https://github.com/Charishma014/Smart-Lender.git

cd Smart-Lender

```



### 2. Create a virtual environment

```bash

python -m venv venv

# Windows

venv\Scripts\activate

# macOS/Linux

source venv/bin/activate

```



### 3. Install dependencies

```bash

pip install -r requirements.txt

```



### 4. Configure environment variables

```bash

cp .env.example .env

# Edit .env with your MongoDB URI and secret key

```



### 5. Start MongoDB

Make sure MongoDB is running locally on port `27017`, or update `MONGODB_URI` in `.env`.



### 6. Generate dataset & train the model

```bash

python data/generate_dataset.py

python train_model.py

```



### 7. Run the application

```bash

python app.py

```



Visit: **http://127.0.0.1:5000**



---



## 🗄️ Database



- **Database**: MongoDB (local or Atlas)

- **Database Name**: `smart_lender` (configurable via `.env`)

- **Collection**: `predictions` — stores all loan application records



MongoDB creates the database and collection **automatically** on first use — no manual setup required.



---



## 🔌 REST API



### POST `/api/predict`

```json

{

  "gender": "Male",

  "married": "Yes",

  "dependents": "0",

  "education": "Graduate",

  "self_employed": "No",

  "property_area": "Urban",

  "applicant_income": 5000,

  "coapplicant_income": 1500,

  "loan_amount": 150,

  "loan_amount_term": 360,

  "credit_history": 1

}

```



**Response:**

```json

{

  "status": "success",

  "applicant_id": "...",

  "prediction": {

    "eligible": true,

    "confidence_percentage": 87.4,

    "model_name": "XGBoostClassifier"

  }

}

```



### GET `/api/history?limit=50`

Returns the last N prediction records from MongoDB.



---



## 🛠️ Tech Stack



| Layer | Technology |

|---|---|

| Backend | Python 3.11+, Flask 3.x |

| ML Pipeline | scikit-learn, XGBoost, pandas, numpy |

| Database | MongoDB (pymongo) |

| Frontend | HTML5, Vanilla CSS, JavaScript |

| Fonts | Google Fonts (Inter) |



---



## 📝 Environment Variables (`.env`)



| Variable | Default | Description |

|---|---|---|

| `MONGODB_URI` | `mongodb://localhost:27017/` | MongoDB connection string |

| `DATABASE_NAME` | `smart_lender` | MongoDB database name |

| `SECRET_KEY` | (auto-generated) | Flask session secret key |

| `PORT` | `5000` | Application port |

| `FLASK_DEBUG` | `True` | Debug mode |



---



## 📄 License



MIT License — free to use and modify.
