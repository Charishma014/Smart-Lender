import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# MongoDB configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
DATABASE_NAME = os.getenv("DATABASE_NAME", "smart_lender")

# Model configuration
MODEL_DIR = os.path.join(BASE_DIR, "model")
PIPELINE_PATH = os.path.join(MODEL_DIR, "pipeline.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.json")

# Application configuration
DEBUG = os.getenv("FLASK_DEBUG", "True").lower() in ("true", "1", "yes")
PORT = int(os.getenv("PORT", 5000))
LOG_FILE = os.getenv("LOG_FILE", "app.log")
SECRET_KEY = os.getenv("SECRET_KEY", "smart-lender-default-dev-secret-key-12984")
