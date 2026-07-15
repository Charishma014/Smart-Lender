import logging
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import config

logger = logging.getLogger(__name__)

class Database:
    def __init__(self):
        self.client = None
        self.db = None

    def connect(self):
        try:
            logger.info(f"Connecting to MongoDB at: {config.MONGODB_URI}")
            # The client is initialized with a short connection timeout so that it fails fast if not running
            self.client = MongoClient(config.MONGODB_URI, serverSelectionTimeoutMS=3000)
            # Trigger a connection attempt to check if server is available
            self.client.admin.command('ping')
            self.db = self.client[config.DATABASE_NAME]
            logger.info(f"Successfully connected to MongoDB database: {config.DATABASE_NAME}")
            return True
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            self.client = None
            self.db = None
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred while connecting to MongoDB: {e}")
            self.client = None
            self.db = None
            return False

    def get_db(self):
        if self.db is None:
            logger.warning("Database connection requested but database is not connected. Attempting reconnect...")
            self.connect()
        return self.db

    def get_predictions_collection(self):
        db = self.get_db()
        if db is not None:
            return db["predictions"]
        return None

    def get_users_collection(self):
        db = self.get_db()
        if db is not None:
            return db["users"]
        return None

# Singleton database instance
db_instance = Database()
