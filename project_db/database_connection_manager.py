import mysql.connector
from mysql.connector import Error
from pymongo import MongoClient

class DatabaseConnectionManager:
    """
    Centralized DB connection manager for MySQL + MongoDB.
    Responsible for creating, maintaining, and closing ALL database connections.
    """

    def __init__(self, mysql_conf: dict, mongo_conf: dict):
        # 1. Store Configurations
        self.mysql_config = mysql_conf
        self.mysql_config["autocommit"] = True  # Ensure autocommit is on
        
        self.mongo_uri = mongo_conf["uri"]
        self.mongo_db_name = mongo_conf["db_name"]
        self.mongo_default_collection = mongo_conf.get("collection", "animes")

        # 2. Connection Placeholders
        self.mysql_conn = None
        self.mongo_client = None
        self.mongo_db = None

        # 3. Initialize Connections immediately
        self._connect_mysql()
        self._connect_mongo()

    # MySQL connection management
    def _connect_mysql(self):
        """Internal method to establish MySQL connection."""
        try:
            self.mysql_conn = mysql.connector.connect(**self.mysql_config)
            print("Connected to MySQL")
        except Error as e:
            print(f"MySQL Init Error: {e}")
            self.mysql_conn = None

    def get_mysql_connection(self):
        """Returns the active MySQL connection (reconnects if dropped)."""
        try:
            if self.mysql_conn is None:
                print("Reconnecting MySQL (None)...")
                self._connect_mysql()
            elif not self.mysql_conn.is_connected():
                print("Reconnecting MySQL (Lost)...")
                self._connect_mysql()
            return self.mysql_conn
        except Exception as e:
            print(f"Connection Error: {e}")
            self._connect_mysql()
            return self.mysql_conn
    
    def new_mysql_connection(self):
        """
        Creates a fresh, independent connection.
        Crucial for accurate Performance Benchmarking (bypassing session cache) and concurrency testing.
        """
        try:
            return mysql.connector.connect(**self.mysql_config)
        except Error as e:
            print(f"New Connection Error: {e}")
            raise e

    # MongoDB management
    def _connect_mongo(self):
        """Internal method to establish MongoDB connection."""
        try:
            self.mongo_client = MongoClient(self.mongo_uri)
            self.mongo_db = self.mongo_client[self.mongo_db_name]
            # Trigger a lazy connection check
            self.mongo_client.admin.command('ping')
            print("Connected to MongoDB")
        except Exception as e:
            print(f"MongoDB Init Error: {e}")
            self.mongo_client = None
            self.mongo_db = None

    def get_mongo_collection(self, collection_name=None):
        """
        Returns a MongoDB Collection object.
        Uses default collection if name is not provided.
        """
        if self.mongo_client is None:
            self._connect_mongo()
        
        target_col = collection_name if collection_name else self.mongo_default_collection
        return self.mongo_db[target_col]

    # Cleanup
    def close_all(self):
        """Closes all open connections."""
        if self.mysql_conn and self.mysql_conn.is_connected():
            self.mysql_conn.close()
            print("MySQL Closed")
        
        if self.mongo_client:
            self.mongo_client.close()
            print("MongoDB Closed")