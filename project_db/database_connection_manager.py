import mysql.connector
from mysql.connector import Error
from pymongo import MongoClient


class DatabaseConnectionManager:
    """
    Centralized DB connection manager for MySQL + MongoDB.
    Ensures consistent connections, reconnection handling and cleaner architecture.
    """

    def __init__(
        self,
        mysql_host="mars-db.cdcmiuuwqo5z.eu-north-1.rds.amazonaws.com",
        mysql_user="msamosudova",
        mysql_password="Duckling25!",
        mysql_database="mars_db",
        mongo_uri="mongodb+srv://msamosudova:Duckling@mars-cluster.8ruotdw.mongodb.net/?appName=MARS-Cluster",
        mongo_db_name="anime_db"
    ):
        self.mysql_config = {
            "host": mysql_host,
            "user": mysql_user,
            "password": mysql_password,
            "database": mysql_database,
            "autocommit": True
        }

        self.mongo_uri = mongo_uri
        self.mongo_db_name = mongo_db_name

        self.mysql_conn = None
        self.mongo_client = None
        self.mongo_db = None

        # Open initial connections
        self.connect_mysql()
        self.connect_mongo()

    # -----------------------------
    # MySQL
    # -----------------------------
    def connect_mysql(self):
            try:
                self.mysql_conn = mysql.connector.connect(**self.mysql_config)
                print("Connected to MySQL")
                return self.mysql_conn
            except Error as e:
                print("MySQL connection error:", e)
                self.mysql_conn = None
                return None

    def get_mysql_connection(self):
        """Returns a live connection or reconnects."""
        try:
            # If no connection object
            if self.mysql_conn is None:
                print("Reconnecting to MySQL (conn=None)…")
                return self.connect_mysql()

            # If connection object exists but is dead
            if not self.mysql_conn.is_connected():
                print("Reconnecting to MySQL (lost connection)…")
                return self.connect_mysql()

            # Otherwise return working connection
            return self.mysql_conn

        except Exception as e:
            print(f"MySQL get-connection error: {e}")
            return self.connect_mysql()
        
    def get_mysql_cursor(self):
        conn = self.get_mysql_connection()
        return conn.cursor()
    
    def new_mysql_connection(self):
        """Always returns a NEW independent MySQL connection."""
        try:
            return mysql.connector.connect(**self.mysql_config)
        except Error as e:
            print("MySQL new connection error:", e)
            return None

    # -----------------------------
    # MongoDB
    # -----------------------------
    def connect_mongo(self):
        try:
            self.mongo_client = MongoClient(self.mongo_uri)
            self.mongo_db = self.mongo_client[self.mongo_db_name]
            print("Connected to MongoDB")
        except Exception as e:
            print("MongoDB connection error:", e)
            self.mongo_client = None

    def get_mongo_collection(self, name):
        """Returns a MongoDB collection object."""
        if self.mongo_client is None:
            self.connect_mongo()
        return self.mongo_db[name]

    # -----------------------------
    # Closing
    # -----------------------------
    def close_all(self):
        if self.mysql_conn:
            self.mysql_conn.close()
        if self.mongo_client:
            self.mongo_client.close()
        print("All DB connections closed.")
