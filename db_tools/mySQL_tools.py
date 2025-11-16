import mysql.connector
from mysql.connector import Error
from common_tools import CommonTools
from pathlib import Path

class MySQLTools:

    DDL = Path("db_tools/schema.sql").read_text(encoding="utf-8")
    HOST = "mars-db.cdcmiuuwqo5z.eu-north-1.rds.amazonaws.com"
    DB   = "mars_db"

    def __init__(self,
                 host: str = HOST,
                 user: str = "msamosudova",
                 password: str = "Duckling25!",
                 database: str = DB,
                 port: int = 3306):

        try:
            self.connection = mysql.connector.connect(
                host=host,
                user=user,
                password=password,
                database=database,
                port=port,
                connection_timeout=10
            )
            if self.connection.is_connected():
                print("Connected to MySQL instance")
        except Error as e:
            print("MySQL Error:", e)

    def get_cursor(self):
        return self.connection.cursor()

    def close(self):
        if self.connection.is_connected():
            self.connection.close()
            print("MySQL connection is closed")

    def show_databases(self):
        cursor = self.get_cursor()
        cursor.execute("SHOW DATABASES;")
        for row in cursor.fetchall():
            print(row[0])
        cursor.close()

    def create_schema(self, ddl: str = DDL):
        cursor = self.get_cursor()
        try:
            for stmt in [s.strip() for s in ddl.split(";\n") if s.strip()]:
                cursor.execute(stmt)
            self.connection.commit()
            print("Schema 'mars_db' created/verified.")
        except Error as e:
            print("MySQL Error:", e)
        finally:
            cursor.close()

    def create_index(self, table, index_name, columns_csv):
        """Creates index if it does not exist. columns_csv: "col1, col2"""
        connection   = self.connection
        cursor = connection.cursor()

        # Check existence
        cursor.execute("""
            SELECT 1
            FROM information_schema.statistics
            WHERE table_schema = DATABASE()
            AND table_name = %s
            AND index_name = %s
            LIMIT 1
        """, (table, index_name))

        exists = cursor.fetchone()

        if exists is None:
            sql = f"ALTER TABLE {table} ADD INDEX {index_name} ({columns_csv})"
            cursor.execute(sql)
            connection.commit()
            print(f"Created index {index_name} on {table}({columns_csv})")
        else:
            print(f"✔ Index {index_name} already exists on {table}")

        cursor.close()
 
    def ensure_indexes(self):
        """
        Creates all required indexes for fast recommendation queries.
        """
        print("\n=== Ensuring required MySQL indexes ===")

        # Bridge tables (already in your old code)
        self.create_index("anime_genre",    "idx_ag_by_genre",    "genreID, MAL_ID")
        self.create_index("anime_genre",    "idx_ag_by_anime",    "MAL_ID")       # NEW!

        # NEW & IMPORTANT
        self.create_index("anime_user_rating", "idx_aur_user", "userID")
        self.create_index("anime_user_rating", "idx_aur_anime", "MAL_ID")

        # Users
        self.create_index("users", "idx_users_country_age", "country, age")

        # Optional: anime name search
        self.create_index("anime", "idx_anime_name", "name")

        print("=== Indexes ensured ===\n")

       
