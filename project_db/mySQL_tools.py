import mysql.connector
from mysql.connector import Error
from common_tools import CommonTools
from pathlib import Path
from dotenv import load_dotenv
import os
from dataclasses import dataclass

load_dotenv()

@dataclass
class MySQLConfig:
    host: str
    port: int
    user: str
    password: str
    database: str

    @classmethod
    def from_env(cls, prefix: str):
        return cls(
            host=os.getenv(f"{prefix}_MYSQL_HOST"),
            port=os.getenv(f"{prefix}_MYSQL_PORT"),
            user=os.getenv(f"{prefix}_MYSQL_USER"),
            password=os.getenv(f"{prefix}_MYSQL_PASSWORD"),
            database=os.getenv(f"{prefix}_MYSQL_DB"),
        )


@dataclass
class ConnectionConfigType:
    real_connection_config = MySQLConfig.from_env("REAL")
    test_connection_config = MySQLConfig.from_env("TEST")
    local_connection_config = MySQLConfig.from_env("LOCAL")


class Connection:
    """Создаёт соединения — удобно тестировать и подменять."""
    @staticmethod
    def create(config: MySQLConfig):
        return mysql.connector.connect(
            host=config.host,
            port=config.port,
            user=config.user,
            password=config.password,
            database=config.database,
            connection_timeout=10
        )


class MySQLTools:

    load_dotenv()
    DDL = Path("db_tools/schema.sql").read_text(encoding="utf-8")
    HOST = os.getenv("MYSQL_HOST")
    PORT = os.getenv("MYSQL_PORT")
    DB   = os.getenv("MYSQL_DB")
    LOCAL_USER = os.getenv("MYSQL_USER")
    LOCAL_PASSWORD = os.getenv("MYSQL_PASSWORD")

    def __init__(self,
                 host: str = HOST,
                 user: str = LOCAL_USER,
                 password: str = LOCAL_PASSWORD,
                 database: str = DB,
                 port: int = PORT):

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
        cursor = self.get_cursor()
        cursor.execute("""
        SELECT 1
        FROM information_schema.statistics
        WHERE table_schema = %s
          AND table_name = %s
          AND index_name = %s
        LIMIT 1
        """, (self.DB, table, index_name))
        if cursor.fetchone() is None:
            cursor.execute(f"ALTER TABLE {table} ADD INDEX {index_name} ({columns_csv})")
            print(f"Created index {index_name} on {table}({columns_csv})")
        else:
            print(f"Index {index_name} already exists on {table}")
        connection = self.__init__()
        connection.ping(reconnect=True, attempts=3, delay=2)
        connection.commit()

        cursor.close()
        connection.close()
 
    def ensure_index(self):
        cursor = self.get_cursor()
        connection = self.__init__()
        connection.ping(reconnect=True, attempts=3, delay=2)
 
        try:
            self.create_index(cursor, "anime_genre",    "idx_ag_by_genre",    "genreID, MAL_ID")
            self.create_index(cursor, "anime_studio",   "idx_as_by_studio",   "studioID, MAL_ID")
            self.create_index(cursor, "anime_producer", "idx_ap_by_producer", "producerID, MAL_ID")
            self.create_index(cursor, "anime_licensor", "idx_al_by_licensor", "licensorID, MAL_ID")
            self.create_index(cursor, "anime",          "idx_anime_name",     "name")
            connection.commit()
        finally:
            cursor.close()
            connection.close()

       