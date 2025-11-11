from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
# from pymongo.mongo_client import MongoClient
import mysql.connector
from mysql.connector import Error
from common_tools import CommonTools


class MongoTools:
    def __init__(self, username, password, cluster_name, app_name):
        uri = f"mongodb+srv://{username}:{password}@{cluster_name}.mongodb.net/?appName={app_name}"
        self.client = MongoClient(uri, server_api=ServerApi('1'))
        try:
            self.client.admin.command('ping')
            print("Pinged your deployment. You successfully connected to MongoDB!")
        except Exception as e:
            print(e)

    def get_database(self, db_name):
        return self.client[db_name]



class MySQLTools:
    DDL=f"""
    -- ---------- SCHEMA DEFINITION FOR MARS_DB ----------
    CREATE DATABASE IF NOT EXISTS mars_db
    DEFAULT CHARACTER SET utf8mb4
    COLLATE utf8mb4_0900_ai_ci;

    USE mars_db;

    -- ---------- LOOKUPS (single-valued) ----------
    CREATE TABLE IF NOT EXISTS type (
    typeID SMALLINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    type   VARCHAR(32) NOT NULL UNIQUE
    ) ENGINE=InnoDB;

    CREATE TABLE IF NOT EXISTS source (
    sourceID SMALLINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    source   VARCHAR(64) NOT NULL UNIQUE
    ) ENGINE=InnoDB;

    CREATE TABLE IF NOT EXISTS age_rating (
    age_ratingID TINYINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    age_rating   VARCHAR(16) NOT NULL UNIQUE,  -- G, PG-13, R-17+, R+, RX
    descr        VARCHAR(128) NULL
    ) ENGINE=InnoDB;

    -- ---------- LOOKUPS (multi-valued) ----------
    CREATE TABLE IF NOT EXISTS genre (
    genreID SMALLINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    genre   VARCHAR(64) NOT NULL UNIQUE
    ) ENGINE=InnoDB;

    CREATE TABLE IF NOT EXISTS studio (
    studioID INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    studio   VARCHAR(128) NOT NULL UNIQUE
    ) ENGINE=InnoDB;

    CREATE TABLE IF NOT EXISTS producer (
    producerID INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    producer   VARCHAR(128) NOT NULL UNIQUE
    ) ENGINE=InnoDB;

    CREATE TABLE IF NOT EXISTS licensor (
    licensorID INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    licensor   VARCHAR(128) NOT NULL UNIQUE
    ) ENGINE=InnoDB;

    -- ---------- CORE ----------
    CREATE TABLE IF NOT EXISTS users (
    userID INT UNSIGNED PRIMARY KEY,
    sex         VARCHAR(16)  NULL,
    age         SMALLINT     NULL,
    geoLocation VARCHAR(128) NULL
    ) ENGINE=InnoDB;

    CREATE TABLE IF NOT EXISTS anime (
    MAL_ID       INT UNSIGNED PRIMARY KEY,
    name         VARCHAR(255) NOT NULL,
    episodes     INT NULL,
    aired        VARCHAR(128) NULL,   #transform in date later
    premiered    VARCHAR(32)  NULL,
    duration     VARCHAR(64)  NULL,
    sourceID     SMALLINT UNSIGNED NULL,
    typeID       SMALLINT UNSIGNED NULL,
    age_ratingID TINYINT  UNSIGNED NULL,
    CONSTRAINT fk_anime_source    FOREIGN KEY (sourceID)     REFERENCES source(sourceID),
    CONSTRAINT fk_anime_type      FOREIGN KEY (typeID)       REFERENCES type(typeID),
    CONSTRAINT fk_anime_age       FOREIGN KEY (age_ratingID) REFERENCES age_rating(age_ratingID)
    ) ENGINE=InnoDB;

    -- ---------- BRIDGES (M:N) ----------
    CREATE TABLE IF NOT EXISTS anime_genre (
    MAL_ID  INT UNSIGNED NOT NULL,
    genreID SMALLINT UNSIGNED NOT NULL,
    PRIMARY KEY (MAL_ID, genreID),
    CONSTRAINT fk_ag_anime  FOREIGN KEY (MAL_ID)  REFERENCES anime(MAL_ID),
    CONSTRAINT fk_ag_genre  FOREIGN KEY (genreID) REFERENCES genre(genreID)
    ) ENGINE=InnoDB;

    CREATE TABLE IF NOT EXISTS anime_studio (
    MAL_ID   INT UNSIGNED NOT NULL,
    studioID INT UNSIGNED NOT NULL,
    PRIMARY KEY (MAL_ID, studioID),
    CONSTRAINT fk_as_anime  FOREIGN KEY (MAL_ID)   REFERENCES anime(MAL_ID),
    CONSTRAINT fk_as_studio FOREIGN KEY (studioID) REFERENCES studio(studioID)
    ) ENGINE=InnoDB;

    CREATE TABLE IF NOT EXISTS anime_producer (
    MAL_ID     INT UNSIGNED NOT NULL,
    producerID INT UNSIGNED NOT NULL,
    PRIMARY KEY (MAL_ID, producerID),
    CONSTRAINT fk_ap_anime    FOREIGN KEY (MAL_ID)     REFERENCES anime(MAL_ID),
    CONSTRAINT fk_ap_producer FOREIGN KEY (producerID) REFERENCES producer(producerID)
    ) ENGINE=InnoDB;

    CREATE TABLE IF NOT EXISTS anime_licensor (
    MAL_ID     INT UNSIGNED NOT NULL,
    licensorID INT UNSIGNED NOT NULL,
    PRIMARY KEY (MAL_ID, licensorID),
    CONSTRAINT fk_al_anime     FOREIGN KEY (MAL_ID)     REFERENCES anime(MAL_ID),
    CONSTRAINT fk_al_licensor  FOREIGN KEY (licensorID) REFERENCES licensor(licensorID)
    ) ENGINE=InnoDB;

    -- ---------- FACTS ----------
    CREATE TABLE IF NOT EXISTS watching_status (
    watching_statusID TINYINT UNSIGNED PRIMARY KEY,
    watching_status   VARCHAR(32) NOT NULL UNIQUE
    ) ENGINE=InnoDB;

    CREATE TABLE IF NOT EXISTS anime_user_rating (
    MAL_ID  INT UNSIGNED NOT NULL,
    userID  INT UNSIGNED NOT NULL,
    user_rating TINYINT NULL CHECK (user_rating BETWEEN 1 AND 10),
    watching_statusID TINYINT UNSIGNED NULL,
    watched_episodes  INT NULL,
    rated_at DATETIME NULL,
    PRIMARY KEY (userID, MAL_ID),
    KEY idx_rating_anime (MAL_ID, userID),
    KEY idx_rating_user  (userID),
    CONSTRAINT fk_aur_anime   FOREIGN KEY (MAL_ID)  REFERENCES anime(MAL_ID),
    CONSTRAINT fk_aur_user    FOREIGN KEY (userID)  REFERENCES users(userID),
    CONSTRAINT fk_aur_status  FOREIGN KEY (watching_statusID) REFERENCES watching_status(watching_statusID)
    ) ENGINE=InnoDB;

    -- ---------- AGGREGATES ----------
    CREATE TABLE IF NOT EXISTS anime_statistics (
    MAL_ID        INT UNSIGNED PRIMARY KEY,
    score         DECIMAL(4,2) NULL,
    `rank`        INT NULL,
    popularity    INT NULL,
    members       INT NULL,
    favorites     INT NULL,
    watching      INT NULL,
    completed     INT NULL,
    on_hold       INT NULL,
    dropped       INT NULL,
    plan_to_watch INT NULL,
    CONSTRAINT fk_ast_anime FOREIGN KEY (MAL_ID) REFERENCES anime(MAL_ID)
    ) ENGINE=InnoDB;
    """

    def __init__(self,
                 host: str = "mars-db.cdcmiuuwqo5z.eu-north-1.rds.amazonaws.com",
                 user: str = "msamosudova",
                 password: str = "Duckling25!",
                 database: str = "mars_db",
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
