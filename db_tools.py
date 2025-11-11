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

    def create_schema(self, ddl):
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



# ==========================================

def ensure_index(cur, table, index_name, columns_csv):
    cur.execute("""
        SELECT 1
        FROM information_schema.statistics
        WHERE table_schema = %s
          AND table_name = %s
          AND index_name = %s
        LIMIT 1
    """, (DB, table, index_name))
    if cur.fetchone() is None:
        cur.execute(f"ALTER TABLE {table} ADD INDEX {index_name} ({columns_csv})")
        print(f"Created index {index_name} on {table}({columns_csv})")
    else:
        print(f"Index {index_name} already exists on {table}")

conn = mysql.connector.connect(host=HOST, user=USER, password=PWD, database=DB, connection_timeout=10)
conn.ping(reconnect=True, attempts=3, delay=2)
cur = conn.cursor()

try:
    ensure_index(cur, "anime_genre",    "idx_ag_by_genre",    "genreID, MAL_ID")
    ensure_index(cur, "anime_studio",   "idx_as_by_studio",   "studioID, MAL_ID")
    ensure_index(cur, "anime_producer", "idx_ap_by_producer", "producerID, MAL_ID")
    ensure_index(cur, "anime_licensor", "idx_al_by_licensor", "licensorID, MAL_ID")
    ensure_index(cur, "anime",          "idx_anime_name",     "name")
    conn.commit()
finally:
    cur.close()
    conn.close()

# ==========================================

# 01_load_anime_meta.py
# Loads anime.csv into: type, source, age_rating, genre, studio, producer, licensor,
# and fills anime, anime_statistics, plus bridges: anime_genre, anime_studio, anime_producer, anime_licensor.

import os, csv, re
import mysql.connector
from mysql.connector import Error

# ---------- CONFIG ----------
HOST = "mars-db.cdcmiuuwqo5z.eu-north-1.rds.amazonaws.com"
USER = "msamosudova"
PWD  = "Duckling25!"
DB   = "mars_db"
DATA_DIR = r"C:\MariaSamosudova\Projects\UNIVER\ADB\Project\DataSet"

ANIME_CSV   = os.path.join(DATA_DIR, "anime.csv")
#ANIME_SYN   = os.path.join(DATA_DIR, "anime_with_synopsis.csv")   #optional gor MongoDB

BATCH = 1000

# ---------- UTILS ----------
def split_list(cell):
    """split comma- or pipe-separated lists from dataset"""
    if cell is None:
        return []
    s = str(cell).strip()
    if not s or s.upper() in {"UNKNOWN","NONE","NULL","N/A"}:
        return []

    parts = re.split(r'[|;,]', s)
    return [p.strip() for p in parts if p.strip()]

def clean_str(s):
    if s is None:
        return None
    s = str(s).strip()
    return s if s else None

def to_int(s):
    try:
        return int(s)
    except:
        return None

def to_float2(s):
    try:
        return round(float(s), 2)
    except:
        return None

# ---------- DB ----------
def connect_db():
    return mysql.connector.connect(
        host=HOST, user=USER, password=PWD, database=DB, port=3306, connection_timeout=10
    )

def insert_ignore(cur, table, col, val):
    cur.execute(f"INSERT IGNORE INTO {table} ({col}) VALUES (%s)", (val,))

def get_id(cur, table, id_col, name_col, val, cache):
    """Upsert into lookup and return id (cached)."""
    if val is None:
        return None
    if val in cache:
        return cache[val]
    # try select
    cur.execute(f"SELECT {id_col} FROM {table} WHERE {name_col}=%s", (val,))
    row = cur.fetchone()
    if row:
        cache[val] = row[0]
        return row[0]
    # insert
    cur.execute(f"INSERT INTO {table} ({name_col}) VALUES (%s)", (val,))
    cache[val] = cur.lastrowid
    return cache[val]

# ---------- LOAD ----------
def load_anime():
    if not os.path.exists(ANIME_CSV):
        raise FileNotFoundError(ANIME_CSV)

    conn = connect_db()
    conn.autocommit = False
    cur = conn.cursor()

    type_cache = {}
    source_cache = {}
    age_cache = {}
    genre_cache = {}
    studio_cache = {}
    producer_cache = {}
    licensor_cache = {}

    # Counters
    n_anime = n_stats = n_gen = n_st = n_prod = n_lic = 0

    with open(ANIME_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # columns vary slightly across exports; be defensive
    def col(name, row):
        # try exact, then case-insensitive
        if name in row: return row[name]
        for k in row.keys():
            if k.lower() == name.lower():
                return row[k]
        return None

    anime_batch = []
    stats_batch = []
    gen_bridge, st_bridge, prod_bridge, lic_bridge = [], [], [], []

    for r in rows:
        mal_id  = to_int(col("MAL_ID", r))
        if not mal_id:  # skip bad rows
            continue

        name    = clean_str(col("Name", r))
        episodes= to_int(col("Episodes", r))
        aired   = clean_str(col("Aired", r))
        premiered = clean_str(col("Premiered", r))
        duration = clean_str(col("Duration", r))

        # lookups single-valued
        type_val   = clean_str(col("Type", r))
        type_id    = get_id(cur, "type", "typeID", "type", type_val, type_cache) if type_val else None

        source_val = clean_str(col("Source", r))
        source_id  = get_id(cur, "source", "sourceID", "source", source_val, source_cache) if source_val else None

        age_val    = clean_str(col("Rating", r)) or clean_str(col("age_rating", r))
        age_id     = get_id(cur, "age_rating", "age_ratingID", "age_rating", age_val, age_cache) if age_val else None

        # upsert anime
        anime_batch.append((
            mal_id, name, episodes, aired, premiered, duration, source_id, type_id, age_id
        ))

        # stats (denormalized snapshot in anime_statistics)
        stats_batch.append((
            mal_id,
            to_float2(col("Score", r)),
            to_int(col("Rank", r)),
            to_int(col("Popularity", r)),
            to_int(col("Members", r)),
            to_int(col("Favorites", r)),
            to_int(col("Watching", r)),
            to_int(col("Completed", r)),
            to_int(col("On-Hold", r)) or to_int(col("On_Hold", r)),
            to_int(col("Dropped", r)),
            to_int(col("Plan to Watch", r)) or to_int(col("Plan_to_Watch", r))
        ))

        # multi-valued lists -> lookups + bridges
        for g in split_list(col("Genres", r)):
            gid = get_id(cur, "genre", "genreID", "genre", g, genre_cache)
            gen_bridge.append((mal_id, gid))

        for st in split_list(col("Studios", r)):
            sid = get_id(cur, "studio", "studioID", "studio", st, studio_cache)
            st_bridge.append((mal_id, sid))

        for pr in split_list(col("Producers", r)):
            pid = get_id(cur, "producer", "producerID", "producer", pr, producer_cache)
            prod_bridge.append((mal_id, pid))

        for lc in split_list(col("Licensors", r)):
            lid = get_id(cur, "licensor", "licensorID", "licensor", lc, licensor_cache)
            lic_bridge.append((mal_id, lid))

        if len(anime_batch) >= BATCH:
            cur.executemany("""
                INSERT INTO anime (MAL_ID, name, episodes, aired, premiered, duration, sourceID, typeID, age_ratingID)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON DUPLICATE KEY UPDATE
                  name=VALUES(name), episodes=VALUES(episodes), aired=VALUES(aired),
                  premiered=VALUES(premiered), duration=VALUES(duration),
                  sourceID=VALUES(sourceID), typeID=VALUES(typeID), age_ratingID=VALUES(age_ratingID)
            """, anime_batch)
            n_anime += len(anime_batch); anime_batch.clear()

        if len(stats_batch) >= BATCH:
            cur.executemany("""
                INSERT INTO anime_statistics
                  (MAL_ID, score, `rank`, popularity, members, favorites, watching, completed, on_hold, dropped, plan_to_watch)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON DUPLICATE KEY UPDATE
                  score=VALUES(score), `rank`=VALUES(`rank`), popularity=VALUES(popularity),
                  members=VALUES(members), favorites=VALUES(favorites),
                  watching=VALUES(watching), completed=VALUES(completed),
                  on_hold=VALUES(on_hold), dropped=VALUES(dropped), plan_to_watch=VALUES(plan_to_watch)
            """, stats_batch)
            n_stats += len(stats_batch); stats_batch.clear()

        # Bridges batched
        if len(gen_bridge) >= BATCH:
            cur.executemany("INSERT IGNORE INTO anime_genre (MAL_ID, genreID) VALUES (%s,%s)", gen_bridge)
            n_gen += len(gen_bridge); gen_bridge.clear()

        if len(st_bridge) >= BATCH:
            cur.executemany("INSERT IGNORE INTO anime_studio (MAL_ID, studioID) VALUES (%s,%s)", st_bridge)
            n_st += len(st_bridge); st_bridge.clear()

        if len(prod_bridge) >= BATCH:
            cur.executemany("INSERT IGNORE INTO anime_producer (MAL_ID, producerID) VALUES (%s,%s)", prod_bridge)
            n_prod += len(prod_bridge); prod_bridge.clear()

        if len(lic_bridge) >= BATCH:
            cur.executemany("INSERT IGNORE INTO anime_licensor (MAL_ID, licensorID) VALUES (%s,%s)", lic_bridge)
            n_lic += len(lic_bridge); lic_bridge.clear()

    # flush remaining
    if anime_batch:
        cur.executemany("""
            INSERT INTO anime (MAL_ID, name, episodes, aired, premiered, duration, sourceID, typeID, age_ratingID)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON DUPLICATE KEY UPDATE
              name=VALUES(name), episodes=VALUES(episodes), aired=VALUES(aired),
              premiered=VALUES(premiered), duration=VALUES(duration),
              sourceID=VALUES(sourceID), typeID=VALUES(typeID), age_ratingID=VALUES(age_ratingID)
        """, anime_batch)
        n_anime += len(anime_batch)

    if stats_batch:
        cur.executemany("""
            INSERT INTO anime_statistics
              (MAL_ID, score, `rank`, popularity, members, favorites, watching, completed, on_hold, dropped, plan_to_watch)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON DUPLICATE KEY UPDATE
              score=VALUES(score), `rank`=VALUES(`rank`), popularity=VALUES(popularity),
              members=VALUES(members), favorites=VALUES(favorites),
              watching=VALUES(watching), completed=VALUES(completed),
              on_hold=VALUES(on_hold), dropped=VALUES(dropped), plan_to_watch=VALUES(plan_to_watch)
        """, stats_batch)
        n_stats += len(stats_batch)

    if gen_bridge:
        cur.executemany("INSERT IGNORE INTO anime_genre (MAL_ID, genreID) VALUES (%s,%s)", gen_bridge)
        n_gen += len(gen_bridge)
    if st_bridge:
        cur.executemany("INSERT IGNORE INTO anime_studio (MAL_ID, studioID) VALUES (%s,%s)", st_bridge)
        n_st += len(st_bridge)
    if prod_bridge:
        cur.executemany("INSERT IGNORE INTO anime_producer (MAL_ID, producerID) VALUES (%s,%s)", prod_bridge)
        n_prod += len(prod_bridge)
    if lic_bridge:
        cur.executemany("INSERT IGNORE INTO anime_licensor (MAL_ID, licensorID) VALUES (%s,%s)", lic_bridge)
        n_lic += len(lic_bridge)

    conn.commit()
    cur.close(); conn.close()
    print(f"anime upserted: {n_anime}, stats: {n_stats}, links — genres:{n_gen} studios:{n_st} producers:{n_prod} licensors:{n_lic}")

if __name__ == "__main__":
    load_anime()

# ==========================================

# 02_load_users_and_ratings.py
# - autocommit=True (without long transactions)
# - ensure users before facts
# - validate MAL_ID against anime
# - validate watching_statusID against lookup; unknown/0 -> NULL

import os
import csv
import mysql.connector
from mysql.connector import errorcode

# ---------- CONFIG ----------
HOST = "mars-db.cdcmiuuwqo5z.eu-north-1.rds.amazonaws.com"
USER = "msamosudova"
PWD  = "Duckling25!"
DB   = "mars_db"

DATA_DIR   = r"C:\MariaSamosudova\Projects\UNIVER\ADB\Project\DataSet"
USERS_CSV  = os.path.join(DATA_DIR, "users.csv")
ANIMELIST  = os.path.join(DATA_DIR, "animelist.csv")
RATING_CSV = os.path.join(DATA_DIR, "rating_complete.csv")
WATCH_CSV  = os.path.join(DATA_DIR, "watching_status.csv")

BATCH_USERS   = 20000
BATCH_RATINGS = 5000

# ---------- DB ----------
def connect_db():
    return mysql.connector.connect(
        host=HOST, user=USER, password=PWD, database=DB,
        port=3306, connection_timeout=10, autocommit=True
    )

# ---------- UTIL ----------
def to_int_safe(v):
    try:
        if v is None:
            return None
        s = str(v).strip()
        if s == "":
            return None
        return int(s)
    except:
        return None

# ---------- LOOKUPS FROM DB ----------
def load_known_anime_ids(cur):
    cur.execute("SELECT MAL_ID FROM anime")
    return {row[0] for row in cur.fetchall()}

def load_known_status_ids(cur):
    cur.execute("SELECT watching_statusID FROM watching_status")
    return {row[0] for row in cur.fetchall()}

# ---------- LOAD lookup files ----------
def upsert_watching_status(cur):
    """Load watching_status.csv if present."""
    if not os.path.exists(WATCH_CSV):
        return 0
    rows = []
    with open(WATCH_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            sid = to_int_safe(r.get("status_id"))
            status = (r.get("status") or "").strip()
            if sid is None or not status:
                continue
            rows.append((sid, status))
    if rows:
        cur.executemany(
            "INSERT INTO watching_status (watching_statusID, watching_status) VALUES (%s,%s) "
            "ON DUPLICATE KEY UPDATE watching_status=VALUES(watching_status)",
            rows
        )
    return len(rows)

def insert_users_from_users_csv(cur):
    if not os.path.exists(USERS_CSV):
        return 0
    buf, total = [], 0
    with open(USERS_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            uid = to_int_safe(r.get("user_id") or r.get("userID"))
            if uid is None:
                continue
            buf.append((uid,))
            if len(buf) >= BATCH_USERS:
                cur.executemany("INSERT IGNORE INTO users (userID) VALUES (%s)", buf)
                total += len(buf); buf.clear()
    if buf:
        cur.executemany("INSERT IGNORE INTO users (userID) VALUES (%s)", buf)
        total += len(buf)
    return total

def collect_user_ids_from_file(path):
    users = set()
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            uid = to_int_safe(r.get("user_id") or r.get("userID"))
            if uid is not None:
                users.add(uid)
    return users

def ensure_users_for_source(cur, source_path):
    if not os.path.exists(source_path):
        return 0
    users = sorted(collect_user_ids_from_file(source_path))
    total, buf = 0, []
    for uid in users:
        buf.append((uid,))
        if len(buf) >= BATCH_USERS:
            cur.executemany("INSERT IGNORE INTO users (userID) VALUES (%s)", buf)
            total += len(buf); buf.clear()
    if buf:
        cur.executemany("INSERT IGNORE INTO users (userID) VALUES (%s)", buf)
        total += len(buf)
    return total

# ---------- FACT LOADERS ----------
def load_ratings_from_animelist(cur, known_anime, known_status):
    """
    animelist.csv: user_id, anime_id, rating, watching_status, watched_episodes
    - rating 0 -> NULL
    - watching_status not in lookup (including 0) -> NULL
    """
    if not os.path.exists(ANIMELIST):
        return 0
    cnt, batch, unknown_ws = 0, [], 0

    with open(ANIMELIST, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            uid = to_int_safe(r.get("user_id") or r.get("userID"))
            aid = to_int_safe(r.get("anime_id") or r.get("MAL_ID"))
            if uid is None or aid is None or aid not in known_anime:
                continue

            rating = to_int_safe(r.get("rating") or r.get("score"))
            if rating is not None and rating == 0:
                rating = None

            ws = to_int_safe(r.get("watching_status"))
            if ws is None or ws not in known_status:
                ws = None
                unknown_ws += 1

            we = to_int_safe(r.get("watched_episodes"))

            batch.append((aid, uid, rating, ws, we))

            if len(batch) >= BATCH_RATINGS:
                batch.sort(key=lambda x: (x[1], x[0]))
                cur.executemany("""
                    INSERT INTO anime_user_rating (MAL_ID, userID, user_rating, watching_statusID, watched_episodes)
                    VALUES (%s,%s,%s,%s,%s)
                    ON DUPLICATE KEY UPDATE
                      user_rating=COALESCE(VALUES(user_rating), user_rating),
                      watching_statusID=COALESCE(VALUES(watching_statusID), watching_statusID),
                      watched_episodes=COALESCE(VALUES(watched_episodes), watched_episodes)
                """, batch)
                cnt += len(batch); batch.clear()

    if batch:
        batch.sort(key=lambda x: (x[1], x[0]))
        cur.executemany("""
            INSERT INTO anime_user_rating (MAL_ID, userID, user_rating, watching_statusID, watched_episodes)
            VALUES (%s,%s,%s,%s,%s)
            ON DUPLICATE KEY UPDATE
              user_rating=COALESCE(VALUES(user_rating), user_rating),
              watching_statusID=COALESCE(VALUES(watching_statusID), watching_statusID),
              watched_episodes=COALESCE(VALUES(watched_episodes), watched_episodes)
        """, batch)
        cnt += len(batch)

    if unknown_ws:
        print(f"Note: {unknown_ws} rows had unknown/zero watching_status -> stored as NULL.")
    return cnt

def load_ratings_from_rating_complete(cur, known_anime):
    """
    rating_complete.csv: user_id, anime_id, rating (1..10)
    """
    if not os.path.exists(RATING_CSV):
        return 0
    cnt, batch = 0, []

    with open(RATING_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            uid = to_int_safe(r.get("user_id") or r.get("userID"))
            aid = to_int_safe(r.get("anime_id") or r.get("MAL_ID"))
            rating = to_int_safe(r.get("rating") or r.get("score"))
            if uid is None or aid is None or rating is None or rating == 0:
                continue
            if aid not in known_anime:
                continue

            batch.append((aid, uid, rating))

            if len(batch) >= BATCH_RATINGS:
                batch.sort(key=lambda x: (x[1], x[0]))
                cur.executemany("""
                    INSERT INTO anime_user_rating (MAL_ID, userID, user_rating)
                    VALUES (%s,%s,%s)
                    ON DUPLICATE KEY UPDATE user_rating=VALUES(user_rating)
                """, batch)
                cnt += len(batch); batch.clear()

    if batch:
        batch.sort(key=lambda x: (x[1], x[0]))
        cur.executemany("""
            INSERT INTO anime_user_rating (MAL_ID, userID, user_rating)
            VALUES (%s,%s,%s)
            ON DUPLICATE KEY UPDATE user_rating=VALUES(user_rating)
        """, batch)
        cnt += len(batch)

    return cnt

# ---------- MAIN ----------
if __name__ == "__main__":
    conn = connect_db()
    cur = conn.cursor()

    # session setup
    cur.execute("SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED")
    cur.execute("SET SESSION innodb_lock_wait_timeout = 50")

    # lookup from db
    known_anime = load_known_anime_ids(cur)
    print("known anime:", len(known_anime))

    # watching_status.csv (if exists)
    ws_loaded = upsert_watching_status(cur)
    known_status = load_known_status_ids(cur)
    print("watching_status upserted:", ws_loaded, "known statuses:", sorted(known_status))

    # users from users.csv (if exist)
    u_from_users = insert_users_from_users_csv(cur)
    print("users from users.csv inserted/ignored:", u_from_users)

    # ensure users from retings sourse + load facts
    if os.path.exists(ANIMELIST):
        added = ensure_users_for_source(cur, ANIMELIST)
        print("ensured users from animelist.csv:", added)
        n = load_ratings_from_animelist(cur, known_anime, known_status)
        print("animelist rows upserted:", n)
    elif os.path.exists(RATING_CSV):
        added = ensure_users_for_source(cur, RATING_CSV)
        print("ensured users from rating_complete.csv:", added)
        n = load_ratings_from_rating_complete(cur, known_anime)
        print("rating_complete rows upserted:", n)
    else:
        print("No ratings file found (expected animelist.csv or rating_complete.csv).")

    cur.close()
    conn.close()
    print("Done.")

# ==========================================

# 03_load_mongo_anime.py
# Load Kaggle Anime dataset into MongoDB (collection: anime_db.animes)
# Sources:
#   - anime_with_synopsis.csv   (required; provides synopsis and often duplicated columns)
#   - anime.csv                 (optional; enrich with fields and stats if present)
#
# Document shape (example):
# { _id: 5114, name: "...", synopsis: "...",
#   genres: [...], studios: [...], producers: [...], licensors: [...],
#   type: "TV", episodes: 64, aired: "...", premiered: "...", duration: "...",
#   age_rating: "PG-13",
#   stats: { score: 9.21, rank: 1, popularity: 1, members: 2850000, favorites: 230000 }
# }

import os, csv, re, math
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError

# ---------------- CONFIG ----------------
DATA_DIR = r"C:\MariaSamosudova\Projects\UNIVER\ADB\Project\DataSet"
CSV_SYN  = os.path.join(DATA_DIR, "anime_with_synopsis.csv")  # obligatory
CSV_META = os.path.join(DATA_DIR, "anime.csv")                # optional

MONGO_URI = "mongodb+srv://msamosudova:Duckling@mars-cluster.8ruotdw.mongodb.net/?appName=MARS-Cluster"
DB_NAME   = "anime_db"
COLL_NAME = "animes"

BATCH = 5000
PROGRESS_EVERY = 50_000

# --------------- HELPERS ----------------
def clean_str(s):
    if s is None:
        return None
    s = str(s).strip()
    return s if s else None

def to_int(s):
    try:
        if s is None or str(s).strip() == "":
            return None
        return int(s)
    except:
        return None

def to_float2(s):
    try:
        if s is None or str(s).strip() == "":
            return None
        return round(float(s), 2)
    except:
        return None

def split_list(cell):
    """Split comma/pipe/semicolon separated values into a normalized string array."""
    if cell is None:
        return []
    s = str(cell).strip()
    if not s or s.upper() in {"UNKNOWN","NONE","NULL","N/A"}:
        return []
    parts = re.split(r'[|;,]', s)
    # normalize spaces, drop empty/dups while preserving order
    seen = set()
    out = []
    for p in parts:
        v = p.strip()
        if not v:
            continue
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out

def lower_keys(d):
    """Return a dict with case-insensitive access via lowercased keys."""
    return { (k.lower() if isinstance(k, str) else k): v for k,v in d.items() }

# --------------- LOAD META (optional enrich) ---------------
def load_meta_map():
    """Read anime.csv if present; return dict MAL_ID -> meta fields."""
    if not os.path.exists(CSV_META):
        print("Note: anime.csv not found, will load from anime_with_synopsis.csv only.")
        return {}

    meta = {}
    with open(CSV_META, encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            r = lower_keys(row)
            mid = to_int(r.get("mal_id"))
            if not mid:
                continue
            meta[mid] = {
                "name"      : clean_str(r.get("name")),
                "episodes"  : to_int(r.get("episodes")),
                "type"      : clean_str(r.get("type")),
                "aired"     : clean_str(r.get("aired")),
                "premiered" : clean_str(r.get("premiered")),
                "duration"  : clean_str(r.get("duration")),
                "age_rating": clean_str(r.get("rating")),  # age code in anime.csv
                "genres"    : split_list(r.get("genres")),
                "studios"   : split_list(r.get("studios")),
                "producers" : split_list(r.get("producers")),
                "licensors" : split_list(r.get("licensors")),
                "stats"     : {
                    "score"     : to_float2(r.get("score")),
                    "rank"      : to_int(r.get("rank")),
                    "popularity": to_int(r.get("popularity")),
                    "members"   : to_int(r.get("members")),
                    "favorites" : to_int(r.get("favorites")),
                }
            }
    print(f"Loaded meta for {len(meta):,} anime from anime.csv")
    return meta

# --------------- MAIN LOAD ----------------
def main():
    if not os.path.exists(CSV_SYN):
        raise FileNotFoundError(f"File not found: {CSV_SYN}")

    client = MongoClient(MONGO_URI)
    db  = client[DB_NAME]
    col = db[COLL_NAME]

    # indexes (create once; safe to re-run)
    # text index: name + synopsis
    try:
        col.create_index([("name", "text"), ("synopsis", "text")])
    except Exception as e:
        print("create text index:", e)
    # filter indexes
    col.create_index([("genres", 1), ("type", 1)])
    col.create_index([("studios", 1)])
    col.create_index([("producers", 1)])
    col.create_index([("licensors", 1)])
    col.create_index([("stats.score", -1)])

    meta_map = load_meta_map()

    ops = []
    total = 0
    upserts = 0

    with open(CSV_SYN, encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            r = lower_keys(row)

            mal_id = to_int(r.get("mal_id"))
            if not mal_id:
                continue

            # base fields: take from synopsis CSV, fallback to meta_map if missing
            name       = clean_str(r.get("name")) or (meta_map.get(mal_id, {}).get("name"))
            synopsis   = clean_str(r.get("synopsis"))
            episodes   = to_int(r.get("episodes")) if r.get("episodes") is not None else meta_map.get(mal_id, {}).get("episodes")
            type_      = clean_str(r.get("type")) or (meta_map.get(mal_id, {}).get("type"))
            aired      = clean_str(r.get("aired")) or (meta_map.get(mal_id, {}).get("aired"))
            premiered  = clean_str(r.get("premiered")) or (meta_map.get(mal_id, {}).get("premiered"))
            duration   = clean_str(r.get("duration")) or (meta_map.get(mal_id, {}).get("duration"))
            age_rating = clean_str(r.get("rating")) or (meta_map.get(mal_id, {}).get("age_rating"))

            # multi-value arrays (prefer synopsis csv; if empty use meta_map)
            genres    = split_list(r.get("genres"))    or meta_map.get(mal_id, {}).get("genres")    or []
            studios   = split_list(r.get("studios"))   or meta_map.get(mal_id, {}).get("studios")   or []
            producers = split_list(r.get("producers")) or meta_map.get(mal_id, {}).get("producers") or []
            licensors = split_list(r.get("licensors")) or meta_map.get(mal_id, {}).get("licensors") or []

            # stats
            stats = {
                "score"     : to_float2(r.get("score"))      if "score" in r else meta_map.get(mal_id, {}).get("stats", {}).get("score"),
                "rank"      : to_int(r.get("rank"))          if "rank" in r else meta_map.get(mal_id, {}).get("stats", {}).get("rank"),
                "popularity": to_int(r.get("popularity"))    if "popularity" in r else meta_map.get(mal_id, {}).get("stats", {}).get("popularity"),
                "members"   : to_int(r.get("members"))       if "members" in r else meta_map.get(mal_id, {}).get("stats", {}).get("members"),
                "favorites" : to_int(r.get("favorites"))     if "favorites" in r else meta_map.get(mal_id, {}).get("stats", {}).get("favorites"),
            }
            # drop empty stats keys
            stats = {k:v for k,v in stats.items() if v is not None}
            if not stats:
                stats = None

            doc = {
                "_id": mal_id,
                "name": name,
                "synopsis": synopsis,
                "type": type_,
                "episodes": episodes,
                "aired": aired,
                "premiered": premiered,
                "duration": duration,
                "age_rating": age_rating,
                "genres": genres,
                "studios": studios,
                "producers": producers,
                "licensors": licensors
            }
            if stats: doc["stats"] = stats

            # clean None fields to keep docs compact
            doc = {k:v for k,v in doc.items() if v is not None}

            ops.append(
                UpdateOne({"_id": mal_id}, {"$set": doc}, upsert=True)
            )

            total += 1
            if len(ops) >= BATCH:
                try:
                    res = col.bulk_write(ops, ordered=False)
                    upserts += (res.upserted_count or 0)
                except BulkWriteError as bwe:
                    print("Bulk error:", bwe.details.get("writeErrors", [])[:3])
                    raise
                ops.clear()

            if total % PROGRESS_EVERY == 0:
                print(f"processed={total:,} upserts≈{upserts:,}")

    if ops:
        res = col.bulk_write(ops, ordered=False)
        upserts += (res.upserted_count or 0)
        ops.clear()

    print(f"Done. processed={total:,}, upserts≈{upserts:,}")
    # simple sanity check
    print("docs in collection:", col.estimated_document_count())

if __name__ == "__main__":
    main()

# ==========================================

import os
import csv
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError

# --- CONFIG ---
DATA_DIR = r"C:\MariaSamosudova\Projects\UNIVER\ADB\Project\DataSet"
CSV_SYN  = os.path.join(DATA_DIR, "anime_with_synopsis.csv")

MONGO_URI = "mongodb+srv://msamosudova:Duckling@mars-cluster.8ruotdw.mongodb.net/?appName=MARS-Cluster"
COLL_NAME = "animes"

BATCH = 3000
OVERWRITE_EXISTING   = True  # True -> rewrite existing synopsis
UPSERT_MISSING_DOCS  = True  # True -> create document if it does not exist

# --- helpers ---
def to_int(val):
    try:
        if val is None or str(val).strip() == "":
            return None
        return int(val)
    except:
        return None

def clean_text(s):
    if s is None:
        return None
    t = str(s).strip()
    return t if t else None

def lower_dict(d):
    return { (k.lower() if isinstance(k,str) else k): v for k,v in d.items() }

def main():
    if not os.path.exists(CSV_SYN):
        raise FileNotFoundError(CSV_SYN)

    client = MongoClient(MONGO_URI)
    col = client[DB_NAME][COLL_NAME]

    # indexes, safe to rerun
    col.create_index([("name","text"), ("synopsis","text")])

    ops, total_rows, upserts = [], 0, 0
    updated, skipped = 0, 0

    with open(CSV_SYN, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            r = lower_dict(row)
            mal_id   = to_int(r.get("mal_id") or r.get("id"))
            synopsis = clean_text(r.get("synopsis") or r.get("synopsys") or r.get("synopsis_text") or r.get("sypnopsis"))

            if not mal_id:
                continue  # no key
            if not synopsis:
                skipped += 1
                continue  # nothing to write

            if OVERWRITE_EXISTING:
                filt = {"_id": mal_id}
            else:
                filt = {"_id": mal_id, "$or": [{"synopsis": {"$exists": False}}, {"synopsis": ""}]}

            ops.append(UpdateOne(filt, {"$set": {"synopsis": synopsis}}, upsert=UPSERT_MISSING_DOCS))
            total_rows += 1

            if len(ops) >= BATCH:
                try:
                    res = col.bulk_write(ops, ordered=False)
                    updated += (res.modified_count or 0)
                    upserts += (res.upserted_count or 0)
                except BulkWriteError as bwe:
                    print("Bulk error sample:", bwe.details.get("writeErrors", [])[:3])
                    raise
                finally:
                    ops.clear()

    # flush
    if ops:
        res = col.bulk_write(ops, ordered=False)
        updated += (res.modified_count or 0)
        upserts += (res.upserted_count or 0)
        ops.clear()

    print(f"Done. read_rows={total_rows:,}, updated={updated:,}, upserts={upserts:,}, skipped_empty={skipped:,}")

    # sanity-check
    have = col.count_documents({"synopsis": {"$exists": True, "$ne": ""}})
    total_docs = col.estimated_document_count()
    print(f"Docs with synopsis: {have:,} / {total_docs:,}")

if __name__ == "__main__":
    main()

# ==========================================

# 04_load_mongo_alt_names.py
# Build anime_db.anime_alternate_names from CSVs (anime_with_synopsis.csv / anime.csv)
# Fields:
#   _id = MAL_ID
#   english_name: str | None
#   japanese_name: str | None
#   synonyms: [str]

import os, csv, re
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError

# ---------- CONFIG ----------
DATA_DIR = r"C:\MariaSamosudova\Projects\UNIVER\ADB\Project\DataSet"
CSV_SYN  = os.path.join(DATA_DIR, "anime_with_synopsis.csv")
CSV_META = os.path.join(DATA_DIR, "anime.csv")  # optional

MONGO_URI = "mongodb+srv://msamosudova:Duckling@mars-cluster.8ruotdw.mongodb.net/?appName=MARS-Cluster"
DB_NAME   = "anime_db"
ALT_COLL  = "anime_alternate_names"
ANIME_COLL= "animes"   # optional copying

BATCH = 5000
PROGRESS_EVERY = 50_000

# ---------- helpers ----------
def lower_keys(d): return { (k.lower() if isinstance(k,str) else k): v for k,v in d.items() }

def get_first(d, keys):
    """returns first non-empty value by key (case-insensitive)."""
    for k in keys:
        v = d.get(k.lower())
        if v is not None:
            s = str(v).strip()
            if s and s.upper() not in {"NULL","NONE","N/A","UNKNOWN"}:
                return s
    return None

def split_synonyms(val):
    """takes in account different delimiters"""
    if val is None:
        return []
    parts = re.split(r'[|;,]', str(val))
    seen, out = set(), []
    for p in parts:
        t = p.strip()
        if not t:
            continue
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

def load_map_from_csv(path):
    """returns dict MAL_ID -> {english_name, japanese_name, synonyms[]} from file."""
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            r = lower_keys(row)
            # MAL_ID
            mal_id = get_first(r, ["mal_id", "id", "malid"])
            if not mal_id or not mal_id.isdigit():
                continue
            mal_id = int(mal_id)

            english = get_first(r, ["english name", "english_name", "english", "title_english"])
            japanese= get_first(r, ["japanese name", "japanese_name", "japanese", "title_japanese"])
            # synonyms could be named differently; try typical variants
            syn_raw = get_first(r, ["synonyms", "other name", "other names", "other_names", "title_synonyms"])
            synonyms = split_synonyms(syn_raw)

            out[mal_id] = {
                "english_name": english,
                "japanese_name": japanese,
                "synonyms": synonyms
            }
    return out

def merge_alt(a, b):
    """merge records of alternative titles with deduplication"""
    if not a: return b or {}
    if not b: return a or {}
    english = a.get("english_name") or b.get("english_name")
    japanese= a.get("japanese_name") or b.get("japanese_name")
    s1 = a.get("synonyms") or []
    s2 = b.get("synonyms") or []
    seen, syn = set(), []
    for x in s1 + s2:
        if not x:
            continue
        if x not in seen:
            seen.add(x)
            syn.append(x)
    return {"english_name": english, "japanese_name": japanese, "synonyms": syn}

# ---------- main ----------
def main():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    alt = db[ALT_COLL]
    animes = db[ANIME_COLL]

    # indexes (safe to re-run)
    alt.create_index([("english_name", 1)])
    alt.create_index([("japanese_name", 1)])
    alt.create_index([("synonyms", 1)])

    # load both maps (order: synopsis csv has priority, then anime.csv as fallback)
    map_syn  = load_map_from_csv(CSV_SYN)
    map_meta = load_map_from_csv(CSV_META)

    # union of keys
    all_ids = set(map_syn.keys()) | set(map_meta.keys())

    ops_alt, ops_embed = [], []
    total = upserts = 0

    for mal_id in all_ids:
        a = map_syn.get(mal_id)
        b = map_meta.get(mal_id)
        rec = merge_alt(a, b)

        # clean empty
        rec["synonyms"] = [s for s in (rec.get("synonyms") or []) if s]
        if not rec.get("english_name") and not rec.get("japanese_name") and not rec["synonyms"]:
            # if no info exists
            continue

        # upsert into dedicated alt collection
        doc = {"_id": mal_id}
        if rec.get("english_name"):  doc["english_name"]  = rec["english_name"]
        if rec.get("japanese_name"): doc["japanese_name"] = rec["japanese_name"]
        if rec["synonyms"]:          doc["synonyms"]      = rec["synonyms"]

        ops_alt.append(UpdateOne({"_id": mal_id}, {"$set": doc}, upsert=True))

        total += 1
        if len(ops_alt) >= BATCH:
            res = alt.bulk_write(ops_alt, ordered=False)
            upserts += (res.upserted_count or 0)
            ops_alt.clear()

        if len(ops_embed) >= BATCH:
            animes.bulk_write(ops_embed, ordered=False)
            ops_embed.clear()

        if total % PROGRESS_EVERY == 0:
            print(f"processed={total:,} upserts_alt≈{upserts:,}")

    if ops_alt:
        res = alt.bulk_write(ops_alt, ordered=False)
        upserts += (res.upserted_count or 0)
        ops_alt.clear()

    if ops_embed:
        animes.bulk_write(ops_embed, ordered=False)
        ops_embed.clear()

    print(f"Done. processed={total:,}, alt upserts≈{upserts:,}")
    print("docs in alt collection:", alt.estimated_document_count())

    # helpful text index for searching alt names
    try:
        alt.create_index([("english_name", "text"), ("japanese_name", "text"), ("synonyms", "text")])
    except Exception as e:
        print("create text index (alt):", e)

if __name__ == "__main__":
    main()

# ==========================================

# 100 best rated

import mysql.connector
import pandas as pd

# ---------- CONFIG ----------
db_config = {
    "host": "mars-db.cdcmiuuwqo5z.eu-north-1.rds.amazonaws.com",
    "user": "msamosudova",           # or ysagan / amaksymchuk
    "password": "Duckling25!",
    "database": "mars_db",
    "port": 3306
}

# ---------- SQL QUERY ----------
query = """
-- Get top 100 best rated anime (based on MAL score, fallback-safe)
SELECT
    a.MAL_ID,
    a.name,
    COALESCE(s.score, 0) AS rating,
    s.members,
    s.favorites,
    s.popularity
FROM anime a
JOIN anime_statistics s USING (MAL_ID)
WHERE s.score IS NOT NULL
  AND COALESCE(s.members, 0) >= 1000     -- small popularity guard
ORDER BY s.score DESC, s.members DESC
LIMIT 100;
"""

# ---------- EXECUTION ----------
try:
    conn = mysql.connector.connect(**db_config)
    df = pd.read_sql(query, conn)

    print("\n=== TOP 100 BEST-RATED ANIME ===")
    print(df.head(20).to_string(index=False))  # show first 20
    print(f"\nTotal rows fetched: {len(df)}")

except mysql.connector.Error as e:
    print("MySQL Error:", e)

finally:
    if 'conn' in locals() and conn.is_connected():
        conn.close()
        print("Connection closed.")

# ==========================================

import mysql.connector
from pymongo import MongoClient
import pandas as pd

# ---------- CONFIG ----------
sql_config = {
    "host": "mars-db.cdcmiuuwqo5z.eu-north-1.rds.amazonaws.com",
    "user": "msamosudova",
    "password": "Duckling25!",
    "database": "mars_db"
}
mongo_uri = "mongodb+srv://msamosudova:Duckling@mars-cluster.8ruotdw.mongodb.net/?appName=MARS-Cluster"
mongo_db = "anime_db"
mongo_coll = "animes"

# ---------- STEP 1 — Top 100 IDs from MySQL ----------
sql = mysql.connector.connect(**sql_config)
query = """
SELECT a.MAL_ID, a.name, s.score
FROM anime a
JOIN anime_statistics s USING (MAL_ID)
WHERE s.score IS NOT NULL
ORDER BY s.score DESC, s.members DESC
LIMIT 100;
"""
df = pd.read_sql(query, sql)
sql.close()

# ---------- STEP 2 — Fetch synopsis from MongoDB ----------
client = MongoClient(mongo_uri)
coll = client[mongo_db][mongo_coll]

syn_map = {
    doc["_id"]: doc.get("synopsis", "")
    for doc in coll.find(
        {"_id": {"$in": df["MAL_ID"].tolist()}},
        {"_id": 1, "synopsis": 1}
    )
}

df["synopsis"] = df["MAL_ID"].map(syn_map)

# ---------- STEP 3 — Display ----------
print("\n=== TOP 100 ANIME WITH SYNOPSIS (MySQL + Mongo) ===")
for _, row in df.iterrows():
    text = (row["synopsis"] or "")[:400]
    print(f"\n{row['name']}  ({row['score']})\n{text}...")

print(f"\nTotal rows fetched: {len(df)}")

# ==========================================

# fetch_top_r_anime_with_genres.py
# Top 30 best-rated anime with age rating starting with “R”
# It also shows all associated genres for each title from MongoDB

import mysql.connector
from mysql.connector import Error
from pymongo import MongoClient

# ---------- MySQL CONFIG ----------
MYSQL = {
    "host": "mars-db.cdcmiuuwqo5z.eu-north-1.rds.amazonaws.com",
    "user": "msamosudova",
    "password": "Duckling25!",
    "database": "mars_db",
    "port": 3306,
}

# ---------- MongoDB CONFIG ----------
MONGO_URI = "mongodb+srv://msamosudova:Duckling@mars-cluster.8ruotdw.mongodb.net/?appName=MARS-Cluster"
MONGO_DB  = "anime_db"
MONGO_COL = "animes"

# ---------- PARAMETERS ----------
AGE_RATING_FILTER = "R%"   # matches R-17+ and R+
MIN_MEMBERS = 1000
LIMIT = 30

SQL_TOP_R = """
SELECT
    a.MAL_ID,
    a.name,
    s.score  AS rating,
    s.members
FROM anime a
JOIN anime_statistics s USING (MAL_ID)
JOIN age_rating ar ON ar.age_ratingID = a.age_ratingID
WHERE ar.age_rating LIKE %s
  AND s.score IS NOT NULL
  AND COALESCE(s.members, 0) >= %s
ORDER BY s.score DESC, s.members DESC
LIMIT %s;
"""

def main():
    # 1) Fetch candidates from MySQL
    try:
        conn = mysql.connector.connect(**MYSQL)
        cur = conn.cursor()
        cur.execute(SQL_TOP_R, (AGE_RATING_FILTER, MIN_MEMBERS, LIMIT))
        rows = cur.fetchall()  # list of tuples: (MAL_ID, name, rating, members)
    except Error as e:
        print("MySQL Error:", e); return
    finally:
        try: cur.close(); conn.close()
        except: pass

    if not rows:
        print("No results from MySQL."); return

    # 2) Pull genres from Mongo for those MAL_IDs
    ids = [int(r[0]) for r in rows]
    client = MongoClient(MONGO_URI)
    coll = client[MONGO_DB][MONGO_COL]

    cursor = coll.find({"_id": {"$in": ids}}, {"_id": 1, "genres": 1})
    genres_map = {}
    for d in cursor:
        g = d.get("genres")
        # normalize to list[str]
        if isinstance(g, list):
            genres_map[int(d["_id"])] = [str(x) for x in g if x]
        elif isinstance(g, str):
            # in case genres accidentally stored as comma-separated string
            genres_map[int(d["_id"])] = [s.strip() for s in g.split(",") if s.strip()]
        else:
            genres_map[int(d["_id"])] = []

    # 3) Print
    print(f"\nTop {LIMIT} R-rated anime (genres from Mongo):\n")
    for mal_id, name, score, members in rows:
        glist = genres_map.get(int(mal_id), [])
        genres_str = ", ".join(sorted(set(glist))) if glist else "—"
        print(f"- [{mal_id}] {name} — {score}| members={members:,} | genres: {genres_str}")

if __name__ == "__main__":
    main()

# ==========================================
"""
WITH user_genre_pref AS (
    SELECT
        ag.genre,
        AVG(r.score) AS avg_score
    FROM ratings r
    JOIN anime_genres ag ON r.anime_id = ag.anime_id
    WHERE r.user_id = @user_id
    GROUP BY ag.genre
),

best_genre AS (
    SELECT genre
    FROM user_genre_pref
    ORDER BY avg_score DESC
    LIMIT 1
),

user_watched AS (
    SELECT anime_id
    FROM ratings
    WHERE user_id = @user_id
)

SELECT
    a.id,
    a.title,
    a.global_score
FROM anime a
JOIN anime_genres ag ON a.id = ag.anime_id
WHERE ag.genre IN (SELECT genre FROM best_genre)
  AND a.id NOT IN (SELECT anime_id FROM user_watched)
ORDER BY a.global_score DESC
LIMIT 5;
"""