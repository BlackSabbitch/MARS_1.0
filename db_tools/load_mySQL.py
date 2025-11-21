import mysql
import pandas as pd
import re
from pathlib import Path
from db_tools.database_connection_manager import DatabaseConnectionManager


class LoadMySQL:
    """
    Loads all structured data into MySQL based on the project schema.
    Uses DatabaseConnectionManager for all DB operations.
    """

    def __init__(self, db: DatabaseConnectionManager):
        self.db = db
        self.paths = {}

    # ----------------------------------------------------------------------
    # LOAD CSV PATHS
    # ----------------------------------------------------------------------

    def load_paths(self, config: dict[str, str]):
        """
        Loads file paths for CSV datasets.
        Example:
            loader.load_paths({
                "anime": "DataSet/anime.csv",
                "users": "DataSet/users.csv",
                "ratings": "DataSet/rating_complete.csv"
            })
        """
        self.paths = config

    def read_csv(self, key: str) -> pd.DataFrame:
        """Read CSV by key from loaded paths."""
        if key not in self.paths:
            raise ValueError(f"CSV path for key '{key}' not loaded.")
        return pd.read_csv(self.paths[key])

    # ----------------------------------------------------------------------
    # UTILS
    # ----------------------------------------------------------------------

    @staticmethod
    def split_list(cell):
        """Split comma/pipe-separated values."""
        if not isinstance(cell, str):
            return []
        parts = re.split(r"[|;,]", cell.strip())
        return [p.strip() for p in parts if p.strip()]

    # ----------------------------------------------------------------------
    # MAIN LOADERS
    # ----------------------------------------------------------------------

    def load_anime(self):
        """
        Loads anime.csv into:
        - type lookup table
        - source lookup table
        - age_rating lookup table
        - anime main table
        Fully consistent with schema.sql
        """
        print("\n=== Loading anime (with types, sources, age ratings) ===")

        df = self.read_csv("anime")

        conn = self.db.get_mysql_connection()
        cur = self.db.get_mysql_cursor()

        # -------------------------------------------------------------
        # 1. LOAD TYPE LOOKUP
        # -------------------------------------------------------------
        types = set(df["Type"].dropna().unique())
        cur.executemany(
            "INSERT IGNORE INTO type (type) VALUES (%s)",
            [(t,) for t in types]
        )
        conn.commit()

        cur.execute("SELECT typeID, type FROM type")
        type_map = {t: tid for tid, t in cur.fetchall()}

        print(f"Loaded {len(types)} types")

        # -------------------------------------------------------------
        # 2. LOAD SOURCE LOOKUP
        # -------------------------------------------------------------
        sources = set(df["Source"].dropna().unique())
        cur.executemany(
            "INSERT IGNORE INTO source (source) VALUES (%s)",
            [(s,) for s in sources]
        )
        conn.commit()

        cur.execute("SELECT sourceID, source FROM source")
        source_map = {s: sid for sid, s in cur.fetchall()}

        print(f"Loaded {len(sources)} sources")

        # -------------------------------------------------------------
        # 3. LOAD AGE-RATING LOOKUP
        # -------------------------------------------------------------
        ratings = set(df["Rating"].dropna().unique())

        cur.executemany(
            "INSERT IGNORE INTO age_rating (age_rating) VALUES (%s)",
            [(r,) for r in ratings]
        )
        conn.commit()

        cur.execute("SELECT age_ratingID, age_rating FROM age_rating")
        age_rating_map = {r: rid for rid, r in cur.fetchall()}

        print(f"Loaded {len(ratings)} age ratings")

        # -------------------------------------------------------------
        # 4. LOAD ANIME TABLE
        # -------------------------------------------------------------
        rows = []

        for _, r in df.iterrows():
            mal_id = int(r["MAL_ID"])
            name = r["Name"]

            ep = r.get("Episodes")
            if pd.isna(ep):
                episodes = None
            else:
                try:
                    episodes = int(ep)
                except ValueError:
                    episodes = None

            aired = r.get("Aired")
            premiered = r.get("Premiered")
            duration = r.get("Duration")

            typeID = type_map.get(r["Type"])
            sourceID = source_map.get(r["Source"])
            age_ratingID = age_rating_map.get(r["Rating"])

            rows.append([
                mal_id,
                name,
                episodes,
                aired,
                premiered,
                duration,
                sourceID,
                typeID,
                age_ratingID
            ])

        cur.executemany("""
            INSERT INTO anime
            (MAL_ID, name, episodes, aired, premiered, duration,
                sourceID, typeID, age_ratingID)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON DUPLICATE KEY UPDATE
                name=VALUES(name),
                episodes=VALUES(episodes),
                aired=VALUES(aired),
                premiered=VALUES(premiered),
                duration=VALUES(duration),
                sourceID=VALUES(sourceID),
                typeID=VALUES(typeID),
                age_ratingID=VALUES(age_ratingID)
        """, rows)

        conn.commit()
        print(f"Inserted/updated {len(rows)} anime rows.\n")
        cur.close()


    # ----------------------------------------------------------------------

    def load_genres(self):
        """Load genre lookup table and mapping anime_genre."""
        print("\n=== Loading genres and anime_genre mapping ===")

        df = self.read_csv("anime")

        conn = self.db.get_mysql_connection()
        cur = self.db.get_mysql_cursor()

        # 1. genre lookup
        genres = set()
        for g in df["Genres"].dropna():
            genres.update(self.split_list(g))

        genres = sorted(genres)

        cur.executemany(
            "INSERT IGNORE INTO genre (genre) VALUES (%s)",
            [(g,) for g in genres]
        )
        conn.commit()
        print(f"Inserted {len(genres)} unique genres.")

        # 2. mapping table
        cur.execute("SELECT genreID, genre FROM genre")
        genre_map = {g: gid for gid, g in cur.fetchall()}

        rows = []
        for _, r in df.iterrows():
            mal_id = int(r["MAL_ID"])
            g_list = self.split_list(r["Genres"])
            for g in g_list:
                gid = genre_map.get(g)
                if gid:
                    rows.append((mal_id, gid))

        cur.executemany(
            "INSERT IGNORE INTO anime_genre (MAL_ID, genreID) VALUES (%s,%s)",
            rows
        )
        conn.commit()

        print(f"Inserted {len(rows)} anime_genre mappings.")
        cur.close()

    # ----------------------------------------------------------------------

    def load_users_from_animelist(self):
        """
        Loads unique user IDs from animelist.csv into `users` table.
        This is the primary source of user IDs in the Kaggle dataset.
        """
        print("\n=== Loading user IDs from animelist.csv ===")

        df = self.read_csv("animelist")

        conn = self.db.get_mysql_connection()
        cur = self.db.get_mysql_cursor()

        user_ids = sorted(set(int(u) for u in df["user_id"].dropna()))

        cur.executemany(
            "INSERT IGNORE INTO users (userID) VALUES (%s)",
            [(uid,) for uid in user_ids]
        )

        conn.commit()
        print(f"Inserted {len(user_ids)} users.")
        cur.close()

    # ----------------------------------------------------------------------

    def load_profiles_map(self):
        """
        Production-safe FAST profile mapping using a temporary table + UPDATE JOIN.
        Loads 325k profiles into tmp table and updates all users in one SQL.
        """

        print("\n=== Profile mapping ===")

        df = self.read_csv("profiles")
        df = df.where(pd.notnull(df), None)
        profiles_len = len(df)
        print(f"Loaded {profiles_len} profiles")

        conn = self.db.get_mysql_connection()
        cur = conn.cursor()

        # 1. Create TEMP TABLE
        print("Creating temporary table tmp_profiles...")
        cur.execute("DROP TABLE IF EXISTS tmp_profiles")
        cur.execute("""
            CREATE TABLE tmp_profiles (
                sex VARCHAR(16),
                age SMALLINT,
                country VARCHAR(64),
                city VARCHAR(64),
                latitude FLOAT,
                longitude FLOAT,
                mapping_idx INT PRIMARY KEY
            ) ENGINE=InnoDB;
        """)
        conn.commit()

        # 2. Insert profiles (batched)
        print("Inserting profiles into tmp_profiles...")
        sql_insert = """
            INSERT INTO tmp_profiles
            (sex, age, country, city, latitude, longitude, mapping_idx)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
        """

        batch = []
        batch_size = 10000
        idx = 0

        for _, row in df.iterrows():
            batch.append((
                row["sex"],
                int(row["age"]) if row["age"] is not None else None,
                row["country"],
                row["city"],
                float(row["latitude"]) if row["latitude"] is not None else None,
                float(row["longitude"]) if row["longitude"] is not None else None,
                idx
            ))
            idx += 1

            if len(batch) >= batch_size:
                cur.executemany(sql_insert, batch)
                conn.commit()
                print(f"Inserted {idx}/{profiles_len}")
                batch.clear()

        if batch:
            cur.executemany(sql_insert, batch)
            conn.commit()
            print(f"Inserted all {profiles_len} profiles")

        # 3. Fetch rating users
        print("Fetching rating users...")
        cur.execute("SELECT userID FROM users ORDER BY userID")
        rating_users = [u[0] for u in cur.fetchall()]
        total_users = len(rating_users)
        print(f"Rating users: {total_users}")

        # 4. Create mapping table for users
        print("Creating tmp_user_map...")
        cur.execute("DROP TABLE IF EXISTS tmp_user_map")
        cur.execute("""
            CREATE TABLE tmp_user_map (
                userID INT PRIMARY KEY,
                mapping_idx INT
            ) ENGINE=InnoDB;
        """)
        conn.commit()

        # 5. Fill user→profile mapping (cyclic)
        print("Inserting mapping rows...")
        batch = []
        idx = 0

        for uid in rating_users:
            batch.append((uid, idx % profiles_len))
            idx += 1

            if len(batch) >= batch_size:
                cur.executemany(
                    "INSERT INTO tmp_user_map (userID, mapping_idx) VALUES (%s,%s)",
                    batch
                )
                conn.commit()
                batch.clear()

        if batch:
            cur.executemany(
                "INSERT INTO tmp_user_map (userID, mapping_idx) VALUES (%s,%s)",
                batch
            )
            conn.commit()

        print("Mapping table filled")

        # 6. Single-shot UPDATE with JOIN
        print("Updating users via JOIN ...")
        cur.execute("""
            UPDATE users u
            JOIN tmp_user_map m ON u.userID = m.userID
            JOIN tmp_profiles p ON p.mapping_idx = m.mapping_idx
            SET 
                u.sex       = p.sex,
                u.age       = p.age,
                u.country   = p.country,
                u.city      = p.city,
                u.latitude  = p.latitude,
                u.longitude = p.longitude;
        """)
        conn.commit()

        print("\n=== DONE: Users updated ===")

        # Cleanup
        cur.execute("DROP TABLE IF EXISTS tmp_profiles")
        cur.execute("DROP TABLE IF EXISTS tmp_user_map")
        conn.commit()

        cur.close()

    # ----------------------------------------------------------------------

    def load_rating_complete(self):
        """
        Optimized loader for rating_complete.csv:
        user_id, anime_id, rating.
        """
        print("\n=== Loading rating_complete.csv ===")

        path = self.paths.get("ratings")
        if not path:
            print("ratings path not configured")
            return

        conn = self.db.get_mysql_connection()
        cur = self.db.get_mysql_cursor()

        # Disable FK & indexes for maximum speed
        cur.execute("SET FOREIGN_KEY_CHECKS = 0")
        cur.execute("ALTER TABLE anime_user_rating DISABLE KEYS")
        conn.commit()

        import csv

        batch = []
        batch_size = 10000
        total = 0

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                try:
                    uid = int(row["user_id"])
                    mal = int(row["anime_id"])

                    rating = row["rating"]
                    rating = float(rating) if rating not in ("", "Unknown", None) else None

                    batch.append((mal, uid, rating))

                except Exception:
                    continue

                if len(batch) >= batch_size:
                    cur.executemany("""
                        INSERT INTO anime_user_rating (MAL_ID, userID, user_rating)
                        VALUES (%s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                            user_rating = VALUES(user_rating)
                    """, batch)
                    conn.commit()
                    total += len(batch)
                    print(f"Committed {total:,} rows...")
                    batch.clear()

        # Final batch
        if batch:
            cur.executemany("""
                INSERT INTO anime_user_rating (MAL_ID, userID, user_rating)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    user_rating = VALUES(user_rating)
            """, batch)
            conn.commit()
            total += len(batch)
            print(f"Committed {total:,} rows...")

        # Re-enable keys
        cur.execute("ALTER TABLE anime_user_rating ENABLE KEYS")
        cur.execute("SET FOREIGN_KEY_CHECKS = 1")
        conn.commit()

        cur.close()
        print(f"\n=== DONE: Loaded {total:,} rating rows ===\n")

    # ----------------------------------------------------------------------

    def load_score_distribution(self, df_anime):
        """
        Loads Score-1 ... Score-10 → score_1 ... score_10
        Uses batching to avoid MySQL lock.
        """
        print("\n=== Loading score distributions (batched) ===")

        conn = self.db.get_mysql_connection()
        cur = self.db.get_mysql_cursor()

        source_cols = [f"Score-{i}" for i in range(1, 11)]
        target_cols = [f"score_{i}" for i in range(1, 11)]

        # SET clause
        set_clause = ", ".join([f"{t}=%s" for t in target_cols])
        sql = f"UPDATE anime_statistics SET {set_clause} WHERE MAL_ID=%s"

        updates = []
        for _, r in df_anime.iterrows():
            mal_id = int(r["MAL_ID"])
            dist = [r.get(f"Score-{i}", None) for i in range(1, 11)]
            updates.append(dist + [mal_id])

        batch_size = 100
        done = 0

        # BATCHED UPDATE
        for i in range(0, len(updates), batch_size):
            batch = updates[i:i + batch_size]
            cur.executemany(sql, batch)
            conn.commit()
            done += len(batch)
            print(f"Committed {done}/{len(updates)}")

        print("Score distribution updated.")
        cur.close()

    def load_watching_status(self):

        print("\n=== Loading watching_status lookup ===")

        conn = self.db.get_mysql_connection()
        cur = conn.cursor()

        statuses = [
            (1, "Watching"),
            (2, "Completed"),
            (3, "On-Hold"),
            (4, "Dropped"),
            (5, "Plan to Watch"),
            (6, "Unknown")
        ]

        sql = """
            INSERT INTO watching_status (watching_statusID, watching_status)
            VALUES (%s, %s)
            ON DUPLICATE KEY UPDATE watching_status=VALUES(watching_status)
        """

        cur.executemany(sql, statuses)
        conn.commit()
        cur.close()

        print("watching_status table populated.")

    def load_animelist(self):
        """
        Load animelist.csv into anime_user_rating in a safe, CHECK-compatible way.
        - rating = 0 → NULL
        - rating outside 1-10 → NULL
        - missing watching_status → NULL
        - missing watched_episodes → NULL
        """

        print("\n=== Loading animelist.csv ===")

        df = self.read_csv("animelist")

        print("Normalizing records...")

        # normalization
        df = df[["user_id", "anime_id", "rating", "watching_status", "watched_episodes"]]

        BATCH = 5000
        total_rows = len(df)
        processed = 0

        sql = """
            INSERT INTO anime_user_rating (
                userID, MAL_ID, user_rating, watching_statusID, watched_episodes
            )
            VALUES (%s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                user_rating = VALUES(user_rating),
                watching_statusID = VALUES(watching_statusID),
                watched_episodes = VALUES(watched_episodes)
        """

        conn = self.db.get_mysql_connection()
        cur = conn.cursor()

        batch = []

        for idx, row in df.iterrows():

            user_id = row["user_id"]
            mal_id = row["anime_id"]

            # --- RATING FIX: 0 => NULL ---
            raw_rating = row["rating"]
            rating = None

            try:
                if pd.notna(raw_rating):
                    r = int(raw_rating)
                    if 1 <= r <= 10:
                        rating = r
                    # r = 0 → NULL
            except:
                rating = None

            # --- WATCHING_STATUS ---
            ws = row["watching_status"]
            try:
                ws = int(ws) if pd.notna(ws) else None
            except:
                ws = None

            # --- WATCHED_EPISODES ---
            we = row["watched_episodes"]
            try:
                we = int(we) if pd.notna(we) else None
            except:
                we = None

            batch.append((user_id, mal_id, rating, ws, we))

            if len(batch) >= BATCH:
                try:
                    cur.executemany(sql, batch)
                    conn.commit()
                except mysql.connector.Error as e:
                    print("ERROR in batch:", e)
                    print("Last row:", batch[-1])
                    raise

                processed += len(batch)
                print(f"Committed {processed}/{total_rows} rows...")
                batch = []

        # commit last batch
        if batch:
            cur.executemany(sql, batch)
            conn.commit()
            processed += len(batch)

        print(f"=== DONE: Loaded {processed} rows into anime_user_rating ===")

        cur.close()
        conn.close()



