

import os, csv, re
import mysql.connector
from mysql.connector import Error
from mysql.connector import errorcode
from mySQL_tools import MySQLTools
from common_tools import CommonTools  
from mySQL_tools import MySQLTools

class LoadMySQL:

    anime_csv = CommonTools.load_csv_files_pathes()['anime']
    users_csv = CommonTools.load_csv_files_pathes()['users']
    animelist_csv = CommonTools.load_csv_files_pathes()['animelist']
    rating_complete_csv = CommonTools.load_csv_files_pathes()['rating_complete']
    watching_status_csv = CommonTools.load_csv_files_pathes()['watching_status']

    # ---------- UTILS ----------
    @staticmethod
    def split_list(cell):
        """split comma- or pipe-separated lists from dataset"""
        if cell is None:
            return []
        s = str(cell).strip()
        if not s or s.upper() in {"UNKNOWN","NONE","NULL","N/A"}:
            return []

        parts = re.split(r'[|;,]', s)
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def clean_str(s):
        if s is None:
            return None
        s = str(s).strip()
        return s if s else None
    
    @staticmethod
    def to_int(s):
        try:
            return int(s)
        except:
            return None
        
    @staticmethod
    def to_float2(s):
        try:
            return round(float(s), 2)
        except:
            return None

    @staticmethod
    def insert_ignore(cursor, table, col, val):
        cursor.execute(f"INSERT IGNORE INTO {table} ({col}) VALUES (%s)", (val,))

    @staticmethod
    def get_id(cursor, table, id_col, name_col, val, cache):
        """Upsert into lookup and return id (cached)."""
        if val is None:
            return None
        if val in cache:
            return cache[val]
        # try select
        cursor.execute(f"SELECT {id_col} FROM {table} WHERE {name_col}=%s", (val,))
        row = cursor.fetchone()
        if row:
            cache[val] = row[0]
            return row[0]
        # insert
        cursor.execute(f"INSERT INTO {table} ({name_col}) VALUES (%s)", (val,))
        cache[val] = cursor.lastrowid
        return cache[val]
    
    @staticmethod
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

    # ---------- LOAD 1 ----------
    # load_anime
    # Loads anime.csv into: type, source, age_rating, genre, studio, producer, licensor,
    # and fills anime, anime_statistics, plus bridges: anime_genre, anime_studio, anime_producer, anime_licensor.
    def load_anime(batch_size: int = 1000):

        connection = MySQLTools.__init__.self.connection
        if not os.path.exists(LoadMySQL.anime_csv):
            raise FileNotFoundError(LoadMySQL.anime_csv)
        
        connection.autocommit = False
        cursor = connection.cursor()

        type_cache = {}
        source_cache = {}
        age_cache = {}
        genre_cache = {}
        studio_cache = {}
        producer_cache = {}
        licensor_cache = {}

        # Counters
        n_anime = n_stats = n_gen = n_st = n_prod = n_lic = 0

        with open(LoadMySQL.anime_csv, encoding="utf-8") as f:
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
            mal_id  = LoadMySQL.to_int(col("MAL_ID", r))
            if not mal_id:  # skip bad rows
                continue

            name    = LoadMySQL.clean_str(col("Name", r))
            episodes= LoadMySQL.to_int(col("Episodes", r))
            aired   = LoadMySQL.clean_str(col("Aired", r))
            premiered = LoadMySQL.clean_str(col("Premiered", r))
            duration = LoadMySQL.clean_str(col("Duration", r))

            # lookups single-valued
            type_val   = LoadMySQL.clean_str(col("Type", r))
            type_id    = LoadMySQL.get_id(cursor, "type", "typeID", "type", type_val, type_cache) if type_val else None

            source_val = LoadMySQL.clean_str(col("Source", r))
            source_id  = LoadMySQL.get_id(cursor, "source", "sourceID", "source", source_val, source_cache) if source_val else None

            age_val    = LoadMySQL.clean_str(col("Rating", r)) or LoadMySQL.clean_str(col("age_rating", r))
            age_id     = LoadMySQL.get_id(cursor, "age_rating", "age_ratingID", "age_rating", age_val, age_cache) if age_val else None

            # upsert anime
            anime_batch.append((
                mal_id, name, episodes, aired, premiered, duration, source_id, type_id, age_id
            ))

            # stats (denormalized snapshot in anime_statistics)
            stats_batch.append((
                mal_id,
                LoadMySQL.to_float2(col("Score", r)),
                LoadMySQL.to_int(col("Rank", r)),
                LoadMySQL.to_int(col("Popularity", r)),
                LoadMySQL.to_int(col("Members", r)),
                LoadMySQL.to_int(col("Favorites", r)),
                LoadMySQL.to_int(col("Watching", r)),
                LoadMySQL.to_int(col("Completed", r)),
                LoadMySQL.to_int(col("On-Hold", r)) or LoadMySQL.to_int(col("On_Hold", r)),
                LoadMySQL.to_int(col("Dropped", r)),
                LoadMySQL.to_int(col("Plan to Watch", r)) or LoadMySQL.to_int(col("Plan_to_Watch", r))
            ))

            # multi-valued lists -> lookups + bridges
            for g in LoadMySQL.split_list(col("Genres", r)):
                gid = LoadMySQL.get_id(cursor, "genre", "genreID", "genre", g, genre_cache)
                gen_bridge.append((mal_id, gid))

            for st in LoadMySQL.split_list(col("Studios", r)):
                sid = LoadMySQL.get_id(cursor, "studio", "studioID", "studio", st, studio_cache)
                st_bridge.append((mal_id, sid))

            for pr in LoadMySQL.split_list(col("Producers", r)):
                pid = LoadMySQL.get_id(cursor, "producer", "producerID", "producer", pr, producer_cache)
                prod_bridge.append((mal_id, pid))

            for lc in LoadMySQL.split_list(col("Licensors", r)):
                lid = LoadMySQL.get_id(cursor, "licensor", "licensorID", "licensor", lc, licensor_cache)
                lic_bridge.append((mal_id, lid))

            if len(anime_batch) >= batch_size:
                cursor.executemany("""
                    INSERT INTO anime (MAL_ID, name, episodes, aired, premiered, duration, sourceID, typeID, age_ratingID)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON DUPLICATE KEY UPDATE
                    name=VALUES(name), episodes=VALUES(episodes), aired=VALUES(aired),
                    premiered=VALUES(premiered), duration=VALUES(duration),
                    sourceID=VALUES(sourceID), typeID=VALUES(typeID), age_ratingID=VALUES(age_ratingID)
                """, anime_batch)
                n_anime += len(anime_batch); anime_batch.clear()

            if len(stats_batch) >= batch_size:
                cursor.executemany("""
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
            if len(gen_bridge) >= batch_size:
                cursor.executemany("INSERT IGNORE INTO anime_genre (MAL_ID, genreID) VALUES (%s,%s)", gen_bridge)
                n_gen += len(gen_bridge); gen_bridge.clear()

            if len(st_bridge) >= batch_size:
                cursor.executemany("INSERT IGNORE INTO anime_studio (MAL_ID, studioID) VALUES (%s,%s)", st_bridge)
                n_st += len(st_bridge); st_bridge.clear()

            if len(prod_bridge) >= batch_size:
                cursor.executemany("INSERT IGNORE INTO anime_producer (MAL_ID, producerID) VALUES (%s,%s)", prod_bridge)
                n_prod += len(prod_bridge); prod_bridge.clear()

            if len(lic_bridge) >= batch_size:
                cursor.executemany("INSERT IGNORE INTO anime_licensor (MAL_ID, licensorID) VALUES (%s,%s)", lic_bridge)
                n_lic += len(lic_bridge); lic_bridge.clear()

        # flush remaining
        if anime_batch:
            cursor.executemany("""
                INSERT INTO anime (MAL_ID, name, episodes, aired, premiered, duration, sourceID, typeID, age_ratingID)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON DUPLICATE KEY UPDATE
                name=VALUES(name), episodes=VALUES(episodes), aired=VALUES(aired),
                premiered=VALUES(premiered), duration=VALUES(duration),
                sourceID=VALUES(sourceID), typeID=VALUES(typeID), age_ratingID=VALUES(age_ratingID)
            """, anime_batch)
            n_anime += len(anime_batch)

        if stats_batch:
            cursor.executemany("""
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
            cursor.executemany("INSERT IGNORE INTO anime_genre (MAL_ID, genreID) VALUES (%s,%s)", gen_bridge)
            n_gen += len(gen_bridge)
        if st_bridge:
            cursor.executemany("INSERT IGNORE INTO anime_studio (MAL_ID, studioID) VALUES (%s,%s)", st_bridge)
            n_st += len(st_bridge)
        if prod_bridge:
            cursor.executemany("INSERT IGNORE INTO anime_producer (MAL_ID, producerID) VALUES (%s,%s)", prod_bridge)
            n_prod += len(prod_bridge)
        if lic_bridge:
            cursor.executemany("INSERT IGNORE INTO anime_licensor (MAL_ID, licensorID) VALUES (%s,%s)", lic_bridge)
            n_lic += len(lic_bridge)

        connection.commit()
        cursor.close(); connection.close()
        print(f"anime upserted: {n_anime}, stats: {n_stats}, links — genres:{n_gen} studios:{n_st} producers:{n_prod} licensors:{n_lic}")

# ---------- LOAD 2 ----------
# 02_load_users_and_ratings
# - autocommit=True (without long transactions)
# - ensure users before facts
# - validate MAL_ID against anime
# - validate watching_statusID against lookup; unknown/0 -> NULL

# ---------- LOOKUPS FROM DB ----------
    def load_known_anime_ids(cursor):
        cursor.execute("SELECT MAL_ID FROM anime")
        return {row[0] for row in cursor.fetchall()}

    def load_known_status_ids(cursor):
        cursor.execute("SELECT watching_statusID FROM watching_status")
        return {row[0] for row in cursor.fetchall()}

    # ---------- LOAD lookup files ----------
    def upsert_watching_status(cursor):
        """Load watching_status.csv if present."""
        if not os.path.exists(LoadMySQL.watching_status_csv):
            return 0
        rows = []
        with open(LoadMySQL.watching_status_csv, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                sid = LoadMySQL.to_int_safe(r.get("status_id"))
                status = (r.get("status") or "").strip()
                if sid is None or not status:
                    continue
                rows.append((sid, status))
        if rows:
            cursor.executemany(
                "INSERT INTO watching_status (watching_statusID, watching_status) VALUES (%s,%s) "
                "ON DUPLICATE KEY UPDATE watching_status=VALUES(watching_status)",
                rows
            )
        return len(rows)

    def insert_users_from_users_csv(cursor, batch_users: int = 10000):
        if not os.path.exists(LoadMySQL.users_csv):
            return 0
        buf, total = [], 0
        with open(LoadMySQL.users_csv, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                uid = LoadMySQL.to_int_safe(r.get("user_id") or r.get("userID"))
                if uid is None:
                    continue
                buf.append((uid,))
                if len(buf) >= batch_users:
                    cursor.executemany("INSERT IGNORE INTO users (userID) VALUES (%s)", buf)
                    total += len(buf); buf.clear()
        if buf:
            cursor.executemany("INSERT IGNORE INTO users (userID) VALUES (%s)", buf)
            total += len(buf)
        return total

    def collect_user_ids_from_file(path):
        users = set()
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                uid = LoadMySQL.to_int_safe(r.get("user_id") or r.get("userID"))
                if uid is not None:
                    users.add(uid)
        return users

    def ensure_users_for_source(cursor, source_path, batch_users: int = 10000):
        if not os.path.exists(source_path):
            return 0
        users = sorted(LoadMySQL.collect_user_ids_from_file(source_path))
        total, buf = 0, []
        for uid in users:
            buf.append((uid,))
            if len(buf) >= batch_users:
                cursor.executemany("INSERT IGNORE INTO users (userID) VALUES (%s)", buf)
                total += len(buf); buf.clear()
        if buf:
            cursor.executemany("INSERT IGNORE INTO users (userID) VALUES (%s)", buf)
            total += len(buf)
        return total

    # ---------- FACT LOADERS ----------
    def load_ratings_from_animelist(cursor, known_anime, known_status, batch_ratings: int = 5000):
        """
        animelist.csv: user_id, anime_id, rating, watching_status, watched_episodes
        - rating 0 -> NULL
        - watching_status not in lookup (including 0) -> NULL
        """
        if not os.path.exists(LoadMySQL.animelist_csv):
            return 0
        cnt, batch, unknown_ws = 0, [], 0

        with open(LoadMySQL.animelist_csv, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                uid = LoadMySQL.to_int_safe(r.get("user_id") or r.get("userID"))
                aid = LoadMySQL.to_int_safe(r.get("anime_id") or r.get("MAL_ID"))
                if uid is None or aid is None or aid not in known_anime:
                    continue

                rating = LoadMySQL.to_int_safe(r.get("rating") or r.get("score"))
                if rating is not None and rating == 0:
                    rating = None

                ws = LoadMySQL.to_int_safe(r.get("watching_status"))
                if ws is None or ws not in known_status:
                    ws = None
                    unknown_ws += 1

                we = LoadMySQL.to_int_safe(r.get("watched_episodes"))

                batch.append((aid, uid, rating, ws, we))

                if len(batch) >= batch_ratings:
                    batch.sort(key=lambda x: (x[1], x[0]))
                    cursor.executemany("""
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
            cursor.executemany("""
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

    def load_ratings_from_rating_complete(cursor, known_anime, batch_ratings: int = 5000):
        """
        rating_complete.csv: user_id, anime_id, rating (1..10)
        """
        if not os.path.exists(LoadMySQL.rating_complete_csv):
            return 0
        cnt, batch = 0, []

        with open(LoadMySQL.rating_complete_csv, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                uid = LoadMySQL.to_int_safe(r.get("user_id") or r.get("userID"))
                aid = LoadMySQL.to_int_safe(r.get("anime_id") or r.get("MAL_ID"))
                rating = LoadMySQL.to_int_safe(r.get("rating") or r.get("score"))
                if uid is None or aid is None or rating is None or rating == 0:
                    continue
                if aid not in known_anime:
                    continue

                batch.append((aid, uid, rating))

                if len(batch) >= batch_ratings:
                    batch.sort(key=lambda x: (x[1], x[0]))
                    cursor.executemany("""
                        INSERT INTO anime_user_rating (MAL_ID, userID, user_rating)
                        VALUES (%s,%s,%s)
                        ON DUPLICATE KEY UPDATE user_rating=VALUES(user_rating)
                    """, batch)
                    cnt += len(batch); batch.clear()

        if batch:
            batch.sort(key=lambda x: (x[1], x[0]))
            cursor.executemany("""
                INSERT INTO anime_user_rating (MAL_ID, userID, user_rating)
                VALUES (%s,%s,%s)
                ON DUPLICATE KEY UPDATE user_rating=VALUES(user_rating)
            """, batch)
            cnt += len(batch)

        return cnt

    def load_users_and_ratings_and_watching_status(batch_users: int = 10000, batch_ratings: int = 5000):
        
        if not os.path.exists(MySQLTools.users_csv):
            raise FileNotFoundError(MySQLTools.users_csv)
        if not os.path.exists(MySQLTools.animelist_csv):
            raise FileNotFoundError(MySQLTools.animelist_csv)
        if not os.path.exists(MySQLTools.rating_complete_csv):
            raise FileNotFoundError(MySQLTools.rating_complete_csv)
        if not os.path.exists(MySQLTools.watching_status_csv):
            raise FileNotFoundError(MySQLTools.watching_status_csv)
        
        connection = MySQLTools.__init__.self.connection
        connection.autocommit = True
        cursor = connection.cursor()

        # session setup
        cursor.execute("SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED")
        cursor.execute("SET SESSION innodb_lock_wait_timeout = 50")

        # lookup from db
        known_anime = LoadMySQL.load_known_anime_ids(cursor)
        print("known anime:", len(known_anime))

        # watching_status.csv (if exists)
        ws_loaded = LoadMySQL.upsert_watching_status(cursor)
        known_status = LoadMySQL.load_known_status_ids(cursor)
        print("watching_status upserted:", ws_loaded, "known statuses:", sorted(known_status))

        # users from users.csv (if exist)
        u_from_users = LoadMySQL.insert_users_from_users_csv(cursor)
        print("users from users.csv inserted/ignored:", u_from_users)

        # ensure users from retings sourse + load facts
        if os.path.exists(LoadMySQL.animelist_csv):
            added = LoadMySQL.ensure_users_for_source(cursor, LoadMySQL.animelist_csv)
            print("ensured users from animelist.csv:", added)
            n = LoadMySQL.load_ratings_from_animelist(cursor, known_anime, known_status)
            print("animelist rows upserted:", n)
        elif os.path.exists(LoadMySQL.rating_complete_csv):
            added = LoadMySQL.ensure_users_for_source(cursor, LoadMySQL.rating_complete_csv)
            print("ensured users from rating_complete.csv:", added)
            n = LoadMySQL.load_ratings_from_rating_complete(cursor, known_anime)
            print("rating_complete rows upserted:", n)
        else:
            print("No ratings file found (expected animelist.csv or rating_complete.csv).")

        connection.commit()
        cursor.close(); connection.close()
        print("Done.")