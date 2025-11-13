# 01_load_anime_meta.py
# Loads anime.csv into: type, source, age_rating, genre, studio, producer, licensor,
# and fills anime, anime_statistics, plus bridges: anime_genre, anime_studio, anime_producer, anime_licensor.

import os, csv, re
import mysql.connector
from mysql.connector import Error
from mySQL_tools import MySQLTools

class LoadMySQL:

    DATA_DIR = r"C:\MariaSamosudova\Projects\UNIVER\ADB\Project\DataSet"
    ANIME_CSV   = os.path.join(DATA_DIR, "anime.csv")
    BATCH = 1000

    def __init__(self):
        data_dir = self.DATA_DIR
        batch = self.BATCH
 


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

    def insert_ignore(cursor, table, col, val):
        cursor.execute(f"INSERT IGNORE INTO {table} ({col}) VALUES (%s)", (val,))

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

    # ---------- LOAD ----------
    # load_anime
    # Loads anime.csv into: type, source, age_rating, genre, studio, producer, licensor,
    # and fills anime, anime_statistics, plus bridges: anime_genre, anime_studio, anime_producer, anime_licensor.
    def load_anime(ANIME_CSV, connection, self):
        if not os.path.exists(ANIME_CSV):
            raise FileNotFoundError(ANIME_CSV)

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
            mal_id  = self.to_int(col("MAL_ID", r))
            if not mal_id:  # skip bad rows
                continue

            name    = self.clean_str(col("Name", r))
            episodes= self.to_int(col("Episodes", r))
            aired   = self.clean_str(col("Aired", r))
            premiered = self.clean_str(col("Premiered", r))
            duration = self.clean_str(col("Duration", r))

            # lookups single-valued
            type_val   = self.clean_str(col("Type", r))
            type_id    = self.get_id(cursor, "type", "typeID", "type", type_val, type_cache) if type_val else None

            source_val = self.clean_str(col("Source", r))
            source_id  = self.get_id(cursor, "source", "sourceID", "source", source_val, source_cache) if source_val else None

            age_val    = self.clean_str(col("Rating", r)) or self.clean_str(col("age_rating", r))
            age_id     = self.get_id(cursor, "age_rating", "age_ratingID", "age_rating", age_val, age_cache) if age_val else None

            # upsert anime
            anime_batch.append((
                mal_id, name, episodes, aired, premiered, duration, source_id, type_id, age_id
            ))

            # stats (denormalized snapshot in anime_statistics)
            stats_batch.append((
                mal_id,
                self.to_float2(col("Score", r)),
                self.to_int(col("Rank", r)),
                self.to_int(col("Popularity", r)),
                self.to_int(col("Members", r)),
                self.to_int(col("Favorites", r)),
                self.to_int(col("Watching", r)),
                self.to_int(col("Completed", r)),
                self.to_int(col("On-Hold", r)) or self.to_int(col("On_Hold", r)),
                self.to_int(col("Dropped", r)),
                self.to_int(col("Plan to Watch", r)) or self.to_int(col("Plan_to_Watch", r))
            ))

            # multi-valued lists -> lookups + bridges
            for g in self.split_list(col("Genres", r)):
                gid = self.get_id(cursor, "genre", "genreID", "genre", g, genre_cache)
                gen_bridge.append((mal_id, gid))

            for st in self.split_list(col("Studios", r)):
                sid = self.get_id(cursor, "studio", "studioID", "studio", st, studio_cache)
                st_bridge.append((mal_id, sid))

            for pr in self.split_list(col("Producers", r)):
                pid = self.get_id(cursor, "producer", "producerID", "producer", pr, producer_cache)
                prod_bridge.append((mal_id, pid))

            for lc in self.split_list(col("Licensors", r)):
                lid = self.get_id(cursor, "licensor", "licensorID", "licensor", lc, licensor_cache)
                lic_bridge.append((mal_id, lid))

            if len(anime_batch) >= self.BATCH:
                cursor.executemany("""
                    INSERT INTO anime (MAL_ID, name, episodes, aired, premiered, duration, sourceID, typeID, age_ratingID)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON DUPLICATE KEY UPDATE
                    name=VALUES(name), episodes=VALUES(episodes), aired=VALUES(aired),
                    premiered=VALUES(premiered), duration=VALUES(duration),
                    sourceID=VALUES(sourceID), typeID=VALUES(typeID), age_ratingID=VALUES(age_ratingID)
                """, anime_batch)
                n_anime += len(anime_batch); anime_batch.clear()

            if len(stats_batch) >= self.BATCH:
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
            if len(gen_bridge) >= self.BATCH:
                cursor.executemany("INSERT IGNORE INTO anime_genre (MAL_ID, genreID) VALUES (%s,%s)", gen_bridge)
                n_gen += len(gen_bridge); gen_bridge.clear()

            if len(st_bridge) >= self.BATCH:
                cursor.executemany("INSERT IGNORE INTO anime_studio (MAL_ID, studioID) VALUES (%s,%s)", st_bridge)
                n_st += len(st_bridge); st_bridge.clear()

            if len(prod_bridge) >= self.BATCH:
                cursor.executemany("INSERT IGNORE INTO anime_producer (MAL_ID, producerID) VALUES (%s,%s)", prod_bridge)
                n_prod += len(prod_bridge); prod_bridge.clear()

            if len(lic_bridge) >= self.BATCH:
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

