import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import mysql.connector
from pymongo import MongoClient


class ConcurrencyRatingTest:

    def __init__(self, mysql_conf, mongo_uri):
        self.mysql_conf = mysql_conf
        self.mongo_uri = mongo_uri


    # --------------------------------
    # MySQL helper
    # --------------------------------
    def mysql_connect(self):
        return mysql.connector.connect(
            host=self.mysql_conf["host"],
            user=self.mysql_conf["user"],
            password=self.mysql_conf["password"],
            database=self.mysql_conf["database"]
        )


    # --------------------------------
    # Find anime with 0 ratings
    # --------------------------------
    def find_unrated_anime(self):
        conn = self.mysql_connect()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT a.MAL_ID, a.name
            FROM anime a
            LEFT JOIN anime_user_rating r ON r.MAL_ID = a.MAL_ID
            WHERE r.MAL_ID IS NULL
            LIMIT 1;
        """)

        row = cursor.fetchone()
        cursor.close()
        conn.close()

        if not row:
            return None, None
        return row[0], row[1]


    # --------------------------------
    # Get initial score (MySQL)
    # --------------------------------
    def get_initial_scores(self, mal_id):
        conn = self.mysql_connect()
        cursor = conn.cursor()

        cursor.execute("SELECT score FROM anime_statistics WHERE MAL_ID = %s", (mal_id,))
        row = cursor.fetchone()
        initial_score = row[0] if row else None

        cursor.execute("""
            SELECT AVG(user_rating)
            FROM anime_user_rating
            WHERE MAL_ID = %s
        """, (mal_id,))
        row = cursor.fetchone()
        initial_avg_rating = float(row[0]) if row[0] is not None else 0.0

        cursor.close()
        conn.close()
        return initial_score, initial_avg_rating


    # --------------------------------
    # Worker thread: user updates rating
    # --------------------------------
    def user_rate(self, mal_id, user_id):
        """
        One user/thread: read, write
        """
        print(f"\n=== THREAD {user_id} START ===")

        try:
            # --------------------- CONNECT ---------------------
            sql = self.mysql_connect()
            cur = sql.cursor()

            mongo = MongoClient(self.mongo_uri)
            col = mongo["anime_db"]["animes"]

            # --------------------- SQL READ #1 ---------------------
            cur.execute("SELECT score FROM anime_statistics WHERE MAL_ID=%s", (mal_id,))
            before_sql = cur.fetchone()[0]
            print(f"[{user_id}] SQL FIRST READ → score={before_sql}")

            time.sleep(random.uniform(0.05, 0.2))

            # --------------------- SQL WRITE ---------------------
            rating = random.randint(1, 10)
            print(f"[{user_id}] SQL WRITE rating={rating}")

            cur.execute("""
                INSERT INTO anime_user_rating (MAL_ID, userID, user_rating)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE user_rating = VALUES(user_rating)
            """, (mal_id, user_id, rating))

            # Recalculate average rating for this anime
            cur.execute("SELECT AVG(user_rating) FROM anime_user_rating WHERE MAL_ID=%s", (mal_id,))
            avg_score = float(cur.fetchone()[0])

            cur.execute("UPDATE anime_statistics SET score=%s WHERE MAL_ID=%s", (avg_score, mal_id))

            # ----------------Commit to have correct data for other users/threads --------------------
            sql.commit()
            print(f"[{user_id}] SQL UPDATED → new average score={avg_score}")

            time.sleep(random.uniform(0.05, 0.2))

            # --------------------- SQL READ AFTER UPDATE ---------------------
            cur.execute("SELECT score FROM anime_statistics WHERE MAL_ID=%s", (mal_id,))
            confirmed_sql = float(cur.fetchone()[0])
            print(f"[{user_id}] SQL CONFIRMED READ → score={confirmed_sql}")

            # --------------------- MONGO READ BEFORE ---------------------
            doc_before = col.find_one({"_id": mal_id}, {"score": 1})
            mongo_before = doc_before.get("score") if doc_before else None
            print(f"[{user_id}] MONGO FIRST READ → score={mongo_before}")

            time.sleep(random.uniform(0.05, 0.2))

            # --------------------- MONGO WRITE ---------------------
            col.update_one(
                {"_id": mal_id},
                {"$set": {"score": confirmed_sql}},  # average from mySQL db
                upsert=True
            )
            print(f"[{user_id}] MONGO UPDATED → score={confirmed_sql}")

            time.sleep(random.uniform(0.05, 0.2))

            # --------------------- MONGO READ AFTER ---------------------
            doc_after = col.find_one({"_id": mal_id}, {"score": 1})
            mongo_after = doc_after.get("score")
            print(f"[{user_id}] MONGO FINAL READ → score={mongo_after}")

            # --------------------- CLEANUP ---------------------
            cur.close()
            sql.close()

            print(f"=== THREAD {user_id} DONE ===")

            return {
                "user": user_id,
                "rating": rating,
                "sql_score": confirmed_sql,
                "mongo_score": mongo_after,
                "error": None
            }

        except Exception as e:
            print(f"[{user_id}] ERROR → {e}")
            return {
                "user": user_id,
                "rating": None,
                "sql_score": None,
                "mongo_score": None,
                "error": str(e)
            }



    # --------------------------------
    # MAIN TEST
    # --------------------------------
    def run(self, threads=10, ratings=50):

        print("\n=== Concurrency Rating Test ===")

        # 1. Find anime with zero votes
        mal_id, name = self.find_unrated_anime()
        if not mal_id:
            print("No unrated anime found.")
            return

        print(f"\nSelected anime for test:")
        print(f"  MAL_ID: {mal_id}")
        print(f"  Name:   {name}")

        # 2. Print initial values
        init_score, init_avg = self.get_initial_scores(mal_id)
        print("\nInitial MySQL state:")
        print(f"  anime_statistics.score = {init_score}")
        print(f"  average user rating    = {init_avg}")

        print(f"\nLaunching {ratings} concurrent threads...\n")

        # 3. Start users/threads
        results = []
        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [
                executor.submit(self.user_rate, mal_id, 200000 + i)
                for i in range(ratings)
            ]

            for f in as_completed(futures):
                res = f.result()
                results.append(res)

        # 4. Final MySQL score
        conn = self.mysql_connect()
        cursor = conn.cursor()
        cursor.execute("SELECT score FROM anime_statistics WHERE MAL_ID=%s", (mal_id,))
        final_mysql = cursor.fetchone()[0]
        cursor.close()
        conn.close()

        # 5. Final MongoDB score
        mongo = MongoClient(self.mongo_uri)
        doc = mongo["anime_db"]["animes"].find_one({"_id": mal_id}, {"score": 1})
        final_mongo = doc.get("score") if doc else None

        print("\n=== FINAL RESULTS ===")
        print(f"Final MySQL score:   {final_mysql}")
        print(f"Final MongoDB score: {final_mongo}")
        print(f"Total operations:    {len(results)}\n")

        return results
    
    def reverse_score(self, mal_id):
        print("\n=== REVERSING CONCURRENCY TEST CHANGES ===")

        print(f"Affected MAL_ID: {mal_id}")

        # 1. Remove test ratings (userID ≥ 200000)
        conn = self.mysql_connect()
        cursor = conn.cursor()

        print("Deleting test ratings from MySQL...")

        cursor.execute("""
            DELETE FROM anime_user_rating
            WHERE MAL_ID = %s AND userID >= 200000
        """, (mal_id,))
        deleted = cursor.rowcount

        conn.commit()

        print(f"  Deleted {deleted} test ratings.")

        # 2. Recompute real score
        print("\nRecomputing real MySQL score...")

        cursor.execute("""
            SELECT AVG(user_rating)
            FROM anime_user_rating
            WHERE MAL_ID = %s
        """, (mal_id,))
        row = cursor.fetchone()
        real_score = float(row[0]) if row[0] is not None else None

        cursor.execute("""
            UPDATE anime_statistics
            SET score = %s
            WHERE MAL_ID = %s
        """, (real_score, mal_id))
        conn.commit()

        print(f"  New real score: {real_score}")

        cursor.close()
        conn.close()

        # 3. Update MongoDB (remove score or set null)
        print("\nResetting MongoDB score...")

        mongo = MongoClient(self.mongo_uri)
        db = mongo["anime_db"]
        col = db["animes"]

        if real_score is None:
            # Remove the score field entirely
            col.update_one(
                {"_id": mal_id},
                {"$unset": {"score": ""}}
            )
            print("  Mongo: score field removed")
        else:
            # Set real score
            col.update_one(
                {"_id": mal_id},
                {"$set": {"score": real_score}}
            )
            print(f"  Mongo: score set to {real_score}")

        print("\n=== REVERSAL COMPLETE ===")
        return deleted, real_score
