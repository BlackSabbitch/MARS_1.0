import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pymongo import MongoClient


class ConcurrencyRatingTest:

    def __init__(self, db_connection):
        """
        db_connection = instance of DatabaseConnectionManager
        """
        self.db = db_connection


    # ---------------------------------------------------------
    # Find anime with zero ratings
    # ---------------------------------------------------------
    def find_unrated_anime(self):
        conn = self.db.new_mysql_connection()
        if conn is None:
            return None, None

        cur = conn.cursor()
        cur.execute("""
            SELECT a.MAL_ID, a.name
            FROM anime a
            LEFT JOIN anime_user_rating r ON r.MAL_ID = a.MAL_ID
            WHERE r.MAL_ID IS NULL
            LIMIT 1;
        """)
        row = cur.fetchone()

        cur.close()
        conn.close()

        if row:
            return row[0], row[1]
        return None, None


    # ---------------------------------------------------------
    # NON-TRANSACTIONAL user action (race-conditions allowed)
    # ---------------------------------------------------------
    def user_action(self, mal_id, user_id):
        print(f"\n--- USER {user_id} START ---")

        # Each user gets a NEW MySQL connection (thread-safe)
        conn = self.db.new_mysql_connection()
        if conn is None:
            print(f"[{user_id}] ERROR: MySQL Connection not available.")
            return {"user": user_id, "error": "No MySQL connection"}

        cur = conn.cursor()

        try:
            # ------------------ READ BEFORE ------------------
            cur.execute("SELECT score FROM anime_statistics WHERE MAL_ID=%s", (mal_id,))
            row = cur.fetchone()
            before_score = float(row[0]) if row and row[0] is not None else 0.0
            print(f"[{user_id}] READ BEFORE -> {before_score}")

            # Force overlap between threads
            time.sleep(random.uniform(0.10, 0.40))

            # ------------------ CALCULATE NEW VALUE ------------------
            # This is exactly how race conditions happen:
            # each thread computes based on stale BEFORE value.
            increment = random.randint(1, 5)
            new_score = before_score + increment
            print(
                f"[{user_id}] ATTEMPT WRITE -> "
                f"new_score={new_score}  (based on BEFORE={before_score}, +{increment})"
            )

            # ------------------ WRITE ------------------
            cur.execute("""
                UPDATE anime_statistics
                SET score = %s
                WHERE MAL_ID=%s
            """, (new_score, mal_id))
            conn.commit()

            # Force more overlap
            time.sleep(random.uniform(0.10, 0.30))

            # ------------------ READ AFTER ------------------
            cur.execute("SELECT score FROM anime_statistics WHERE MAL_ID=%s", (mal_id,))
            after_score = float(cur.fetchone()[0])
            print(f"[{user_id}] READ AFTER -> {after_score}")

            # ------------------ MONGO UPDATE (not part of race, just replicated) ------------------
            col = self.db.get_mongo_collection("animes")
            col.update_one({"_id": mal_id}, {"$set": {"score": after_score}}, upsert=True)

            print(f"--- USER {user_id} DONE ---")

            return {
                "user": user_id,
                "before": before_score,
                "attempt": new_score,
                "after": after_score,
                "error": None
            }

        except Exception as e:
            print(f"[{user_id}] ERROR: {e}")
            return {"user": user_id, "error": str(e)}

        finally:
            cur.close()
            conn.close()


    # ---------------------------------------------------------
    # STRICT ACID TRANSACTION user operation
    # ---------------------------------------------------------
    def user_rate(self, mal_id, user_id):
        print(f"\n=== TX-THREAD {user_id} START ===")

        conn = self.db.new_mysql_connection()
        if conn is None:
            print(f"[{user_id}] ERROR: MySQL Connection not available.")
            return {"user": user_id, "error": "No MySQL connection"}

        cur = conn.cursor()

        try:
            # -----------------------------------------------------------
            # Begin transaction
            # -----------------------------------------------------------
            cur.execute("START TRANSACTION;")
            print(f"[{user_id}] BEGIN TX")

            # -----------------------------------------------------------
            # Attempt to lock row (FOR UPDATE)
            # This line BLOCKS if another thread holds the lock.
            # -----------------------------------------------------------
            t0 = time.time()
            print(f"[{user_id}] TRYING TO ACQUIRE LOCK on MAL_ID={mal_id}...")

            cur.execute(
                "SELECT score FROM anime_statistics WHERE MAL_ID=%s FOR UPDATE",
                (mal_id,)
            )

            lock_wait = time.time() - t0
            if lock_wait > 0.05:
                print(f"[{user_id}] LOCK ACQUIRED after waiting {lock_wait:.3f}s "
                    "(another TX held the lock)")
            else:
                print(f"[{user_id}] LOCK ACQUIRED immediately (no contention)")

            # Now inside critical section — race-free zone.
            row = cur.fetchone()
            before_sql = float(row[0]) if row and row[0] is not None else 0.0
            print(f"[{user_id}] SQL BEFORE -> {before_sql}")

            # -----------------------------------------------------------
            # Write rating (still inside exclusive lock)
            # -----------------------------------------------------------
            rating = random.randint(1, 10)
            print(f"[{user_id}] SQL WRITE rating={rating}")

            cur.execute("""
                INSERT INTO anime_user_rating (MAL_ID, userID, user_rating)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE user_rating = VALUES(user_rating)
            """, (mal_id, user_id, rating))

            # Recalculate average
            cur.execute("SELECT AVG(user_rating) FROM anime_user_rating WHERE MAL_ID=%s", (mal_id,))
            avg_score = float(cur.fetchone()[0])

            print(f"[{user_id}] NEW AVG CALCULATED -> {avg_score}")

            # Update statistics
            cur.execute(
                "UPDATE anime_statistics SET score=%s WHERE MAL_ID=%s",
                (avg_score, mal_id)
            )

            # -----------------------------------------------------------
            # Commit (releases lock)
            # -----------------------------------------------------------
            conn.commit()
            print(f"[{user_id}] SQL COMMIT - score={avg_score}")

            # -----------------------------------------------------------
            # Verification read (guaranteed consistent)
            # -----------------------------------------------------------
            cur.execute("SELECT score FROM anime_statistics WHERE MAL_ID=%s", (mal_id,))
            confirmed = float(cur.fetchone()[0])
            print(f"[{user_id}] SQL AFTER COMMIT -> {confirmed}")

            if confirmed != avg_score:
                print(f"[{user_id}] WARNING: MISMATCH after commit (should never happen)")
            else:
                print(f"[{user_id}] CONFIRMED CONSISTENT WRITE (no race)")

            # -----------------------------------------------------------
            # Mongo update AFTER commit (safe, non-locking)
            # -----------------------------------------------------------
            col = self.db.get_mongo_collection("animes")
            col.update_one({"_id": mal_id}, {"$set": {"score": avg_score}}, upsert=True)

            print(f"[{user_id}] MONGO UPDATED -> {avg_score}")
            print(f"=== TX-THREAD {user_id} DONE ===")

            return {
                "user": user_id,
                "rating": rating,
                "score": avg_score,
                "wait_time": lock_wait,
                "error": None
            }

        except Exception as e:
            print(f"[{user_id}] ERROR -> {e}")
            return {"user": user_id, "error": str(e)}

        finally:
            cur.close()
            conn.close()

    # ---------------------------------------------------------
    # Run NON-TRANSACTIONAL simulation, concurrent users, race condition
    # ---------------------------------------------------------
    def run_user_simulation(self, mal_id, num_users=5):
        print("\n==============================")
        print("      USER SIMULATION TEST")
        print("==============================")
        print(f"Anime MAL_ID = {mal_id}")
        print(f"Users = {num_users}\n")

        # -------------------------------
        # Generate unique user IDs
        # -------------------------------
        user_ids = [300000 + i for i in range(num_users)]

        print("Launching threads...\n")

        # -------------------------------
        # Run threads concurrently
        # -------------------------------
        results = []
        with ThreadPoolExecutor(max_workers=num_users) as executor:
            futures = [
                executor.submit(self.user_action, mal_id, uid)
                for uid in user_ids
            ]

            for f in as_completed(futures):
                result = f.result()
                results.append(result)

        # -------------------------------
        # FINAL SCORE FROM DATABASE
        # -------------------------------
        conn = self.db.new_mysql_connection()
        cur = conn.cursor()
        cur.execute("SELECT score FROM anime_statistics WHERE MAL_ID=%s", (mal_id,))
        final_score = cur.fetchone()[0]
        cur.close()
        conn.close()

        print("\n==============================")
        print("           SUMMARY")
        print("==============================\n")

        # Sort results by user for consistent reading
        results_sorted = sorted(results, key=lambda r: r["user"])

        for r in results_sorted:
            if r.get("error"):
                print(f"[User {r['user']}] ERROR -> {r['error']}")
                continue

            before = r["before"]
            attempt = r["attempt"]
            after = r["after"]

            print(
                f"User {r['user']:>6} | "
                f"BEFORE={before:>5} | "
                f"ATTEMPT={attempt:>5} | "
                f"AFTER={after:>5}"
            )

        # -------------------------------
        # DETECT RACE CONDITIONS
        # -------------------------------
        print("\n==============================")
        print("     RACE CONDITION ANALYSIS")
        print("==============================\n")

        # Collect all attempted writes
        attempts = [r["attempt"] for r in results_sorted if r.get("attempt") is not None]

        if len(set(attempts)) == 1:
            print("No visible race condition: all users attempted same value.")
        else:
            print("Race conditions DETECTED:")
            print("- Multiple users read SAME initial value (stale read).")
            print("- Writes based on old value overlapped.")
            print("- Some updates may be LOST.")

        # Identify lost updates
        print("\nLost update analysis:")
        for r in results_sorted:
            if r.get("attempt") != final_score:
                print(
                    f"User {r['user']} wrote {r['attempt']} "
                    f"but final score is {final_score} -> LOST UPDATE"
                )
            else:
                print(
                    f"✔ User {r['user']} wrote {r['attempt']} "
                    f"which MATCHES final score -> LAST WRITER WINS"
                )

        print("\n==============================")
        print(f"FINAL DATABASE SCORE = {final_score}")
        print("==============================\n")

        return final_score, results_sorted


    # ---------------------------------------------------------
    # Run STRICT ACID TRANSACTION simulation
    # ---------------------------------------------------------
    def run_transactional_simulation(self, mal_id, num_users=5):
        print("\n=== STRICT TRANSACTIONAL USER SIMULATION ===")
        print(f"MAL_ID: {mal_id}")
        print(f"Users: {num_users}")

        user_ids = [200000 + i for i in range(num_users)]
        results = []

        with ThreadPoolExecutor(max_workers=num_users) as exec:
            futures = [exec.submit(self.user_rate, mal_id, uid) for uid in user_ids]

            for f in as_completed(futures):
                results.append(f.result())

        print("\n=== STRICT TX SUMMARY ===")
        for r in results:
            print(r)

        return results


    # ---------------------------------------------------------
    # Clean up after tests
    # ---------------------------------------------------------
    def reverse_all_changes(self, mal_id):
        print("\n==============================")
        print(" REVERSING ALL TEST CHANGES")
        print("==============================")

        print(f"\nTarget MAL_ID = {mal_id}")

        conn = self.db.new_mysql_connection()
        if conn is None:
            print("MySQL connection unavailable.")
            return

        cur = conn.cursor()

        # Delete test ratings
        print("\nRemoving test ratings (userID >= 200000)...")
        cur.execute("""
            DELETE FROM anime_user_rating
            WHERE MAL_ID=%s AND userID >= 200000
        """, (mal_id,))
        deleted = cur.rowcount
        conn.commit()
        print(f"Deleted {deleted} test ratings")

        # Recompute real score
        print("\nRecomputing real MySQL score...")
        cur.execute("""
            SELECT AVG(user_rating)
            FROM anime_user_rating
            WHERE MAL_ID=%s
        """, (mal_id,))
        row = cur.fetchone()
        real_score = float(row[0]) if row and row[0] is not None else None
        print(f"Real average score = {real_score}")

        cur.execute("""
            UPDATE anime_statistics SET score=%s WHERE MAL_ID=%s
        """, (real_score, mal_id))
        conn.commit()
        print("anime_statistics updated")

        cur.close()
        conn.close()

        # Update MongoDB
        print("\nUpdating MongoDB...")
        col = self.db.get_mongo_collection("animes")

        if real_score is None:
            col.update_one({"_id": mal_id}, {"$unset": {"score": ""}})
            print("No real score -> removing MongoDB score field")
        else:
            col.update_one({"_id": mal_id}, {"$set": {"score": real_score}})
            print("MongoDB updated")

        print("\n=== REVERSAL COMPLETE ===")
