import random
import time
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from mysql.connector import Error as MySQLError

# Lock for console output
print_lock = threading.Lock()

def synchronized_print(msg):
    """Prints a message ensuring no other thread interjects."""
    with print_lock:
        print(msg)

class ConcurrencyRatingTest:

    def __init__(self, db_connection):
        self.db = db_connection
        self.TARGET_ANIME_ID = 1 

    # Helper: finda anime suitable for test
    def find_zero_rated_anime(self):
        print("Searching for an existing Anime with MEMBERS=0...")
        conn = self.db.new_mysql_connection()
        cur = conn.cursor(dictionary=True)
        
        try:
            cur.execute("""
                SELECT s.MAL_ID, a.name 
                FROM anime_statistics s
                JOIN anime a ON a.MAL_ID = s.MAL_ID
                WHERE s.members = 0 
                LIMIT 1
            """)
            target = cur.fetchone()
            
            if target:
                self.TARGET_ANIME_ID = target['MAL_ID']
                print(f"Test setup: found not rated anime: '{target['name']}' (ID: {self.TARGET_ANIME_ID})")
            else:
                self.TARGET_ANIME_ID = 1 
                print(f"Warning: No clean anime found. Using ID {self.TARGET_ANIME_ID}.")
            
            return self.TARGET_ANIME_ID

        except Exception as e:
            print(f"Test setup: error{e}")
            return None
        finally:
            cur.close()
            conn.close()

    def setup_environment(self, num_users):
        target_id = self.find_zero_rated_anime()
        if not target_id: return [], None, None
        self.TARGET_ANIME_ID = target_id
        
        print(f"Test setup: Preparing {num_users} Temp Users for Anime ID {self.TARGET_ANIME_ID}...")
        conn = self.db.new_mysql_connection()
        cur = conn.cursor(dictionary=True, buffered=True) 

        try:
            # Snapshot Original State
            cur.execute("SELECT members, score FROM anime_statistics WHERE MAL_ID=%s", (self.TARGET_ANIME_ID,))
            original_sql_data = cur.fetchone() or {'members': 0, 'score': 0.0}

            # RESET STATS TO ZERO 
            cur.execute("UPDATE anime_statistics SET members=0, score=0.0 WHERE MAL_ID=%s", (self.TARGET_ANIME_ID,))
            
            # Reset Mongo
            col = self.db.get_mongo_collection("animes")
            original_mongo = col.find_one({"_id": self.TARGET_ANIME_ID}, {"members": 1, "score": 1})
            col.update_one({"_id": self.TARGET_ANIME_ID}, {"$set": {"members": 0, "score": 0.0}})

            # Create Temp Users
            base_id = random.randint(8800000, 9000000) 
            temp_user_ids = []
            for i in range(num_users):
                uid = base_id + i
                cur.execute("INSERT IGNORE INTO users (userID) VALUES (%s)", (uid,))
                temp_user_ids.append(uid)
            
            conn.commit()
            return temp_user_ids, original_sql_data, original_mongo

        except Exception as e:
            print(f"Test setup: error {e}")
            conn.rollback()
            return [], None, None
        finally:
            cur.close()
            conn.close()

    def get_current_state(self, temp_users):
        conn = self.db.new_mysql_connection()
        cur = conn.cursor(dictionary=True, buffered=True) 
        
        cur.execute("SELECT members, score FROM anime_statistics WHERE MAL_ID=%s", (self.TARGET_ANIME_ID,))
        stats = cur.fetchone() or {'members': 0, 'score': 0.0}

        fmt = ','.join(['%s'] * len(temp_users))
        cur.execute(f"SELECT COUNT(*) as cnt FROM anime_user_rating WHERE MAL_ID=%s AND userID IN ({fmt})", 
                    (self.TARGET_ANIME_ID, *temp_users))
        real_rows = int(cur.fetchone()['cnt'])
        
        cur.close()
        conn.close()
        return stats, real_rows

    def cleanup(self, temp_users, original_sql, original_mongo):
        print(f"\nRollback: Restoring database...")
        conn = self.db.new_mysql_connection()
        cur = conn.cursor()
        try:
            # 1. Delete Temp Ratings
            if temp_users:
                fmt = ','.join(['%s'] * len(temp_users))
                cur.execute(f"DELETE FROM anime_user_rating WHERE MAL_ID=%s AND userID IN ({fmt})", 
                            (self.TARGET_ANIME_ID, *temp_users))
                
                # 2. Delete Temp Users
                cur.execute(f"DELETE FROM users WHERE userID IN ({fmt})", tuple(temp_users))

            # 3. Restore MySQL Stats (original data)
            if original_sql:
                cur.execute("UPDATE anime_statistics SET members=%s, score=%s WHERE MAL_ID=%s", 
                            (original_sql['members'], original_sql['score'], self.TARGET_ANIME_ID))
            
            # 4. Restore Mongo Stats
            if original_mongo:
                col = self.db.get_mongo_collection("animes")
                col.update_one({"_id": self.TARGET_ANIME_ID}, {"$set": {
                    "members": original_mongo.get('members', 0),
                    "score": original_mongo.get('score', 0.0)
                }})
            
            conn.commit()
            print("Rollback: Success. Original data restored.")
        except Exception as e:
            print(f"Rollback: error {e}")
        finally:
            cur.close()
            conn.close()



    # SCENARIO 1: Unsafe operation (Lost Updates, race condition)
    def user_action_unsafe(self, mal_id, user_id):
        start_time = time.perf_counter()
        conn = self.db.new_mysql_connection()
        cur = conn.cursor(buffered=True)

        try:
            # 1. Decide Rating
            my_rating = random.randint(1, 10) 
            
            # 2. Insert Rating Row
            cur.execute("INSERT INTO anime_user_rating (MAL_ID, userID, user_rating) VALUES (%s, %s, %s)", 
                        (mal_id, user_id, my_rating))
            
            # 3. Dirty read
            cur.execute("SELECT members, score FROM anime_statistics WHERE MAL_ID=%s", (mal_id,))
            row = cur.fetchone()
            curr_members = int(row[0])
            curr_score = float(row[1])
            
            # Log the STALE READ and INTENT
            synchronized_print(f"[Unsafe] User {user_id} READ: M={curr_members}, S={curr_score:.2f}. Submitting RATING {my_rating}. Writes {curr_members + 1}")
            
            # Delay to force overlap
            time.sleep(random.uniform(0.1, 0.2)) 

            # 4. Naive score calculations 
            new_members = curr_members + 1
            if new_members > 0:
                total_sum = (curr_score * curr_members) + my_rating
                new_score = total_sum / new_members
            else:
                new_score = float(my_rating)

            # 5. Write anime statistics
            cur.execute("UPDATE anime_statistics SET members=%s, score=%s WHERE MAL_ID=%s", 
                        (new_members, new_score, mal_id))
            conn.commit()

            # 6. Mongo
            self.db.get_mongo_collection("animes").update_one({"_id": mal_id}, 
                {"$set": {"members": new_members, "score": new_score}})

            return {"duration": time.perf_counter() - start_time, "error": None}
        except Exception as e:
            # Capture errors async
            synchronized_print(f"[Unsafe] User {user_id} FAILED: {e}") 
            return {"error": str(e)}
        finally:
            cur.close()
            conn.close()

    # SCENARIO 2: Transactional (Safe) - optimistic locking (CAS)
    def user_action_safe(self, mal_id, user_id):
        MAX_RETRIES = 10      
        RETRY_DELAY = 2.0     
        
        start_time = time.perf_counter()
        
        # Initial Stagger
        time.sleep(random.uniform(0.01, 0.1)) 
        
        for attempt in range(MAX_RETRIES):
            conn = self.db.new_mysql_connection()
            cur = conn.cursor(dictionary=True, buffered=True)

            try:
                conn.start_transaction()
                
                my_rating = random.randint(1, 10)  
                
                # 1. Log the intent
                if attempt == 0:
                    synchronized_print(f"[Transact] User {user_id} STARTING attempt 1 to insert rating {my_rating}...")
                else:
                    synchronized_print(f"[Transact] User {user_id} RETRY {attempt+1}/{MAX_RETRIES}. Rerunning CAS...")

                # 2. READ (Non-locked read for CAS)
                cur.execute("SELECT members, score FROM anime_statistics WHERE MAL_ID=%s", (mal_id,))
                stats = cur.fetchone()
                curr_members = int(stats['members'])
                curr_score = float(stats['score'])
                
                # 3. INSERT new rating record
                cur.execute("INSERT INTO anime_user_rating (MAL_ID, userID, user_rating) VALUES (%s, %s, %s)", 
                            (mal_id, user_id, my_rating))

                # 4. CALCULATION
                new_members = curr_members + 1
                total_sum = (curr_score * curr_members) + my_rating
                real_avg = total_sum / new_members if new_members > 0 else float(my_rating)

                # 5. OPTIMISTIC UPDATE (Compare-and-Swap - The SOFT LOCK check)
                cur.execute("""
                    UPDATE anime_statistics 
                    SET members=%s, score=%s 
                    WHERE MAL_ID=%s AND members=%s
                """, (new_members, real_avg, mal_id, curr_members))
                
                # 6. CHECK SUCCESS (rows affected)
                if cur.rowcount == 1:
                    # Successful CAS: Log Soft Lock acquisition and commit
                    conn.commit()
                    
                    synchronized_print(f"[Transact] User {user_id} COMMITTED (SUCCESSFUL CAS). New Stats: Members={new_members}, Score={real_avg:.2f}")
                    
                    # 7. Mongo
                    self.db.get_mongo_collection("animes").update_one({"_id": mal_id}, 
                        {"$set": {"members": new_members, "score": real_avg}})

                    # Success: Exit the retry loop
                    return {"duration": time.perf_counter() - start_time, "error": None}
                
                else:
                    # Failed CAS: Row was changed by another thread. Rollback and retry.
                    conn.rollback()
                    synchronized_print(f"[Transact] User {user_id} FAILED CAS (Data changed). Rolling back and retrying...") 
                    time.sleep(RETRY_DELAY)
                    continue # Go to the next attempt

            except Exception as e:
                # Handle general errors (e.g., connection fail)
                conn.rollback()
                synchronized_print(f"[Transact] User {user_id} CRITICAL FAILURE: {e}")
                return {"error": str(e)}

            finally:
                cur.close()
                conn.close()
        
        # Fallback if all retries fail
        return {"error": "Max retries failed."}

    # Test runner
    def run_benchmark(self, mal_id=None, num_users=5, mode="unsafe"):

        print(f"Test start: {mode.upper()} simulating {num_users} users rates")
        print(f"{'='*80}")
        
        # 1. Setup
        temp_users, orig_sql, orig_mongo = self.setup_environment(num_users)
        if not temp_users: return

        # 2. Initial state
        start_stats, start_rows = self.get_current_state(temp_users)
        print(f"Initial anime state:")
        print(f"  - Score:       {start_stats['score']:.2f} (Reset for test)")
        print(f"  - Members:     {start_stats['members']}")
        print(f"  - Real Rows:   {start_rows}")
        print("-" * 80)

        target_func = self.user_action_unsafe if mode == "unsafe" else self.user_action_safe

        # 3. Run tests
        results = []
        concurrency_level = num_users
        
        # Unsafe scenario
        if mode == "unsafe":
            results = self._execute_concurrency_test(self.TARGET_ANIME_ID, temp_users, target_func, concurrency_level)
            
        # Safe scenario (CAS)
        elif mode == "safe":
            results = self._execute_concurrency_test(self.TARGET_ANIME_ID, temp_users, target_func, concurrency_level)


        # 4. Final state
        print("-" * 80)
        end_stats, end_rows = self.get_current_state(temp_users)
        
        # Analysis
        lost_members = end_rows - end_stats['members']
        
        print(f"Final anime state:")
        print(f"  - Score:       {end_stats['score']:.2f}")
        print(f"  - Members:     {end_stats['members']} (Final Count)")
        print(f"  - Real Rows:   {end_rows} (Actual Inserts)")

        # 5. Summary
        print("\nSummary:")
        
        # Calculate Response Times
        success_ops = [r for r in results if not r.get("error")]
        durations = [r["duration"] for r in success_ops]
        avg_response_time = statistics.mean(durations) if durations else 0

        print(f"Avg Response Time: {avg_response_time:.4f} sec")
        
        errors = [r for r in results if r.get("error")]
        
        if mode == "unsafe":
            if lost_members > 0:
                print(f"Data loss: {lost_members} votes count lost.")
            elif end_rows > 0:
                print("No data loss observed (Lucky run).")
            else:
                print("ALL OPERATIONS FAILED.")
        else:
            if lost_members == 0 and end_rows == num_users:
                print("Integrity OK: Counter matches Inserts exactly.")
            elif len(errors) > 0:
                print(f"Contention: {len(errors)} transactions failed due to deadlock.")
            else:
                print("Unknown error accured.")

        # 6. Rollback
        self.cleanup(temp_users, orig_sql, orig_mongo)
        return results

    def _execute_concurrency_test(self, mal_id, temp_users, target_func, concurrency_level):
        """Helper to run the thread pool."""
        results = []
        with ThreadPoolExecutor(max_workers=concurrency_level) as executor:
            futures = [executor.submit(target_func, mal_id, uid) for uid in temp_users]
            for f in as_completed(futures):
                results.append(f.result())
        return results

