import time
import random
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from mysql.connector import Error as MySQLError
from pymongo.errors import PyMongoError

from db_tools.mySQL_tools import MySQLTools
from db_tools.mongoDB_tools import MongoTools


class ConcurrencyTest:

    @staticmethod
    def mysql_read_op():
        mysql = MySQLTools()
        conn = mysql.connection

        query = """
            SELECT MAL_ID, name, score
            FROM anime_statistics
            JOIN anime USING (MAL_ID)
            ORDER BY score DESC LIMIT 20;
        """

        start = time.time()
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(query)
            rows = cursor.fetchall()
            cursor.close()
            mysql.close()
            return ("mysql_read", True, (time.time() - start)*1000, None)
        except MySQLError as e:
            mysql.close()
            return ("mysql_read", False, (time.time() - start)*1000, str(e))


    @staticmethod
    def mysql_write_op():
        mysql = MySQLTools()
        conn = mysql.connection

        user_id = random.randint(1, 1000)
        anime_id = random.randint(1, 20000)
        rating = random.randint(1, 10)

        query = """
            INSERT INTO anime_user_rating (MAL_ID, userID, user_rating)
            VALUES (%s,%s,%s)
            ON DUPLICATE KEY UPDATE user_rating = VALUES(user_rating);
        """

        start = time.time()
        try:
            cursor = conn.cursor()
            cursor.execute(query, (anime_id, user_id, rating))
            conn.commit()
            cursor.close()
            mysql.close()
            return ("mysql_write", True, (time.time() - start)*1000, None)
        except MySQLError as e:
            mysql.close()
            return ("mysql_write", False, (time.time() - start)*1000, str(e))


    @staticmethod
    def mongo_read_op():
        mongo = MongoTools(
            uri="mongodb+srv://msamosudova:Duckling@mars-cluster.8ruotdw.mongodb.net/?appName=MARS-Cluster",
            db_name="anime_db",
            collection="animes"
        )
        coll = mongo.get_collection("animes")

        start = time.time()
        try:
            doc = coll.find_one({}, {"synopsis": 1})
            return ("mongo_read", True, (time.time() - start)*1000, None)
        except PyMongoError as e:
            return ("mongo_read", False, (time.time() - start)*1000, str(e))


    @staticmethod
    def mongo_write_op():
        mongo = MongoTools(
            uri="mongodb+srv://msamosudova:Duckling@mars-cluster.8ruotdw.mongodb.net/?appName=MARS-Cluster",
            db_name="anime_db",
            collection="animes"
        )
        coll = mongo.get_collection("animes")

        test_id = random.randint(1, 1000)

        start = time.time()
        try:
            coll.update_one(
                {"_id": test_id},
                {"$set": {"load_test_value": random.random()}},
                upsert=True
            )
            return ("mongo_write", True, (time.time() - start)*1000, None)
        except PyMongoError as e:
            return ("mongo_write", False, (time.time() - start)*1000, str(e))


    # -------------------------------
    # MAIN RUN LOOP (THREAD SAFE)
    # -------------------------------
    @staticmethod
    def run(n_threads=4, iterations=20):

        print(f"\n=== Running concurrency test with {n_threads} threads, {iterations} iterations ===")

        ops = [
            ConcurrencyTest.mysql_read_op,
            ConcurrencyTest.mysql_write_op,
            ConcurrencyTest.mongo_read_op,
            ConcurrencyTest.mongo_write_op
        ]

        results = []

        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = []
            for _ in range(iterations):
                op = random.choice(ops)
                futures.append(executor.submit(op))

            for future in as_completed(futures):
                op_name, success, time_ms, error = future.result()
                results.append((op_name, success, time_ms, error))
                print(f"{op_name:12s} | success={success} | time={round(time_ms,2)} ms | error={error}")

        # Convert to DataFrame
        df = pd.DataFrame(results, columns=["operation", "success", "time_ms", "error"])

        # PRINT SUMMARY ALWAYS
        print("\n=== SUMMARY (per operation) ===")
        for op in df["operation"].unique():
            df_op = df[df["operation"] == op]
            print(f"\nOperation: {op}")
            print(f"  Count:     {len(df_op)}")
            print(f"  Success:   {df_op['success'].sum()}")
            print(f"  Fail:      {len(df_op) - df_op['success'].sum()}")
            print(f"  Avg time:  {round(df_op['time_ms'].mean(),2)} ms")
            print(f"  Min time:  {round(df_op['time_ms'].min(),2)} ms")
            print(f"  Max time:  {round(df_op['time_ms'].max(),2)} ms")

        return df
