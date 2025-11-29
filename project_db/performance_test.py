import time
import pandas as pd
from anime_recommendation import AnimeRecommendation
from mongoDB_tools import MongoTools
from concurrency_rating_test import ConcurrencyRatingTest
from database_connection_manager import DatabaseConnectionManager
from schema_manager import SchemaManager
from mysql_index_manager import MySQLIndexManager
from mongo_index_manager import MongoIndexManager


class PerformanceTest:
    """
    Full benchmark:
        - Drop MySQL + Mongo indexes
        - Run all heavy recommendation queries (5 total)
        - Recreate indexes
        - Run all queries again
        - Produce summary table with performance comparison
    """

    def __init__(self, db, mysql_index_manager: MySQLIndexManager, mongo_index_manager: MongoIndexManager):
        self.db = db
        self.mysql_idx = mysql_index_manager
        self.mongo_idx = mongo_index_manager

    # ---------------------------------------------------
    # Utility: measure time of a function call
    # ---------------------------------------------------
    def time_query(self, func, label):
        try:
            start = time.perf_counter()
            result = func()
            end = time.perf_counter()
            duration = end - start
            print(f"[{label}] finished in {duration:.3f} sec")
            return duration, result

        except Exception as e:
            print(f"[{label}] FAILED due to {e}")
            return float("inf"), None


    # ---------------------------------------------------
    # Run all recommendation queries
    # ---------------------------------------------------
    def run_sql_tests(self):

        rec = AnimeRecommendation()
        times = {}

        # 1) Portugal top30
        times["PORTUGAL_TOP30"] = self.time_query(
            lambda: rec.recommend_portugal_top30(self.db.get_mysql_connection(), self.db),
            "PORTUGAL_TOP30"
        )

        # 2) Top-100 from SQL only
        times["TOP100_SQL"] = self.time_query(
            lambda: rec.hundred_best(self.db.get_mysql_connection()),
            "TOP100_SQL"
        )

        # 3) Top-100 by user preferences
        times["TOP10_BY_USER_PREFERENCE"] = self.time_query(
            lambda: rec.recommend_by_genres(self.db.get_mysql_connection(), user_id = 21, limit = 10),
            "TOP10_BY_USER_PREFERENCE"
        )

        # 4) Top-100 with Mongo synopsis
        times["TOP100_SQL_MONGO"] = self.time_query(
            lambda: rec.hundred_best_with_synopsis(self.db.get_mysql_connection(), self.db),
            "TOP100_SQL_MONGO"
        )

        # 5) PT Adventure + Magic
        times["PT_ADV_MAGIC"] = self.time_query(
            lambda: rec.recommend_pt_adventure_magic(self.db.get_mysql_connection(), self.db),
            "PT_ADV_MAGIC"
        )

        return times

    # ---------------------------------------------------
    # RUN THE FULL BENCHMARK PIPELINE
    # ---------------------------------------------------
    def run(self):
        print("\n=====================================")
        print("      PERFORMANCE BENCHMARK START")
        print("=====================================\n")

        # ---------------------------------------------------
        # A) DROP ALL INDEXES
        # ---------------------------------------------------
        print("\n=== DROPPING ALL INDEXES (MySQL + Mongo) ===")
        self.mysql_idx.drop_all_indexes()
        self.mongo_idx.drop_all_indexes()

        # ---------------------------------------------------
        # B) RUN TEST QUERIES WITHOUT INDEXES
        # ---------------------------------------------------
        print("\n=== RUNNING TESTS WITHOUT INDEXES ===")
        no_idx_times = self.run_sql_tests()

        # ---------------------------------------------------
        # C) CREATE INDEXES
        # ---------------------------------------------------
        print("\n=== CREATING INDEXES (MySQL + Mongo) ===")
        self.mysql_idx.ensure_all_indexes()
        self.mongo_idx.ensure_all_indexes()

        # ---------------------------------------------------
        # D) RUN TESTS WITH INDEXES
        # ---------------------------------------------------
        print("\n=== RUNNING TESTS WITH INDEXES ===")
        with_idx_times = self.run_sql_tests()

        # ---------------------------------------------------
        # E) BUILD SUMMARY TABLE
        # ---------------------------------------------------
        df = pd.DataFrame({
                "NO_INDEXES": {k: v[0] for k, v in no_idx_times.items()},
                "WITH_INDEXES": {k: v[0] for k, v in with_idx_times.items()},
            })

        df["SPEEDUP"] = df["NO_INDEXES"] / df["WITH_INDEXES"]

        print("\n=====================================")
        print("       PERFORMANCE SUMMARY TABLE")
        print("=====================================")
        print(df.to_string())
        print("\n=====================================")

        return df
