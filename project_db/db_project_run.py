import re
from database_connection_manager import DatabaseConnectionManager
from mongoDB_tools import MongoTools
from anime_recommendation import AnimeRecommendation
from concurrency_rating_test import ConcurrencyRatingTest
import pandas as pd
import networkx as nx
from schema_manager import SchemaManager
from database_connection_manager import DatabaseConnectionManager
from schema_manager import SchemaManager
from mongo_index_manager import MongoIndexManager
from load_mySQL import LoadMySQL
from concurrency_rating_test import ConcurrencyRatingTest
from database_connection_manager import DatabaseConnectionManager
from performance_test import PerformanceTest
from mysql_index_manager import MySQLIndexManager
from mongoDB_tools import MongoTools
from performance_test import PerformanceTest
from mysql_index_manager import MySQLIndexManager
from mongo_index_manager import MongoIndexManager

def db_project_run():

    # Connect to db
    print("I'm here")
    #db_connection = DatabaseConnectionManager(mysql_database="mars_db")

    # # 1. Create schema
    # schema = SchemaManager(db_connection)
    # schema.create_schema()

    # # # 2. Create indexes
    # index_manager = IndexManager(db_connection)
    # index_manager.ensure_all_indexes()

    # 3. Load data
    # loader = LoadMySQL(db_connection)
    # loader.load_paths({
    #     "anime": "data/anime.csv",
    #     "animelist": "data/animelist.csv",
    #     "profiles": "data/profiles.csv",
    #     "ratings": "data/rating_complete.csv"
    # })
    
    # loader.load_watching_status()
    # loader.load_anime()
    # loader.load_genres()

    # loader.load_users_from_animelist()
    # loader.load_profiles_map()
    # loader.load_rating_complete()

    # loader.load_score_distribution(loader.read_csv("anime"))
    # loader.load_animelist()

    # Close all connections
    # db.close_all()


    # mysql.ensure_indexes()
    # print("\n=== RUNNING TOP 100 BEST ANIME ===")
    # df = AnimeRecommendation.hundred_best(connection)
    
    # print("\n=== RUNNING TOP 100 JOINED QUERY (MySQL + Mongo) ===")
    # df = AnimeRecommendation.hundred_best_with_synopsis(connection, mongo)

    # print("\n=== RUNNING TOP recommendations for specific user ===")
    # df = AnimeRecommendation.recommend_by_genres(connection, user_id=21, limit=10)

    # print("\n=== RUNNING TOP 30 BEST FOR PORTUGUESE 18+")
    # df = AnimeRecommendation.recommend_portugal_top30(connection, mongo)

    # print("\n=== Recommend top anime liked by Portuguese users (18+) specifically in genres Adventure + Magic.")
    # df = AnimeRecommendation.recommend_pt_adventure_magic(connection, mongo)

    # print("\n=== Running CONCURRENCY TEST ===")
    # df_conc = ConcurrencyTest.run(n_threads=4, iterations=20)

    # concurrency_rating_test = ConcurrencyRatingTest(
    #         mysql_conf={
    #         "host": "mars-db.cdcmiuuwqo5z.eu-north-1.rds.amazonaws.com",
    #         "user": "msamosudova",
    #         "password": "Duckling25!",
    #         "database": "mars_db"
    #         },
    #         mongo_uri="mongodb+srv://msamosudova:Duckling@mars-cluster.8ruotdw.mongodb.net/"
    #         )

    #concurrency_rating_test.run(threads=2, ratings=10)
    #concurrency_rating_test.reverse_score(mal_id= 48480)

    # cursor = connection.cursor()

    #paths = CommonTools.load_csv_files_pathes()
    #df = pd.read_csv(paths["anime"])
    # df = pd.read_csv("data/anime.csv")
    # print(df.head())
    # print(df.columns)
    # print(df.shape)
    # print(df.columns)
    
    # loader.ensure_score_distribution_columns()
    # loader.load_score_distribution(df)
    

    # connection.commit()
    # cursor.close()
    # connection.close()

    # Close connection 
    # mysql.close()

    # # ----------------------------------------
    # # CONCURRENCY TEST
    # # ----------------------------------------
    # test = ConcurrencyRatingTest(db_connection)

    # # ----------------------------------------
    # # 3. FIND ANIME WITH ZERO RATINGS
    # # ----------------------------------------
    # print("\n=== Locating anime with zero ratings ===")
    # mal_id, name = test.find_unrated_anime()

    # if not mal_id:
    #     print("No unrated anime found — cannot run tests.")
    #     return

    # print(f"Using anime MAL_ID={mal_id} Name='{name}' for simulation")

    # # ----------------------------------------
    # # 4. RUN NORMAL NON-TX TEST
    # # ----------------------------------------
    # print("\n\n==============================")
    # print("RUNNING NON-TRANSACTIONAL TEST")
    # print("==============================\n")

    # test.run_user_simulation(
    #     mal_id=mal_id,
    #     num_users=5
    # )

    # # ----------------------------------------
    # # 5. REVERSE CHANGES
    # # ----------------------------------------
    # print("\n\n==============================")
    # print("REVERSING CHANGES FROM NON-TRANSACTIONAL TEST")
    # print("==============================\n")

    # test.reverse_all_changes(mal_id)

    # # ----------------------------------------
    # # 6. RUN STRICT TRANSACTION TEST
    # # ----------------------------------------
    # print("\n\n==============================")
    # print("RUNNING STRICT ACID TRANSACTION TEST")
    # print("==============================\n")

    # test.run_transactional_simulation(
    #     mal_id=mal_id,
    #     num_users=5
    # )

    # # ----------------------------------------
    # # 7. FINAL CLEANUP
    # # ----------------------------------------
    # print("\n\n==============================")
    # print("REVERSING TRANSACTION TEST")
    # print("==============================\n")

    # test.reverse_all_changes(mal_id=48480)

    # loader.update_rank("data/anime.csv")
    # loader.load_studios_from_csv("data/anime.csv")
    # loader.load_producers_from_csv("data/anime.csv")   

    # #  1. CONNECT
    # db = DatabaseConnectionManager()

    # # 2. INIT managers
    # mysql_idx = MySQLIndexManager(db)
    # mongo_idx = MongoIndexManager(db)

    # # ===============================
    # # CREATE ALL INDEXES
    # # ===============================
    # print("\n=== APPLYING ALL INDEXES ===")
    # mysql_idx.ensure_all_indexes()
    # mongo_idx.ensure_all_indexes()

    # # ===============================
    # # DROP ALL INDEXES
    # # ===============================
    # mysql_idx.drop_all_indexes()
    # mongo_idx.drop_all_indexes()

    # # ===============================
    # # RECREATE INDEXES (drop + create)
    # # ===============================
    # mysql_idx.recreate_indexes()
    # mongo_idx.recreate_indexes()

    # print("\n=== INDEX OPERATIONS COMPLETE ===") 

    
    #loader.optimize_all_tables()
    db_connection = DatabaseConnectionManager()
    perf = PerformanceTest(db_connection, MySQLIndexManager(db_connection), MongoIndexManager(db_connection))
    perf.run()


    #////////////////////////////////////////////////////////////////////////

    #GraphDataGenerator.user_anime_graph_data_generation()


if __name__ == "__main__":
            
    db_project_run()


