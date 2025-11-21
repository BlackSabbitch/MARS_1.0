from common_tools import CommonTools
from db_tools.database_connection_manager import DatabaseConnectionManager
from db_tools.index_manager import IndexManager
from db_tools.mongoDB_tools import MongoTools
from db_tools.anime_recommendation import AnimeRecommendation
from db_tools.concurrency_rating_test import ConcurrencyRatingTest
import pandas as pd
from db_tools.schema_manager import SchemaManager
from db_tools.database_connection_manager import DatabaseConnectionManager
from db_tools.schema_manager import SchemaManager
from db_tools.index_manager import IndexManager
from db_tools.load_mySQL import LoadMySQL


def main():

    # mysql = MySQLTools()           # automatically connects in constructor
    # connection = mysql.connection  # reuse the same open connection
    # mongo = MongoTools(
    #     uri="mongodb+srv://msamosudova:Duckling@mars-cluster.8ruotdw.mongodb.net/?appName=MARS-Cluster",
    #     db_name="anime_db",
    #     collection="animes"   # default collection, but we explicitly request per call
    # )

    # Connect to db
    db = DatabaseConnectionManager(mysql_database="mars_db")

    # # 1. Create schema
    # schema = SchemaManager(db)
    # schema.create_schema()

    # # # 2. Create indexes
    # index_manager = IndexManager(db)
    # index_manager.ensure_all_indexes()

    # 3. Load data
    loader = LoadMySQL(db)
    loader.load_paths({
        "anime": "data/anime.csv",
        "animelist": "data/animelist.csv",
        "profiles": "data/profiles.csv",
        "ratings": "data/rating_complete.csv"
    })
    
    # loader.load_watching_status()
    # loader.load_anime()
    # loader.load_genres()

    # loader.load_users_from_animelist()
    # loader.load_profiles_map()
    # loader.load_rating_complete()

    # loader.load_score_distribution(loader.read_csv("anime"))

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
    loader.load_animelist()
    # loader.ensure_score_distribution_columns()
    # loader.load_score_distribution(df)
    

    # connection.commit()
    # cursor.close()
    # connection.close()

    # Close connection 
    # mysql.close()


if __name__ == "__main__":
    main()

