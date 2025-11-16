from db_tools.mySQL_tools import MySQLTools
from db_tools.mongoDB_tools import MongoTools
from db_tools.anime_recommendation import AnimeRecommendation
from db_tools.concurrency_test import ConcurrencyTest


def main():

    mysql = MySQLTools()           # automatically connects in constructor
    connection = mysql.connection  # reuse the same open connection
    mongo = MongoTools(
        uri="mongodb+srv://msamosudova:Duckling@mars-cluster.8ruotdw.mongodb.net/?appName=MARS-Cluster",
        db_name="anime_db",
        collection="animes"   # default collection, but we explicitly request per call
    )
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

    print("\n=== Running CONCURRENCY TEST ===")
    df_conc = ConcurrencyTest.run(n_threads=4, iterations=20)

    # Close connection 
    mysql.close()


if __name__ == "__main__":
    main()

