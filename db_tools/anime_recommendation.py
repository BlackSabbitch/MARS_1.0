from csv import Error
import mysql.connector
import pandas as pd
from common_tools import CommonTools
from db_tools.mongoDB_tools import MongoTools
from db_tools.mySQL_tools import MySQLTools


class AnimeRecommendation:

    def hundred_best(connection):
        query = """
        SELECT
            a.MAL_ID,
            a.name,
            COALESCE(s.score, 0) AS rating,
            s.members,
            s.favorites,
            s.popularity
        FROM anime a
        JOIN anime_statistics s USING (MAL_ID)
        WHERE s.score IS NOT NULL
        AND COALESCE(s.members, 0) >= 1000
        ORDER BY s.score DESC, s.members DESC
        LIMIT 100;
        """

        try:
            df = pd.read_sql(query, connection)

            print("\n=== TOP 100 BEST-RATED ANIME ===")
            print(df.head(20).to_string(index=False))
            print(f"\nTotal rows fetched: {len(df)}")

            return df  # return results for main()

        except mysql.connector.Error as e:
            print("MySQL Error:", e)
            return None


    def hundred_best_with_synopsis(connection, mongo_tools: MongoTools):
    
        # ---------- STEP 1 — Top 100 IDs from MySQL ----------
        query = """
            SELECT a.MAL_ID, a.name, s.score
            FROM anime a
            JOIN anime_statistics s USING (MAL_ID)
            WHERE s.score IS NOT NULL
            ORDER BY s.score DESC, s.members DESC
            LIMIT 100;
        """
        df = pd.read_sql(query, connection)

            # ---------- STEP 2 — Fetch synopsis from MongoDB ----------
        collection = mongo_tools.get_collection("animes")

        syn_map = {
            doc["_id"]: doc.get("synopsis", "")
            for doc in collection.find(
                {"_id": {"$in": df["MAL_ID"].tolist()}},
                {"_id": 1, "synopsis": 1}
            )
        }

        df["synopsis"] = df["MAL_ID"].map(syn_map)

        # ---------- STEP 3 — Print ----------
        print("\n=== TOP 100 ANIME WITH SYNOPSIS (MySQL + Mongo) ===")
        for _, row in df.iterrows():
            syn = (row["synopsis"] or "")[:400]
            print(f"\n{row['name']}  ({row['score']})\n{syn}...")

        print(f"\nTotal rows fetched: {len(df)}")

        return df
    
    def recommend_by_genres(connection, user_id: int, limit: int = 10):
        """
        Recommend top anime for a user based on their highest-lift genres.
        """

        query = f"""
        WITH
        user_avg AS (
          SELECT AVG(user_rating) AS mu
          FROM anime_user_rating
          WHERE userID = {user_id} AND user_rating IS NOT NULL
        ),
        user_genre AS (
          SELECT g.genreID,
                 AVG(r.user_rating - (SELECT mu FROM user_avg)) AS lift,
                 COUNT(*) AS n_in_genre
          FROM anime_user_rating r
          JOIN anime_genre ag ON ag.MAL_ID = r.MAL_ID
          JOIN genre g ON g.genreID = ag.genreID
          WHERE r.userID = {user_id} AND r.user_rating IS NOT NULL
          GROUP BY g.genreID
        ),
        top_genres AS (
          SELECT genreID
          FROM user_genre
          WHERE n_in_genre >= 5
          ORDER BY lift DESC, n_in_genre DESC
          LIMIT 5
        ),
        seen AS (
          SELECT MAL_ID FROM anime_user_rating WHERE userID = {user_id}
        )
        SELECT
          a.MAL_ID,
          a.name,
          s.score,
          s.members,
          GROUP_CONCAT(DISTINCT g.genre ORDER BY g.genre SEPARATOR ', ') AS genres
        FROM anime a
        JOIN anime_statistics s USING (MAL_ID)
        JOIN anime_genre ag ON ag.MAL_ID = a.MAL_ID
        JOIN genre g ON g.genreID = ag.genreID
        WHERE ag.genreID IN (SELECT genreID FROM top_genres)
          AND a.MAL_ID NOT IN (SELECT MAL_ID FROM seen)
          AND s.score IS NOT NULL
          AND COALESCE(s.members, 0) >= 1000
        GROUP BY a.MAL_ID, a.name, s.score, s.members
        ORDER BY s.score DESC, s.members DESC
        LIMIT {limit};
        """

        try:
            df = pd.read_sql(query, connection)

            print(f"\n=== Recommendations for user {user_id} ===\n")
            for i, row in df.iterrows():
                print(f"{i+1:2d}. {row['name']} ({row['score']}, {row['members']:,} members)")
                print(f"    Genres: {row['genres']}\n")

            print(f"Total recommendations: {len(df)}")

            return df

        except Error as e:
            print("MySQL Error:", e)
            return None