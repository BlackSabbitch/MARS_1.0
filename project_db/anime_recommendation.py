from csv import Error
import mysql.connector
import pandas as pd
from mysql.connector import Error as MySQLError

from mongoDB_tools import MongoTools


class AnimeRecommendation:

    @staticmethod
    def recommend_portugal_top30(connection, mongo_tools):

        query = """
        WITH
        pt_users AS (
            SELECT userID FROM users
            WHERE country = 'Portugal' AND age >= 18
        ),
        pt_avg_by_genre AS (
            SELECT g.genreID,
                   AVG(r.user_rating) AS avg_rating,
                   COUNT(*) AS n_ratings
            FROM anime_user_rating r
            JOIN pt_users u ON u.userID = r.userID
            JOIN anime_genre ag ON ag.MAL_ID = r.MAL_ID
            JOIN genre g ON g.genreID = ag.genreID
            WHERE r.user_rating IS NOT NULL
            GROUP BY g.genreID
            HAVING n_ratings >= 5
        ),
        top_genres AS (
            SELECT genreID
            FROM pt_avg_by_genre
            ORDER BY avg_rating DESC, n_ratings DESC
            LIMIT 5
        ),
        scored_anime AS (
            SELECT a.MAL_ID, a.name, s.score, s.members,
                GROUP_CONCAT(DISTINCT g.genre ORDER BY g.genre SEPARATOR ', ') AS genres
            FROM anime a
            JOIN anime_statistics s USING (MAL_ID)
            JOIN anime_genre ag ON ag.MAL_ID = a.MAL_ID
            JOIN genre g ON g.genreID = ag.genreID
            WHERE ag.genreID IN (SELECT genreID FROM top_genres)
              AND s.score IS NOT NULL
              AND COALESCE(s.members, 0) > 500
            GROUP BY a.MAL_ID, a.name, s.score, s.members
        )
        SELECT *
        FROM scored_anime
        ORDER BY score DESC, members DESC
        LIMIT 30;
        """

        try:
            df = pd.read_sql(query, connection)

            print("\n=== Recommendations for Portugal (18+) ===\n")
            for i, row in df.iterrows():
                print(f"{i+1:2d}. {row['name']} ({row['score']}, {row['members']:,} members)")
                print(f"    Genres: {row['genres']}\n")

            print(f"Total recommendations: {len(df)}")

            # ---- SYNOPSIS FROM MONGO ----
            collection = mongo_tools.get_mongo_collection("animes")
            mal_ids = df["MAL_ID"].astype(int).tolist()
            docs = collection.find(
                {"_id": {"$in": mal_ids}},
                {"_id": 1, "synopsis": 1}
            )

            synopsis_map = {doc["_id"]: doc.get("synopsis", "") for doc in docs}
            df["synopsis"] = df["MAL_ID"].map(synopsis_map)

            return df

        except MySQLError as e:
            print("MySQL Error:", e)
            return None

    @staticmethod
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

    @staticmethod
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
        collection = mongo_tools.get_mongo_collection("animes")

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
    
    @staticmethod
    def recommend_by_genres(connection, user_id: int, limit: int = 10):
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
            print("Executing SQL for recommend_by_genres...")
            cursor = connection.cursor(dictionary=True)
            cursor.execute(query)
            rows = cursor.fetchall()
            cursor.close()
            print("Done.")

            df = pd.DataFrame(rows)

            print(f"\n=== Recommendations for user {user_id} ===\n")
            for i, row in df.iterrows():
                print(f"{i+1:2d}. {row['name']} ({row['score']}, {row['members']:,} members)")
                print(f"    Genres: {row['genres']}\n")

            print(f"Total recommendations: {len(df)}")

            return df

        except Error as e:
            print("MySQL Error:", e)
            return None


    @staticmethod
    def recommend_pt_adventure_magic(connection, mongo_tools):
        """
        Recommend top anime liked by Portuguese users
        specifically in genres Adventure + Magic.
        """

        query_old = """
        WITH
        pt_users AS (
            SELECT userID 
            FROM users
            WHERE country = 'Portugal'
              AND age >= 18
        ),

        pt_ratings AS (
            SELECT r.MAL_ID, r.user_rating
            FROM anime_user_rating r
            JOIN pt_users u ON u.userID = r.userID
            WHERE r.MAL_ID IN (
                SELECT MAL_ID
                FROM anime_genre ag
                JOIN genre g ON g.genreID = ag.genreID
                WHERE g.genre IN ('Adventure', 'Magic')
            )
        ),

        pt_scores AS (
            SELECT MAL_ID,
                   AVG(user_rating) AS avg_pt_score,
                   COUNT(*)          AS cnt_pt_votes
            FROM pt_ratings
            GROUP BY MAL_ID
            HAVING cnt_pt_votes >= 3
        )

        SELECT
            a.MAL_ID,
            a.name,
            s.score AS global_score,
            s.members,
            pt.avg_pt_score,
            pt.cnt_pt_votes,
            GROUP_CONCAT(DISTINCT g.genre ORDER BY g.genre SEPARATOR ', ') AS genres
        FROM pt_scores pt
        JOIN anime a USING (MAL_ID)
        JOIN anime_statistics s USING (MAL_ID)
        JOIN anime_genre ag ON ag.MAL_ID = a.MAL_ID
        JOIN genre g ON g.genreID = ag.genreID
        WHERE s.members >= 500
        GROUP BY a.MAL_ID, a.name, s.score, s.members, pt.avg_pt_score, pt.cnt_pt_votes
        ORDER BY pt.avg_pt_score DESC, pt.cnt_pt_votes DESC, s.score DESC
        LIMIT 30;
        """
        query = """
        WITH
        pt_users AS (
            SELECT userID
            FROM users
            WHERE country = 'Portugal'
            AND age >= 18
        ),

        pt_ratings AS (
            SELECT 
                r.MAL_ID,
                r.user_rating
            FROM anime_user_rating r
            JOIN pt_users u ON u.userID = r.userID
            JOIN anime_genre ag2 ON ag2.MAL_ID = r.MAL_ID
            JOIN genre g2 ON g2.genreID = ag2.genreID
            WHERE g2.genre IN ('Adventure', 'Magic')
        ),

        pt_scores AS (
            SELECT 
                MAL_ID,
                AVG(user_rating) AS avg_pt_score,
                COUNT(*)         AS cnt_pt_votes
            FROM pt_ratings
            GROUP BY MAL_ID
            HAVING cnt_pt_votes >= 3
        ),

        genre_map AS (
            SELECT 
                ag.MAL_ID,
                GROUP_CONCAT(DISTINCT g.genre ORDER BY g.genre SEPARATOR ', ') AS genres
            FROM anime_genre ag
            JOIN genre g ON g.genreID = ag.genreID
            GROUP BY ag.MAL_ID
        )

        SELECT
            a.MAL_ID,
            a.name,
            s.score AS global_score,
            s.members,
            pt.avg_pt_score,
            pt.cnt_pt_votes,
            gm.genres
        FROM pt_scores pt
        JOIN anime a USING (MAL_ID)
        JOIN anime_statistics s USING (MAL_ID)
        LEFT JOIN genre_map gm USING (MAL_ID)
        WHERE s.members >= 500
        ORDER BY 
            pt.avg_pt_score DESC,
            pt.cnt_pt_votes DESC,
            s.score DESC
        LIMIT 30;
        """


        try:
            # RUN SQL (avoid pandas.read_sql -> hangs)
            print("Executing SQL...")
            cursor = connection.cursor(dictionary=True)
            # Set 10-second timeout for this SELECT only
            cursor.execute("SET SESSION max_execution_time = 10000")
            cursor.execute(query)
            rows = cursor.fetchall()
            cursor.close()
            print("SQL done!")

            # Convert to DataFrame
            df = pd.DataFrame(rows)

            print("\n=== TOP 30 Adventure + Magic for Portugal (18+) ===\n")
            for i, row in df.iterrows():
                print(
                    f"{i+1:2d}. {row['name']} "
                    f"(PT score={row['avg_pt_score']:.2f}, "
                    f"votes={row['cnt_pt_votes']}, "
                    f"global={row['global_score']})"
                )
                print(f"    Genres: {row['genres']}\n")

            # Add Mongo synopsis
            mal_ids = df["MAL_ID"].astype(int).tolist()
            collection = mongo_tools.get_mongo_collection("animes")

            docs = collection.find(
                {"_id": {"$in": mal_ids}},
                {"_id": 1, "synopsis": 1}
            )

            synopsis_map = {doc["_id"]: doc.get("synopsis", "") for doc in docs}
            df["synopsis"] = df["MAL_ID"].map(synopsis_map)

            return df

        except MySQLError as e:
            if "maximum statement execution time exceeded" in str(e):
                print("Query timed out after 10 seconds — skipping.")
                return None
            print("MySQL Error:", e)
            return None