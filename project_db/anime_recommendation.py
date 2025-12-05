import pandas as pd
import warnings
from mysql.connector import Error as MySQLError
from mongoDB_tools import MongoTools

# Suppress annoying Pandas warnings once and for all
warnings.filterwarnings('ignore', category=UserWarning)

class AnimeRecommendation:
    
    # SCENARIO 1: FEDERATED (Text Search -> SQL Aggregation)
    @staticmethod
    def recommend_top_by_synopsis_and_country(connection, mongo_tools: MongoTools, keywords="demon sword", country='Portugal'):
        """
        FEDERATED QUERY #1:
        1. Mongo: Searches for anime IDs where the synopsis matches keywords (Full Text Search).
        2. App: Transfers the list of found IDs to the application layer.
        3. MySQL: Calculates the average rating of these specific anime, 
           considering ONLY votes from users in a specific country.
        """
        print(f"[Federation] Mongo Text Search: '{keywords}'...")
        candidate_ids = mongo_tools.search_anime_ids_by_text(keywords, limit=200)
        
        if not candidate_ids:
            return pd.DataFrame()

        ids_sql_string = ",".join(map(str, candidate_ids))

        query = f"""
        /*+ MAX_EXECUTION_TIME(25000) */
        SELECT a.MAL_ID, a.name, AVG(r.user_rating) as local_rating, COUNT(*) as votes
        FROM anime a
        JOIN anime_user_rating r ON r.MAL_ID = a.MAL_ID
        JOIN users u ON u.userID = r.userID
        WHERE a.MAL_ID IN ({ids_sql_string})
          AND u.country = '{country}'
        GROUP BY a.MAL_ID
        HAVING votes >= 3
        ORDER BY local_rating DESC
        LIMIT 20;
        """
        try:
            df = pd.read_sql(query, connection)
            return mongo_tools.enrich_with_synopsis(df)
        except Exception: 
            return None

    # SCENARIO 2: Heavy SQL SELECT (Used for Index Benchmark)
    @staticmethod
    def get_raw_anime_reviews(connection, _mongo_tools: MongoTools, mal_id): 
        """
        Pure SQL: Fetches a large list of raw reviews (10,000 rows).
        FIXED: Filters out NULL/0 ratings to show actual user scores.
        """
        query = f"""
        /*+ MAX_EXECUTION_TIME(60000) */
        SELECT userID, user_rating
        FROM anime_user_rating
        WHERE MAL_ID = {mal_id} 
          AND user_rating IS NOT NULL 
          AND user_rating > 0
        LIMIT 10000;
        """
        try:
            cursor = connection.cursor(dictionary=True)
            cursor.execute(query)
            rows = cursor.fetchall()
            cursor.close()
            return pd.DataFrame(rows)
        except MySQLError as e:
            if "3024" in str(e) or "interrupted" in str(e).lower():
                raise TimeoutError("Query timed out")
            return None

    # SCENARIO 3: Pure SQL (Sorting)
    @staticmethod
    def recommend_global_top30(connection, mongo_tools: MongoTools):
        """
        Pure SQL: Simple sorting query to benchmark 'idx_score_members'.
        """
        query = """
        /*+ MAX_EXECUTION_TIME(20000) */
        SELECT a.MAL_ID, a.name, s.score, s.members
        FROM anime a
        JOIN anime_statistics s USING (MAL_ID)
        WHERE s.score IS NOT NULL
        ORDER BY s.score DESC, s.members DESC
        LIMIT 30;
        """
        try:
            df = pd.read_sql(query, connection)
            return mongo_tools.enrich_with_synopsis(df)
        except Exception:
            return None

    # SCENARIO 4: Federated (Geo-Spatial + Text Context) 
    @staticmethod
    def recommend_geo_contextual(connection, mongo_tools: MongoTools, target_user_id=1, keywords="demon travel"):
        """
        FEDERATED QUERY #2:
        Finds neighbors physically close to the user (MySQL) and recommends 
        Anime matching the keywords (Mongo) that those neighbors liked.
        """
        # 1. Mongo: Get content-relevant Anime IDs
        print(f"[Federation] Mongo Text Search: '{keywords}'...")
        candidate_ids = mongo_tools.search_anime_ids_by_text(keywords, limit=200)
        if not candidate_ids: 
            return pd.DataFrame()

        ids_sql_string = ",".join(map(str, candidate_ids))

        # 2. MySQL: Get Target User Coordinates
        try:
            cursor = connection.cursor()
            cursor.execute(f"SELECT latitude, longitude FROM users WHERE userID = {target_user_id}")
            target = cursor.fetchone()
            cursor.close()
        except: return None
        
        if not target: 
            print(f"User {target_user_id} has no location data.")
            return None
            
        t_lat, t_lon = target
        delta = 10.0 # Search radius ~10 degrees
        
        # 3. MySQL: Spatial Join + Collaborative Filtering
        # Logic: Find users within 'delta' of target, then aggregate THEIR ratings 
        # for the anime identified by Mongo.
        query = f"""
        /*+ MAX_EXECUTION_TIME(25000) */
        WITH neighbors AS (
            SELECT userID
            FROM users
            WHERE userID != {target_user_id}
              AND latitude BETWEEN {t_lat} - {delta} AND {t_lat} + {delta}
              AND longitude BETWEEN {t_lon} - {delta} AND {t_lon} + {delta}
            LIMIT 100
        )
        SELECT 
            a.MAL_ID, 
            a.name, 
            AVG(r.user_rating) as neighbor_score,
            COUNT(r.user_rating) as neighbor_votes
        FROM anime a
        JOIN anime_user_rating r ON r.MAL_ID = a.MAL_ID
        JOIN neighbors n ON n.userID = r.userID
        WHERE a.MAL_ID IN ({ids_sql_string}) -- Content Filter (Mongo)
        GROUP BY a.MAL_ID
        HAVING neighbor_votes >= 1
        ORDER BY neighbor_score DESC
        LIMIT 20;
        """
        try:
            df = pd.read_sql(query, connection)
            return mongo_tools.enrich_with_synopsis(df) 
        except Exception as e:
             return None

    # SCENARIO 5: Personalized recommendation (Dynamic User Profile)
    @staticmethod
    def recommend_personal_demographic_hybrid(connection, mongo_tools: MongoTools, user_id=1, keywords="love robot"):
        """
        Federated query #3 (Personalized):
        Combines Content Search (Mongo), User Profile (MySQL), and Collaborative Filtering (MySQL).
        """
        print(f"[Federation] Mongo Text Search: '{keywords}'...")
        candidate_ids = mongo_tools.search_anime_ids_by_text(keywords, limit=200)
        if not candidate_ids: return pd.DataFrame()
        ids_sql_string = ",".join(map(str, candidate_ids))

        try:
            cursor = connection.cursor()
            cursor.execute(f"SELECT country, sex FROM users WHERE userID = {user_id}")
            user_profile = cursor.fetchone()
            cursor.close()
        except Exception: return None

        if not user_profile:
            print("User not found.")
            return None
        
        u_country, u_gender = user_profile
        if not u_country or not u_gender:
            u_country = "Portugal" 
            u_gender = "Male"
            
        print(f"[Profile] User {user_id} is {u_gender} from {u_country}. Filtering peers...")

        query = f"""
        /*+ MAX_EXECUTION_TIME(25000) */
        SELECT a.MAL_ID, a.name, 
               AVG(r.user_rating) as demographic_score, 
               COUNT(r.user_rating) as demographic_votes
        FROM anime a
        JOIN anime_user_rating r ON r.MAL_ID = a.MAL_ID
        JOIN users u ON u.userID = r.userID
        WHERE a.MAL_ID IN ({ids_sql_string})
          AND u.country = '{u_country}'
          AND u.sex = '{u_gender}'
        GROUP BY a.MAL_ID
        HAVING demographic_votes >= 2
        ORDER BY demographic_score DESC
        LIMIT 20;
        """
        try:
            df = pd.read_sql(query, connection)
            return mongo_tools.enrich_with_synopsis(df)
        except Exception as e:
             if "3024" in str(e) or "interrupted" in str(e).lower():
                raise TimeoutError("Query timed out")
             return None

    # Print helpers (Console Output)
    @classmethod
    def print_recommend_top_by_synopsis_and_country(cls, connection, mongo_tools, keywords, country, limit=10):
        """Executes and prints Scenario 1."""
        print(f"\n[1] Top '{keywords}' Anime in {country}")
        df = cls.recommend_top_by_synopsis_and_country(connection, mongo_tools, keywords, country)
        cls._print_results(df, limit)

    @classmethod
    def print_get_raw_anime_reviews(cls, connection, mongo_tools, mal_id, limit=10):
        """Executes and prints Scenario 2."""
        print(f"\n[2] Raw Reviews for Anime ID {mal_id}")
        df = cls.get_raw_anime_reviews(connection, mongo_tools, mal_id)
        cls._print_results(df, limit)

    @classmethod
    def print_recommend_global_top30(cls, connection, mongo_tools, limit=10):
        """Executes and prints Scenario 3."""
        print(f"\n[3] Global Top 30 Anime")
        df = cls.recommend_global_top30(connection, mongo_tools)
        cls._print_results(df, limit)

    @classmethod
    def print_find_nearest_neighbors_contextual(cls, connection, mongo_tools, user_id, keywords, limit=10):
        """Executes and prints Scenario 4."""
        print(f"\n--- [4] Geo-Contextual Recs for User {user_id} (Neighbors + '{keywords}') ---")
        df = cls.recommend_geo_contextual(connection, mongo_tools, user_id, keywords)
        cls._print_results(df, limit)

    @classmethod
    def print_recommend_personal_demographic_hybrid(cls, connection, mongo_tools, user_id, keywords, limit=10):
        """Executes and prints Scenario 5."""
        print(f"\n[5] Personalized Recs for User {user_id} (Context: '{keywords}')")
        df = cls.recommend_personal_demographic_hybrid(connection, mongo_tools, user_id, keywords)
        cls._print_results(df, limit)

    @staticmethod
    def _print_results(df, limit):
        """Helper to format and print the dataframe."""
        if df is None or df.empty:
            print("No results found.")
            return
        
        # Print summary
        count = len(df)
        print(f"Found {count} results. Showing top {limit}:")
        
        # Adjust display options for console
        with pd.option_context('display.max_rows', limit, 'display.max_columns', None, 'display.width', 1000):
            print(df.head(limit))
        print("-" * 60)