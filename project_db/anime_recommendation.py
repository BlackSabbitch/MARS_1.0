import pandas as pd
import warnings
from mysql.connector import Error as MySQLError
from mongoDB_tools import MongoTools

warnings.filterwarnings('ignore', category=UserWarning)

class AnimeRecommendation:
    
    # SCENARIO 1
    @staticmethod
    def recommend_top_by_synopsis_and_country(connection, mongo_tools: MongoTools, keywords="demon sword", country='Portugal'):
        print(f"[Federation] Mongo Text Search: '{keywords}'...")
        candidate_ids = mongo_tools.search_anime_ids_by_text(keywords, limit=200)
        if not candidate_ids: return pd.DataFrame()
        ids_sql_string = ",".join(map(str, candidate_ids))

        query = f"""
        /*+ MAX_EXECUTION_TIME(25000) */
        SELECT a.MAL_ID, a.name, AVG(r.user_rating) as local_rating, COUNT(*) as votes
        FROM anime a
        JOIN anime_user_rating r ON r.MAL_ID = a.MAL_ID
        JOIN users u ON u.userID = r.userID
        WHERE a.MAL_ID IN ({ids_sql_string}) AND u.country = '{country}'
        GROUP BY a.MAL_ID HAVING votes >= 3
        ORDER BY local_rating DESC LIMIT 20;
        """
        try:
            df = pd.read_sql(query, connection)
            return mongo_tools.enrich_with_synopsis(df)
        except Exception: return None

    # SCENARIO 2
    @staticmethod
    def get_raw_anime_reviews(connection, _mongo_tools: MongoTools, mal_id): 
        query = f"""
        /*+ MAX_EXECUTION_TIME(60000) */
        SELECT userID, user_rating FROM anime_user_rating
        WHERE MAL_ID = {mal_id} AND user_rating IS NOT NULL AND user_rating > 0
        LIMIT 10000;
        """
        try:
            cursor = connection.cursor(dictionary=True)
            cursor.execute(query)
            rows = cursor.fetchall()
            cursor.close()
            return pd.DataFrame(rows)
        except MySQLError as e: return None

    # SCENARIO 3
    @staticmethod
    def recommend_global_top30(connection, mongo_tools: MongoTools):
        query = """
        /*+ MAX_EXECUTION_TIME(20000) */
        SELECT a.MAL_ID, a.name, s.score, s.members
        FROM anime a JOIN anime_statistics s USING (MAL_ID)
        WHERE s.score IS NOT NULL
        ORDER BY s.score DESC, s.members DESC LIMIT 30;
        """
        try:
            df = pd.read_sql(query, connection)
            return mongo_tools.enrich_with_synopsis(df)
        except Exception: return None

    # SCENARIO 4
    @staticmethod
    def recommend_geo_contextual(connection, mongo_tools: MongoTools, target_user_id=1, keywords="demon travel"):
        print(f"[Federation] Mongo Text Search: '{keywords}'...")
        candidate_ids = mongo_tools.search_anime_ids_by_text(keywords, limit=50)
        if not candidate_ids: return pd.DataFrame()
        ids_sql_string = ",".join(map(str, candidate_ids))

        try:
            cursor = connection.cursor()
            cursor.execute(f"SELECT latitude, longitude FROM users WHERE userID = {target_user_id}")
            target = cursor.fetchone()
            cursor.close()
        except: return None
        
        if not target: return None
        t_lat, t_lon = target
        delta = 10.0 
        
        query = f"""
        /*+ MAX_EXECUTION_TIME(25000) */
        WITH neighbors AS (
            SELECT userID FROM users
            WHERE userID != {target_user_id}
              AND latitude BETWEEN {t_lat} - {delta} AND {t_lat} + {delta}
              AND longitude BETWEEN {t_lon} - {delta} AND {t_lon} + {delta}
            LIMIT 100
        )
        SELECT a.MAL_ID, a.name, AVG(r.user_rating) as neighbor_score, COUNT(r.user_rating) as neighbor_votes
        FROM anime a
        JOIN anime_user_rating r ON r.MAL_ID = a.MAL_ID
        JOIN neighbors n ON n.userID = r.userID
        WHERE a.MAL_ID IN ({ids_sql_string})
        GROUP BY a.MAL_ID HAVING neighbor_votes >= 1
        ORDER BY neighbor_score DESC LIMIT 20;
        """
        try:
            df = pd.read_sql(query, connection)
            return mongo_tools.enrich_with_synopsis(df)
        except Exception: return None

    # SCENARIO 5
    @staticmethod
    def recommend_personal_demographic_hybrid(connection, mongo_tools: MongoTools, user_id=1, keywords="love robot"):
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

        if not user_profile: return None
        u_country, u_gender = user_profile
        if not u_country or not u_gender: return None 

        print(f"[Profile] User {user_id} is {u_gender} from {u_country}. Filtering peers...")

        query = f"""
        /*+ MAX_EXECUTION_TIME(25000) */
        SELECT a.MAL_ID, a.name, AVG(r.user_rating) as demographic_score, COUNT(r.user_rating) as demographic_votes
        FROM anime a
        JOIN anime_user_rating r ON r.MAL_ID = a.MAL_ID
        JOIN users u ON u.userID = r.userID
        WHERE a.MAL_ID IN ({ids_sql_string}) AND u.country = '{u_country}' AND u.sex = '{u_gender}'
        GROUP BY a.MAL_ID HAVING demographic_votes >= 2
        ORDER BY demographic_score DESC LIMIT 20;
        """
        try:
            df = pd.read_sql(query, connection)
            return mongo_tools.enrich_with_synopsis(df)
        except Exception: return None

    # Print helpers
    @classmethod
    def print_recommend_top_by_synopsis_and_country(cls, connection, mongo_tools, keywords, country, limit=10):
        print(f"\n--- [1] Top '{keywords}' Anime in {country} ---")
        cls._print_results(cls.recommend_top_by_synopsis_and_country(connection, mongo_tools, keywords, country), limit)

    @classmethod
    def print_get_raw_anime_reviews(cls, connection, mongo_tools, mal_id, limit=10):
        anime_name = "Unknown"
        try:
            cursor = connection.cursor()
            cursor.execute(f"SELECT name FROM anime WHERE MAL_ID = {mal_id}")
            result = cursor.fetchone()
            if result:
                anime_name = result[0]
            cursor.close()
        except:
            pass
        print(f"\n--- [2] Raw Reviews for: '{anime_name}' (ID {mal_id}) ---")
        df = cls.get_raw_anime_reviews(connection, mongo_tools, mal_id)
        cls._print_results(df, limit)

    @classmethod
    def print_recommend_global_top30(cls, connection, mongo_tools, limit=10):
        print(f"\n--- [3] Global Top 30 Anime ---")
        cls._print_results(cls.recommend_global_top30(connection, mongo_tools), limit)

    @classmethod
    def print_find_nearest_neighbors_contextual(cls, connection, mongo_tools, user_id, keywords, limit=10):
        print(f"\n--- [4] Geo-Contextual Recs for User {user_id} ---")
        # Maps to recommend_geo_contextual
        cls._print_results(cls.recommend_geo_contextual(connection, mongo_tools, user_id, keywords), limit)

    @classmethod
    def print_recommend_personal_demographic_hybrid(cls, connection, mongo_tools, user_id, keywords, limit=10):
        print(f"\n--- [5] Personalized Recs for User {user_id} ---")
        cls._print_results(cls.recommend_personal_demographic_hybrid(connection, mongo_tools, user_id, keywords), limit)

    @staticmethod
    def _print_results(df, limit):
        if df is None or df.empty:
            print("No results found.")
            return
        print(f"Found {len(df)} results. Showing top {limit}:")
        with pd.option_context('display.max_rows', limit, 'display.max_columns', None, 'display.width', 1000):
            print(df.head(limit))
        print("-" * 60)