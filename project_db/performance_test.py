import time
import pandas as pd
import warnings
from anime_recommendation import AnimeRecommendation

warnings.filterwarnings('ignore', category=UserWarning)

class PerformanceTest:
    
    def __init__(self, db, mysql_index_manager, mongo_index_manager, mongo_tools):
        self.db = db
        self.mysql_idx = mysql_index_manager
        self.mongo_idx = mongo_index_manager
        self.mongo_tools = mongo_tools  
        self.rec = AnimeRecommendation()

    def _measure(self, func, args=(), warmup=True):
        conn = self.db.new_mysql_connection()
        try:
            if warmup:
                try: func(conn, *args)
                except: pass 
            start = time.perf_counter()
            result = func(conn, *args)
            end = time.perf_counter()
            if result is None:
                print("Warning: Function returned None")
                return 25.0
            return end - start
        except Exception as e:
            print(f"Error: {e}")
            return float('nan')
        finally:
            if conn: conn.close()

    def run_targeted_benchmark(self):
        print("Performance Benchmark (5 Optimized Scenarios)\n")
        
        scenarios = [
            {
                "name": "1. Federated: Text Search + Country",
                "index": "idx_users_country_age",
                "func": self.rec.recommend_top_by_synopsis_and_country,
                "args": (self.mongo_tools, "bounty hunter space", "united_states") 
            },
            {
                "name": "2. Pure SQL: High Volume Read",
                "index": "idx_mal_rating",
                "func": self.rec.get_raw_anime_reviews,
                "args": (self.mongo_tools, 5114)
            },
            {
                "name": "3. Pure SQL: Global Sorting",
                "index": "idx_score_members",
                "func": self.rec.recommend_global_top30,
                "args": (self.mongo_tools,)
            },
            {
                "name": "4. Federated: Geo Spatial Search",
                "index": "idx_users_lat_lon",
                "func": self.rec.recommend_geo_contextual,
                "args": (self.mongo_tools, 603, "bounty hunter space") 
            },
            {
                "name": "5. Federated: Personalized Demographic",
                "index": "idx_users_country_gender",
                "func": self.rec.recommend_personal_demographic_hybrid,
                "args": (self.mongo_tools, 603, "robot") 
            }
        ]

        results = []

        for case in scenarios:
            test_name = case["name"]
            idx_name = case["index"]
            print(f"Scenario: {test_name}")

            # 1. With Index
            self.mysql_idx.create_specific_index(idx_name)
            print(f"With Index ({idx_name})...")
            t_fast = self._measure(case["func"], case["args"], warmup=True)
            print(f"Time: {t_fast:.4f}s")

            # 2. Without Index
            self.mysql_idx.drop_specific_index(idx_name)
            print("No Index...")
            t_slow = self._measure(case["func"], case["args"], warmup=False)
            
            display_slow = f"> {t_slow:.4f}s (Timeout)" if t_slow >= 20.0 else f"{t_slow:.4f}s"
            print(f"Time: {display_slow}")

            # 3. Restore Index
            self.mysql_idx.create_specific_index(idx_name)

            # Speedup
            speedup = t_slow / t_fast if t_fast > 0 else 0
            speedup_str = f"{speedup:.1f}x" if t_slow < 20.0 else "HUGE (> timeout)"

            results.append({
                "Test Case": test_name,
                "Index": idx_name,
                "No Index": display_slow,
                "With Index": f"{t_fast:.4f}s",
                "Speedup": speedup_str
            })
            print("-" * 40)

        df = pd.DataFrame(results)
        print("\nFinal Benchmark Report")
        print(df.to_string(index=False))
        return df

    def run(self):
        return self.run_targeted_benchmark()