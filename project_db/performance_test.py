import time
import pandas as pd
import warnings
from anime_recommendation import AnimeRecommendation

# Suppress Pandas warnings
warnings.filterwarnings('ignore', category=UserWarning)

class PerformanceTest:
    
    def __init__(self, db, mysql_index_manager, mongo_index_manager, mongo_tools):
        self.db = db
        self.mysql_idx = mysql_index_manager
        self.mongo_idx = mongo_index_manager
        self.mongo_tools = mongo_tools  
        self.rec = AnimeRecommendation()

    def _measure(self, func, args=(), warmup=True):
        """
        Runs the function and measures execution time. 
        Creates connection OUTSIDE the timer to measure query time, not handshake time.
        """
        conn = self.db.new_mysql_connection()
        try:
            # 1. Warmup (optional, execute once without measuring to fill Buffer Pool)
            if warmup:
                try:
                    func(conn, *args)
                except:
                    pass # Ignore warmup errors

            # 2. Measurement
            start = time.perf_counter()
            result = func(conn, *args)
            end = time.perf_counter()
            
            # If result is None, it means an internal error occurred
            if result is None:
                print("[!] Warning: Function returned None")
                return 25.0 # Penalty for failure

            return end - start

        except Exception as e:
            print(f"Error during test execution: {e}")
            return float('nan')
        finally:
            if conn:
                conn.close()

    def run_targeted_benchmark(self):
        print("Performance benchmark (5 Optimized Scenarios)\n")
        
        scenarios = [
            # 1. FEDERATED: Mongo Text -> MySQL Join
            {
                "name": "1. [Federated] Top 'bounty hunter space' in united_states",
                "index": "idx_users_country_age", 
                "func": self.rec.recommend_top_by_synopsis_and_country,
                "args": (self.mongo_tools, "bounty hunter space", "united_states") 
            },
            # 2. PURE SQL: Heavy Selection (Reverse Lookup)
            {
                "name": "2. [SQL Pure] 10k Reviews (ID 1 - Cowboy Bebop)",
                "index": "idx_mal_rating",
                "func": self.rec.get_raw_anime_reviews,
                "args": (self.mongo_tools, 1) 
            },
            # 3. PURE SQL: Global Sorting
            {
                "name": "3. [SQL Pure] Global Top 30",
                "index": "idx_score_members",
                "func": self.rec.recommend_global_top30,
                "args": (self.mongo_tools,)
            },
            # 4. FEDERATED: Mongo Text -> MySQL Geo
            {
                "name": "4. [Federated] Neighbors for User 603 ('bounty hunter space')",
                "index": "idx_users_lat_lon",
                "func": self.rec.recommend_geo_contextual, 
                "args": (self.mongo_tools, 603, "bounty hunter space") 
            },
            # 5. FEDERATED: Personalized Hybrid (Demographics)
            {
                "name": "5. [Federated] Personalized for User 603 (Context: 'robot')",
                "index": "idx_users_country_gender",
                "func": self.rec.recommend_personal_demographic_hybrid,
                "args": (self.mongo_tools, 603, "robot") 
            }
        ]

        results = []

        for case in scenarios:
            test_name = case["name"]
            idx_name = case["index"]
            print(f"--- Scenario: {test_name} ---")

            # 1. Ensure Index Exists (Baseline - Fast)
            self.mysql_idx.create_specific_index(idx_name)
            print(f"-- Running WITH index {idx_name}...")
            t_fast = self._measure(case["func"], case["args"], warmup=True)
            print(f"-- Time: {t_fast:.4f}s")

            # 2. Drop Index (Test - Slow)
            self.mysql_idx.drop_specific_index(idx_name)
            print("-- Running WITHOUT index...")
            t_slow = self._measure(case["func"], case["args"], warmup=False) # No warmup for cold test
            
            display_slow = f"> {t_slow:.4f}s (Timeout)" if t_slow >= 20.0 else f"{t_slow:.4f}s"
            print(f"-- Time: {display_slow}")

            # 3. Restore Index immediately
            self.mysql_idx.create_specific_index(idx_name)

            # Calculate Speedup
            speedup = t_slow / t_fast if t_fast > 0 else 0
            speedup_str = f"{speedup:.1f}x" if t_slow < 20.0 else "Huge (> timeout)"

            results.append({
                "Test Case": test_name,
                "Index Toggled": idx_name,
                "Time (No Index)": display_slow,
                "Time (With Index)": f"{t_fast:.4f}s",
                "Speedup": speedup_str
            })
            print("-" * 60)

        # Output Summary Table
        df = pd.DataFrame(results)
        print("\nFinal Benchmark Report")
        print(df.to_string(index=False))
        
        return df

    def run(self):
        return self.run_targeted_benchmark()