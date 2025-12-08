import os
import json
import ijson
import pandas as pd
from datetime import datetime


class UserFeaturePipeline:

    def __init__(self, base_path="../data/helpers/users/"):
        self.base_path = base_path
        self.user_json = base_path + "../user_dict.json"

       
        self.users_deltas = base_path + "users_deltas_for_ML.csv"
        self.final_features = base_path + "final_features_for_ML.csv"
        self.final_features_clean = base_path + "final_features_clean_for_ML.csv"

       
        self.partition_path = "../data/experiments/users/partition_0_1.csv"
        self.users_static_path = "../data/datasets/users_sterilized.csv"
        self.graph_features_path = "../data/helpers/users/graph_features_yearlyall_years_summary.csv"

   

    @staticmethod
    def safe_year(timestamp):
        try:
            return datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f").year
        except:
            return None

    @staticmethod
    def file_exists(path):
        return os.path.exists(path) and os.path.getsize(path) > 0

    def build_users_year(self, start_year=2006, end_year=2018):
        if self.file_exists(self.users_deltas):
            print("File exists, skipping:", self.users_deltas)
            return pd.read_csv(self.users_deltas)

        print("Building user deltas...")

        users_year = {}
        count = 0

        with open(self.user_json, "r", encoding="utf8") as f:
            parser = ijson.kvitems(f, "")

            for username, anime_entries in parser:
                users_year[username] = {}
                count += 1
                if count % 1000 == 0:
                    print("Users processed:", count)

                for year in range(start_year, end_year):
                    filtered = [
                        entry for entry in anime_entries
                        if entry[1] != 0 and
                        self.safe_year(entry[2]) is not None and
                        self.safe_year(entry[2]) <= year
                    ]

                    scores = [entry[1] for entry in filtered]
                    anime_number = len(filtered)
                    av_score = sum(scores) / len(scores) if scores else 0

                    users_year[username][year] = (anime_number, av_score)

        
        records = []
        for user, years_data in users_year.items():
            for year, (num, av) in years_data.items():
                records.append({
                    "username": user,
                    "year": year,
                    "anime_number": num,
                    "av_score": av
                })

        df = pd.DataFrame(records).sort_values(["username", "year"])
        df.to_csv(self.users_deltas, index=False)

        print("Saved:", self.users_deltas)
        return df

    def build_final_features(self):
        if self.file_exists(self.final_features):
            print("File exists, skipping:", self.final_features)
            return pd.read_csv(self.final_features)

        print("Building final feature set...")

        # Load history and filter clusters
        df_history = pd.read_csv(self.partition_path)
        df_history = df_history.sort_values(['username', 'year'])

        df_history['cluster_id_next'] = df_history.groupby('username')['cluster_id'].shift(-1)
        df_history['year_next'] = df_history.groupby('username')['year'].shift(-1)

        df_history_clean = df_history[
            (df_history['year_next'].notna()) &
            (df_history['year_next'] == df_history['year'] + 1)
        ].copy()

        df_history_clean['Y_Delta_Migration'] = (
            df_history_clean['cluster_id'] != df_history_clean['cluster_id_next']
        ).astype(int)

        df_migration_target = df_history_clean[['username', 'year', 'cluster_id', 'Y_Delta_Migration']]

        # Load static user features
        df_users_static = pd.read_csv(self.users_static_path).drop(columns=['Unnamed: 0'])

        # Merge user static data
        df_current_features = pd.merge(
            df_migration_target, df_users_static, on='username', how='left'
        )

        # Birth year & age
        df_current_features['birth_date'] = pd.to_datetime(df_current_features['birth_date'])
        df_current_features['birth_year'] = df_current_features['birth_date'].dt.year
        df_current_features['age'] = df_current_features['year'] - df_current_features['birth_year']
        df_current_features = df_current_features.drop(columns=['birth_date', 'birth_year', 'location'])

        # Add anime deltas per year
        df_anime_lists = pd.read_csv(self.users_deltas)
        df_yearly_stats = df_anime_lists.sort_values(['username', 'year'])
        df_yearly_stats['titles_prev'] = df_yearly_stats.groupby('username')['anime_number'].shift(1)
        df_yearly_stats['score_prev'] = df_yearly_stats.groupby('username')['av_score'].shift(1)
        df_yearly_stats['delta_titles'] = df_yearly_stats['anime_number'] - df_yearly_stats['titles_prev']
        df_yearly_stats['delta_avg_score'] = df_yearly_stats['av_score'] - df_yearly_stats['score_prev']

        # Graph features
        df_graph = pd.read_csv(self.graph_features_path).sort_values(['username', 'year'])
        graph_cols = ["degree", "strength", "pagerank", "kcore", "clustering", "icr", "outside_ratio"]

        # for col in graph_cols:
        #     df_graph[col + "_next"] = df_graph.groupby("username")[col].shift(-1)
        #     df_graph["delta_" + col] = df_graph[col + "_next"] - df_graph[col]

        for col in graph_cols:
            df_graph[col + "_prev"] = df_graph.groupby("username")[col].shift(1)
            df_graph["delta_" + col] = df_graph[col] - df_graph[col + '_prev']


        df_delta_features = df_yearly_stats[['username', 'year', 'delta_titles', 'delta_avg_score']]

        # Merge all features
        df_final = pd.merge(df_current_features, df_delta_features, on=['username', 'year'], how='left')
        df_final = pd.merge(df_final, df_graph, on=['username', 'year'], how='left')

        df_final.to_csv(self.final_features, index=False)
        print("Saved:", self.final_features)

        return df_final

    
    def clean_final_features(self):
        if self.file_exists(self.final_features_clean):
            print("File exists, skipping:", self.final_features_clean)
            return pd.read_csv(self.final_features_clean)

        print("Cleaning final feature set...")

        df = pd.read_csv(self.final_features)

        cols_to_drop = [
            "username", "year", "join_date", "cluster_id_y",
            "degree_prev", "strength_prev", "pagerank_prev",
            "kcore_prev", "clustering_prev", "icr_prev",
            "outside_ratio_prev", "outside_ratio", "delta_outside_ratio"
        ]

        df_clean = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

        df_clean.to_csv(self.final_features_clean, index=False)

        print("FINISH columns:", df_clean.columns.tolist())
        print("Shape:", df_clean.shape)
        print("Saved:", self.final_features_clean)

        return df_clean



