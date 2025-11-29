import json
import pandas as pd
from collections import defaultdict
import numpy as np

class AnimeJSONBuilder:
    def __init__(self, anime_csv, animelist_csv):
        self.anime_csv = anime_csv
        self.animelist_csv = animelist_csv

        self.df_anime = pd.read_csv(anime_csv)
        self.valid_anime_ids = set(self.df_anime["anime_id"])


    def build_anime_dict(self, output_path, chunk_size=200_000):
        anime_dict = defaultdict(list)

        for chunk in pd.read_csv(self.animelist_csv, chunksize=chunk_size, dtype={"my_last_updated": "string"}):
            #print(f"Processing chunk of size: {chunk_size}")

            chunk = chunk[chunk["anime_id"].isin(self.valid_anime_ids)]
            chunk = chunk[chunk["my_score"] != 0]

            for row in chunk.itertuples(index=False):
                anime_dict[row.anime_id].append(
                    (row.username, row.my_score, row.my_last_updated)
                )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(anime_dict, f, ensure_ascii=False, indent=4)

        
        return anime_dict


    def build_user_dict(self, output_path, chunk_size=200_000):
        user_dict = defaultdict(list)

        for chunk in pd.read_csv(self.animelist_csv, chunksize=chunk_size, dtype={"my_last_updated": "string"}):
            #print(f"Processing chunk of size: {chunk_size}")

            for row in chunk.itertuples(index=False):
                if row.my_score != 0:
                    user_dict[row.username].append(
                        (row.anime_id, row.my_score, row.my_last_updated)
                    )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(user_dict, f, ensure_ascii=False, indent=4)

        
        return user_dict
    
    def filter_user_dict_percentile(self, output_path, user_dict, percentile=95):

        anime_counts = np.array([len(v) for v in user_dict.values()])
        
        cutoff = np.percentile(anime_counts, percentile)

        filtered_dict = {
            user: entries
            for user, entries in user_dict.items()
            if len(entries) <= cutoff
        }

        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(filtered_dict, f, ensure_ascii=False, indent=4)

        return filtered_dict