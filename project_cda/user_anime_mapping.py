import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ijson
from datetime import datetime


class UserAnimeMapping:
    def __init__(self, user_clusters_path, anime_clusters_path, user_dict_json):
        self.user_clusters = pd.read_csv(user_clusters_path)
        self.anime_clusters = pd.read_csv(anime_clusters_path)
        self.user_dict_json = user_dict_json
        self.df_ratings = None

    
    @staticmethod
    def safe_year(ts):
        try:
            return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f").year
        except:
            return 9999999

 
    def build_correlation(self):
        rows = []

        with open(self.user_dict_json, "r", encoding="utf-8") as f:
            parser = ijson.kvitems(f, "")
            for username, anime_list in parser:
                for anime_id, score, last_updated in anime_list:
                    year = self.safe_year(last_updated)
                    rows.append((year, username, anime_id, score))

        df = pd.DataFrame(rows, columns=["year", "username", "anime_id", "score"])
        df["score"] = pd.to_numeric(df["score"], errors="coerce")

        df = df.merge(self.user_clusters, on=["year", "username"], how="left")
        df.rename(columns={"cluster_id": "user_cluster"}, inplace=True)

       
        df = df.merge(self.anime_clusters, on=["year", "anime_id"], how="left")
        df.rename(columns={"cluster_id": "anime_cluster"}, inplace=True)

        df = df.dropna(subset=["user_cluster", "anime_cluster"])

        self.df_ratings = df
        return df


    def plot_heatmap(
        self,
        year,
        max_ticks=25,
        log_base=np.e,
        figsize=(12, 8)
    ):
        if self.df_ratings is None:
            raise ValueError("You must call build_correlation() first.")

        df_year = self.df_ratings[self.df_ratings["year"] == year]

        matrix = (
            df_year.groupby(["user_cluster", "anime_cluster"])
            .size()
            .unstack(fill_value=0)
        )

       
        if log_base == 10:
            matrix_plot = np.log10(1 + matrix)
            colorbar_label = "log10(1 + count)"
        else:
            matrix_plot = np.log1p(matrix)
            colorbar_label = "ln(1 + count)"

        
        def select_ticks(values, max_ticks):
            if len(values) <= max_ticks:
                return np.arange(len(values)), values
            idx = np.linspace(0, len(values) - 1, max_ticks).astype(int)
            return idx, values[idx]

        row_idx, row_labels = select_ticks(matrix.index.to_numpy(), max_ticks)
        col_idx, col_labels = select_ticks(matrix.columns.to_numpy(), max_ticks)

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(matrix_plot.values, aspect='auto', origin='lower')

        ax.set_yticks(row_idx)
        ax.set_yticklabels(row_labels)

        ax.set_xticks(col_idx)
        ax.set_xticklabels(col_labels, rotation=90)

        ax.set_xlabel("anime_cluster")
        ax.set_ylabel("user_cluster")
        ax.set_title(f"User-Anime Heatmap ({year})")

        cbar = fig.colorbar(im)
        cbar.set_label(colorbar_label)

        plt.tight_layout()
        plt.show()
