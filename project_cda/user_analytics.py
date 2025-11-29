import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class UserAnalytics:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.users_path = os.path.join(data_dir, "selected_users_top20.csv")
        self.lists_path = os.path.join(data_dir, "selected_users_top20_animelist.csv")
        self.anime_path = os.path.join(data_dir, "selected_anime_top20.csv")

        # Output folder for PNGs
        self.out_dir = os.path.join(data_dir, "plots")
        os.makedirs(self.out_dir, exist_ok=True)

        # Load merged dataset
        self.df = self._load_merged()

    # ============================================================
    # MERGING USER AND RATING DATA
    # ============================================================

    def _load_merged(self):
        print("Loading top-20 user datasets...")
        users = pd.read_csv(self.users_path)
        lists = pd.read_csv(self.lists_path)
        anime = pd.read_csv(self.anime_path)

        # Convert dates
        users["birth_date"] = pd.to_datetime(users["birth_date"], errors="coerce")
        users["join_date"] = pd.to_datetime(users["join_date"], errors="coerce")
        users["last_online"] = pd.to_datetime(users["last_online"], errors="coerce")
        lists["my_last_updated"] = pd.to_datetime(lists["my_last_updated"], errors="coerce")

        # Compute age
        users["age"] = (pd.Timestamp.now() - users["birth_date"]).dt.days / 365.25

        # Merge lists with anime metadata
        merged = lists.merge(anime, on="anime_id", how="left")
        merged = merged.merge(users, on="username", how="left")

        # Activity year: correct definition
        merged["rating_year"] = merged["my_last_updated"].dt.year
        merged = merged[merged["rating_year"] >= 2004]

        print("Loaded successfully.")
        return merged

    # ============================================================
    # 1. SCORE DISTRIBUTION PER USER
    # ============================================================

    def plot_score_distribution(self):
        df = self.df.copy()
        df = df[df["my_score"] > 0]

        plt.figure(figsize=(14, 6))
        sns.boxplot(data=df, x="username", y="my_score")
        plt.xticks(rotation=45)
        plt.title("Score distribution per user")
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "score_distribution.png"))
        plt.close()
        print("Saved: score_distribution.png")

    # ============================================================
    # 2. ACTIVITY BY YEAR PER USER
    # ============================================================

    def plot_activity_by_year(self):
        df = self.df.copy()

        activity = df.groupby(["rating_year", "username"]).size().reset_index(name="count")

        pivot = activity.pivot(index="rating_year", columns="username", values="count").fillna(0)

        pivot.plot(figsize=(14, 6))
        plt.title("User activity by year (ratings count)")
        plt.xlabel("Year")
        plt.ylabel("Ratings")
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "activity_by_year.png"))
        plt.close()
        print("Saved: activity_by_year.png")


    # ============================================================
    # 3. GENRES OVER TIME
    # ============================================================

    def plot_genres_over_time(self):
        df = self.df.copy()

        # explode genre
        df["genre"] = df["genre"].fillna("")
        df["genre_list"] = df["genre"].apply(lambda x: [g.strip() for g in str(x).split(",")])
        df = df.explode("genre_list")
        df = df[df["genre_list"] != ""]

        genre_counts = df.groupby(["rating_year", "genre_list"]).size().reset_index(name="count")

        # choose top-10 genres overall
        top_genres = genre_counts.groupby("genre_list")["count"].sum().nlargest(10).index
        sub = genre_counts[genre_counts["genre_list"].isin(top_genres)]

        plt.figure(figsize=(14, 6))
        for g in top_genres:
            gdf = sub[sub["genre_list"] == g]
            plt.plot(gdf["rating_year"], gdf["count"], label=g)

        plt.legend()
        plt.title("Top genres over time (by user rating activity)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "genres_over_time.png"))
        plt.close()
        print("Saved: genres_over_time.png")

    # ============================================================
    # 4. HEATMAP OF USER ACTIVITY
    # ============================================================

    def plot_heatmap_activity(self):
        df = self.df.copy()
        heat = df.groupby(["username", "rating_year"]).size().unstack(fill_value=0)

        plt.figure(figsize=(14, 6))
        sns.heatmap(heat, cmap="Blues", annot=True, fmt="d")
        plt.title("User activity heatmap (ratings per year)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "activity_heatmap.png"))
        plt.close()
        print("Saved: activity_heatmap.png")

    # ============================================================
    # 5. GENDER DISTRIBUTION WITH USERNAMES
    # ============================================================

    def plot_demographics(self):
        df_users = self.df.drop_duplicates("username")

        # Gender
        plt.figure(figsize=(8, 6))
        sns.countplot(data=df_users, x="gender", hue="username")
        plt.title("Gender distribution (users shown in legend)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "gender_distribution.png"))
        plt.close()

        # Location
        plt.figure(figsize=(12, 6))
        sns.countplot(data=df_users, x="location", hue="username")
        plt.title("Location distribution (users shown in legend)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "location_distribution.png"))
        plt.close()

        print("Saved demographic distributions.")

    # ============================================================
    # 6. SCORE VS ACTIVITY
    # ============================================================

    def plot_score_vs_activity(self):
        df = self.df.copy()

        stats = df.groupby("username").agg(
            num_rated=("anime_id", "count"),
            mean_score=("my_score", "mean")
        ).reset_index()

        plt.figure(figsize=(12, 6))
        sns.scatterplot(data=stats, x="num_rated", y="mean_score", hue="username", s=150)
        plt.title("Score vs rating activity")
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "score_vs_activity.png"))
        plt.close()
        print("Saved: score_vs_activity.png")

    # ============================================================
    # 7. USER TIMELINES
    # ============================================================

    def plot_user_timelines(self):
        df = self.df.copy()
        df = df.sort_values("my_last_updated")

        plt.figure(figsize=(14, 8))
        for user in df["username"].unique():
            sub = df[df["username"] == user]
            plt.plot(sub["my_last_updated"], sub["my_score"], label=user)

        plt.legend()
        plt.title("User rating timelines")
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "user_timelines.png"))
        plt.close()
        print("Saved: user_timelines.png")

    # ============================================================
    # 8. GENRE HEATMAP PER USER
    # ============================================================

    def plot_genre_heatmap(self):
        df = self.df.copy()
        df["genre"] = df["genre"].fillna("")
        df["genre_list"] = df["genre"].apply(lambda x: [g.strip() for g in str(x).split(",")])
        df = df.explode("genre_list")
        df = df[df["genre_list"] != ""]

        heat = df.groupby(["username", "genre_list"]).size().unstack(fill_value=0)

        plt.figure(figsize=(16, 8))
        sns.heatmap(heat, cmap="Reds")
        plt.title("Genre preference heatmap (per user)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "genre_heatmap.png"))
        plt.close()
        print("Saved: genre_heatmap.png")
