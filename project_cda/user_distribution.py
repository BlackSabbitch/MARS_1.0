import pandas as pd
import matplotlib.pyplot as plt
import os
from cda_project.user_data import UserData


class UserDistribution:

    @staticmethod
    def plot_distributions():
        print(">>> Initializing UserDistribution...")

        # Load paths from UserData
        user_data = UserData()
        data_dir = user_data.cda_data_dir

        print(f">>> Data directory: {data_dir}")

        # ============================================================
        # Check that required CSV files exist
        # ============================================================
        required_files = [
            "selected_users.csv",
            "selected_users_animelist.csv",
            "selected_anime.csv"
        ]

        for file in required_files:
            path = os.path.join(data_dir, file)
            print(f"Checking file: {path} ... exists={os.path.exists(path)}")
            if not os.path.exists(path):
                print(f"ERROR: Required file is missing: {path}")
                return

        # ============================================================
        # Load datasets
        # ============================================================
        print(">>> Reading selected_users.csv ...")
        users = pd.read_csv(os.path.join(data_dir, "selected_users.csv"))
        print("    Done. Rows:", len(users))

        print(">>> Reading selected_users_animelist.csv ...")
        lists = pd.read_csv(os.path.join(data_dir, "selected_users_animelist.csv"))
        print("    Done. Rows:", len(lists))

        print(">>> Reading selected_anime.csv ...")
        anime = pd.read_csv(os.path.join(data_dir, "selected_anime.csv"))
        print("    Done. Rows:", len(anime))

        # ============================================================
        # Convert dates
        # ============================================================
        print(">>> Converting date formats...")
        lists["my_last_updated"] = pd.to_datetime(lists["my_last_updated"], errors="coerce")
        anime["aired_from_year"] = pd.to_numeric(anime["aired_from_year"], errors="coerce")
        print("    Date conversion done.")

        # ============================================================
        # Merge lists with anime metadata
        # ============================================================
        print(">>> Merging user lists with anime metadata...")
        merged = lists.merge(
            anime[["anime_id", "title", "genre", "aired_from_year"]],
            on="anime_id",
            how="left"
        )
        print("    Merge complete. Rows:", len(merged))

        # ============================================================
        # 1. Distribution of user scores
        # ============================================================
        print(">>> Plotting score distribution...")
        plt.figure(figsize=(8, 5))
        merged["my_score"].replace(0, pd.NA).dropna().astype(int).hist(bins=10)
        plt.title("Distribution of User Ratings")
        plt.xlabel("Score")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()
        print("    Score plot done.")

        # ============================================================
        # 2. Anime watched by release year
        # ============================================================
        print(">>> Plotting anime release year distribution...")
        plt.figure(figsize=(10, 5))
        merged.dropna(subset=["aired_from_year"])["aired_from_year"].astype(int).hist(bins=30)
        plt.title("Distribution of Watched Anime by Release Year")
        plt.xlabel("Anime Release Year")
        plt.ylabel("Count of Watched Titles")
        plt.tight_layout()
        plt.show()
        print("    Release year plot done.")

        # ============================================================
        # 3. Preferred genres by year
        # ============================================================
        print(">>> Processing genre information...")
        merged["genre"] = merged["genre"].fillna("")
        merged["genre_list"] = merged["genre"].apply(
            lambda x: [g.strip() for g in str(x).split(",") if g.strip()]
        )

        print(">>> Exploding genres...")
        exploded = merged.explode("genre_list")
        print("    Exploded rows:", len(exploded))

        print(">>> Grouping genres by year...")
        genre_by_year = exploded.groupby(["aired_from_year", "genre_list"]).size().reset_index(name="count")
        print("    Genre grouping done.")

        # Top 10 genres
        top_genres = genre_by_year.groupby("genre_list")["count"].sum().nlargest(10).index
        subset = genre_by_year[genre_by_year["genre_list"].isin(top_genres)]

        print(">>> Plotting genres per year...")
        plt.figure(figsize=(12, 6))
        for g in top_genres:
            df = subset[subset["genre_list"] == g]
            plt.plot(df["aired_from_year"], df["count"], label=g)

        plt.title("Top Genres over Years for Selected Users")
        plt.xlabel("Year")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        plt.show()
        print("    Genre plot done.")

        # ============================================================
        # 4. Demographic distribution
        # ============================================================
        print(">>> Plotting gender distribution...")
        plt.figure(figsize=(6, 4))
        users["gender"].value_counts().plot(kind="bar")
        plt.title("Gender Distribution")
        plt.xlabel("Gender")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()
        print("    Gender plot done.")

        # Age distribution
        print(">>> Plotting age distribution...")
        if "birth_date" in users.columns:
            users["birth_date"] = pd.to_datetime(users["birth_date"], errors="coerce")
            users["age"] = (pd.Timestamp.now() - users["birth_date"]).dt.days / 365.25

            plt.figure(figsize=(8, 5))
            users["age"].dropna().astype(int).hist(bins=20)
            plt.title("Age Distribution of Selected Users")
            plt.xlabel("Age")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.show()
            print("    Age plot done.")

        # Locations
        print(">>> Plotting location distribution...")
        plt.figure(figsize=(8, 5))
        users["location"].fillna("Unknown").value_counts().head(10).plot(kind="bar")
        plt.title("Top 10 Locations")
