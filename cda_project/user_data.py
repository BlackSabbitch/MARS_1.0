import os
import pandas as pd

class UserData:
    """
    Extracts users who:
        • joined between 2004-2006
        • rated more than a defined number of anime (default >150)
        • have long-term rating activity (default ≥10 years between first and last rating)
    """
    VALID_COUNTRIES = {
        "United States", "USA", "Canada", "India", "Portugal", "Brazil", "United Kingdom",
        "UK", "Germany", "France", "Spain", "Russia", "Ukraine", "Poland", "Romania", 
        "Italy", "Finland", "Sweden", "Norway", "Denmark", "Netherlands", "Belgium",
        "Australia", "New Zealand", "Mexico", "Argentina", "Chile", "Peru", "Thailand",
        "Malaysia", "Indonesia", "Philippines", "Vietnam", "Japan", "South Korea",
        "China", "Taiwan", "Hong Kong", "Singapore"
    }
    def __init__(self,
             data_dir=r"C:\MariaSamosudova\Projects\UNIVER\ADB\Project\MARS_1.0\data\anime_azathoth42",
             cda_data_dir=r"C:\MariaSamosudova\Projects\UNIVER\ADB\Project\MARS_1.0\data\data_cda"):

        self.data_dir = data_dir
        self.cda_data_dir = cda_data_dir

        # File paths
        self.users_path = self.data_dir + r"\users_cleaned.csv"
        self.lists_path = self.data_dir + r"\animelists_cleaned.csv"
        self.anime_path = self.data_dir + r"\anime_cleaned.csv"

        # Loaded data frames
        self.users = None
        self.lists = None



    def _load_data(self):
        """Load users and animelist tables from disk."""
        self.users = pd.read_csv(self.users_path)
        self.lists = pd.read_csv(self.lists_path)

        # Convert date columns
        self.users["join_date"] = pd.to_datetime(self.users["join_date"], errors="coerce")
        self.lists["my_last_updated"] = pd.to_datetime(self.lists["my_last_updated"], errors="coerce")

    def get_user_data(self, min_year=2004, max_year=2006, min_rated=150, min_span=10):
        """
        Returns users who meet all criteria:
            • join_date between min_year and max_year
            • number of rated anime > min_rated
            • rating activity span ≥ min_span years
        """

        self._load_data()

        # Filter by join year
        early_users = self.users[
            (self.users["join_date"].dt.year >= min_year) &
            (self.users["join_date"].dt.year <= max_year)
        ]

        # Keep only ratings where my_score != 0
        rated = self.lists[self.lists["my_score"] != 0]

        # Count the number of rated anime per user
        ratings_count = rated.groupby("username")["anime_id"].count().rename("num_rated")

        # Calculate first and last rating timestamps
        activity = rated.groupby("username")["my_last_updated"].agg(["min", "max"])
        activity["year_span"] = activity["max"].dt.year - activity["min"].dt.year

        # Merge all criteria
        merged = (
            early_users
            .merge(ratings_count, left_on="username", right_index=True, how="inner")
            .merge(activity, left_on="username", right_index=True, how="inner")
        )

        # Final filtering
        final_users = merged[
            (merged["num_rated"] > min_rated) &
            (merged["year_span"] >= min_span)
        ]

        return final_users
    
    def save_user_data(self, df, filename="selected_users_initial.csv"):
        save_dir = self.cda_data_dir + "\\" + filename
        df.to_csv(save_dir, index=False, encoding="utf-8")
        return save_dir
    
    def extract_full_user_info(self, selected_users_df):

        import time

        usernames = set(selected_users_df["username"].unique())

        print(f"Extracting info for {len(usernames)} users...")

        # -----------------------
        # Load USERS (small file)
        # -----------------------
        users = pd.read_csv(self.users_path)
        users_sub = users[users["username"].isin(usernames)]

        users_out = os.path.join(self.cda_data_dir, "selected_users.csv")
        users_sub.to_csv(users_out, index=False)

        # -----------------------
        # Load LISTS (very large) – read in chunks
        # -----------------------
        print("Filtering animelists_cleaned.csv ...")

        lists_out_path = os.path.join(self.cda_data_dir, "selected_users_animelist.csv")
        lists_out = open(lists_out_path, "w", encoding="utf-8")

        header_written = False
        chunksize = 200_000
        total_rows = 0
        matched_rows = 0
        t0 = time.time()

        for i, chunk in enumerate(pd.read_csv(self.lists_path, chunksize=chunksize)):
            total_rows += len(chunk)

            filtered = chunk[chunk["username"].isin(usernames)]
            matched_rows += len(filtered)

            if not header_written:
                filtered.to_csv(lists_out, index=False)
                header_written = True
            else:
                filtered.to_csv(lists_out, index=False, header=False)

            if i % 10 == 0:
                print(f"Processed {total_rows:,} rows, matched {matched_rows:,}")

        lists_out.close()
        print(f"Done in {time.time()-t0:.2f} sec")

        # -----------------------
        # Load filtered lists
        # -----------------------
        lists_sub = pd.read_csv(lists_out_path)

        # -----------------------
        # Filter anime
        # -----------------------
        anime_ids = lists_sub["anime_id"].unique()
        anime = pd.read_csv(self.anime_path)
        anime_sub = anime[anime["anime_id"].isin(anime_ids)]

        anime_out = os.path.join(self.cda_data_dir, "selected_anime.csv")
        anime_sub.to_csv(anime_out, index=False)

        print("Saved:")
        print("Users:", users_out)
        print("User anime lists:", lists_out_path)
        print("Anime details:", anime_out)

        return users_sub, lists_sub, anime_sub

    
    def show_top_users(self, top_n=20):
        # Load user and activity data
        users = pd.read_csv(self.cda_data_dir + r"\selected_users.csv")
        lists = pd.read_csv(self.cda_data_dir + r"\selected_users_animelist.csv")

        # Convert dates
        users["birth_date"] = pd.to_datetime(users["birth_date"], errors="coerce")
        users["join_date"] = pd.to_datetime(users["join_date"], errors="coerce")
        users["last_online"] = pd.to_datetime(users["last_online"], errors="coerce")

        # Calculate age
        users["age"] = (pd.Timestamp.now() - users["birth_date"]).dt.days / 365.25

        # Only rows where user rated anime
        rated = lists[lists["my_score"] != 0]

        # Count rated anime
        rating_count = rated.groupby("username")["anime_id"].count().rename("num_rated")

        # Count all user list entries
        watch_count = lists.groupby("username")["anime_id"].count().rename("num_records")

        # Merge
        merged = (
            users
            .merge(rating_count, left_on="username", right_index=True, how="left")
            .merge(watch_count, left_on="username", right_index=True, how="left")
        )

        # Sort by number of rated anime
        top_users = merged.sort_values("num_rated", ascending=False).head(top_n)

        # Columns to display
        columns = [
            "username",
            "gender",
            "location",
            "age",
            "join_date",
            "last_online",
            "num_rated",
            "num_records",
            "stats_mean_score"
        ]

        print("\n=== Top Active Users ===")
        return top_users[columns]
    
   

    def normalize_country(raw):
        if pd.isna(raw):
            return None
        parts = [p.strip() for p in str(raw).split(",")]
        return parts[-1]   # last part is usually the country


    def filter_top(self, top_n=20):
        # Step 1 — get top users
        top = self.top_users(top_n).copy()

        # Step 2 — clean the country field
        top["norm_country"] = top["location"].apply(normalize_country)

        # Step 3 — filter by known valid countries
        top_clean = top[top["norm_country"].isin(VALID_COUNTRIES)].copy()

        # Step 4 — update usernames list
        self.top_usernames = set(top_clean["username"].tolist())

        # Step 5 — filter all datasets
        self.users_top = self.users[self.users["username"].isin(self.top_usernames)]
        self.lists_top = self.lists[self.lists["username"].isin(self.top_usernames)]
        self.merged_top = self.merged[self.merged["username"].isin(self.top_usernames)]
        self.exploded_top = self.exploded[self.exploded["username"].isin(self.top_usernames)]

        print("=== REAL TOP USERS (after country filtering) ===")
        print(top_clean[[
            "username", "gender", "location", "norm_country",
            "num_rated", "num_records", "stats_mean_score"
        ]])

        return top_clean
    
    def clean_users(self, df):
        """
        Cleans users by:
            • realistic country
            • activity recency (last_online >= 2017)
        """

        df = df.copy()

        # Normalize location → extract country
        df["norm_country"] = df["location"].apply(UserData.normalize_country)

        # Convert last_online
        df["last_online"] = pd.to_datetime(df["last_online"], errors="coerce")

        # Criteria
        valid_countries = df["norm_country"].isin(UserData.VALID_COUNTRIES)
        recent_activity = df["last_online"] >= pd.Timestamp("2017-01-01")

        cleaned = df[valid_countries & recent_activity].copy()

        print("\n=== USER CLEANING REPORT ===")
        print("Total initial users:", len(df))
        print("Valid countries:", valid_countries.sum())
        print("Recent activity (>=2017):", recent_activity.sum())
        print("Remaining after both filters:", len(cleaned))
        print("Removed:", len(df) - len(cleaned))

        print("\nRemaining users:")
        print(cleaned[["username", "location", "norm_country", "last_online"]])

        return cleaned


    def select_top20(self, cleaned_users_df, lists_df):
        """Select top 20 most active users from already cleaned users."""

        # Ensure date fields
        cleaned_users_df["birth_date"] = pd.to_datetime(cleaned_users_df["birth_date"], errors="coerce")
        cleaned_users_df["join_date"] = pd.to_datetime(cleaned_users_df["join_date"], errors="coerce")
        cleaned_users_df["last_online"] = pd.to_datetime(cleaned_users_df["last_online"], errors="coerce")

        # Compute age
        cleaned_users_df["age"] = (pd.Timestamp.now() - cleaned_users_df["birth_date"]).dt.days / 365.25

        # Filter ratings
        rated = lists_df[lists_df["my_score"] != 0]
        rating_count = rated.groupby("username")["anime_id"].count().rename("num_rated")
        watch_count = lists_df.groupby("username")["anime_id"].count().rename("num_records")

        # Merge
        merged = (
            cleaned_users_df
            .merge(rating_count, left_on="username", right_index=True, how="left")
            .merge(watch_count, left_on="username", right_index=True, how="left")
        )

        # Sort by num_rated
        merged = merged.sort_values("num_rated", ascending=False)

        # Select top20
        top20 = merged.head(20).copy()

        print("\n=== TOP 20 USERS (REAL COUNTRIES ONLY) ===")
        print(top20[
            ["username", "gender", "location", "age",
            "join_date", "last_online",
            "num_rated", "num_records", "stats_mean_score"]
        ])

        return top20

    
    def save_top20_full_info(self, top20_df):
        """
        Save full dataset for top-20 cleaned users:
            • selected_users_top20.csv
            • selected_users_top20_animelist.csv
            • selected_anime_top20.csv
        """

        usernames = top20_df["username"].unique()

        # Load selected 86 datasets
        users = pd.read_csv(os.path.join(self.cda_data_dir, "selected_users.csv"))
        lists = pd.read_csv(os.path.join(self.cda_data_dir, "selected_users_animelist.csv"))
        anime = pd.read_csv(os.path.join(self.cda_data_dir, "selected_anime.csv"))

        # 1. Users
        users20 = users[users["username"].isin(usernames)]
        users_out = os.path.join(self.cda_data_dir, "selected_users_top20.csv")
        users20.to_csv(users_out, index=False)

        # 2. Animelists
        lists20 = lists[lists["username"].isin(usernames)]
        lists_out = os.path.join(self.cda_data_dir, "selected_users_top20_animelist.csv")
        lists20.to_csv(lists_out, index=False)

        # 3. Anime details
        anime_ids = lists20["anime_id"].unique()
        anime20 = anime[anime["anime_id"].isin(anime_ids)]
        anime_out = os.path.join(self.cda_data_dir, "selected_anime_top20.csv")
        anime20.to_csv(anime_out, index=False)

        print("\nSaved TOP-20 datasets:")
        print("Users:", users_out)
        print("User anime lists:", lists_out)
        print("Anime details:", anime_out)

        return users20, lists20, anime20





