import os
import pandas as pd
from pathlib import Path
import duckdb
import json
import numpy as np


class CommonTools:
    @staticmethod
    def load_json_config(config_file_path: str = "datasets.json") -> dict:
        """Parse a JSON configuration file and return its contents as a dictionary.
        Args:
            file_path (str): The path to the JSON configuration file.
        Returns:
            dict: A dictionary containing the parsed JSON configuration.
        """
        with open(config_file_path) as f:
            datasets = json.load(f)
        return datasets

    @classmethod
    def load_csv_files_pathes(cls, config_file_path: str = "datasets.json") -> dict[str, str]:
        """Load all CSV file paths from a specified folder into a dictionary.
        Args:
            config_file_path (str): The path to the JSON configuration file.
        Returns:
            dict: A dictionary where keys are file names (without extension) and values are file paths.
        """
        files = {}
        datasets_metadata = cls.load_json_config(config_file_path)
        for key, value in datasets_metadata.items():
            folder = value['folder']
            for entity in os.listdir(folder):
                if os.path.isfile(f"{folder}/" + entity) and Path(entity).suffix == ".csv":
                    files[Path(entity).stem] = f"{folder}/{entity}"
        return files

    @classmethod
    def load_anime_tables_from_csv(cls, config_file_path: str = "datasets.json", mode='duckdb') -> dict:
        """Load all CSV files from a specified folder into a dictionary of data frames.
        Args:
            data_folder (str): The path to the folder containing CSV files.
            mode (str): The mode of loading data frames, either 'pandas' or 'duckdb'.
        Returns:
            dict: A dictionary where keys are file names (without extension) and values are data frames or duckdb objects.
        """
        datasets_metadata = cls.load_json_config(config_file_path)
        data_folder = datasets_metadata['anime_ranks']['folder']
        data_frames = {}
        for entity in os.listdir(data_folder):
            if os.path.isfile(f"{data_folder}/" + entity) and Path(entity).suffix == ".csv":
                print(f"[FOUND]: {entity}")
                entity_name = Path(entity).stem
                if mode == 'pandas':
                    # too big for pandas
                    # also the cleaned version of the animelist is rating_complete.csv
                    if entity == "animelist.csv":
                        continue
                    data_frames[entity_name] = pd.read_csv(f"{data_folder}/{entity}")
                    shape = data_frames[entity_name].shape
                    columns = data_frames[entity_name].columns.tolist()
                    print(f"[LOADED] {entity_name} in {mode} mode. Shape: {shape}. Columns: {columns}")                
                elif mode == 'duckdb':
                    duckdb.query(f"""
                               CREATE VIEW {entity_name}
                               AS SELECT *
                               FROM read_csv_auto('{data_folder}/{entity}')""")
                    rel = duckdb.query(f"SELECT * FROM {entity_name}")
                    rows, cols = duckdb.query(f"""
                                            SELECT (SELECT COUNT(*)
                                            FROM {entity_name}),
                                            (SELECT COUNT(*) FROM
                                            pragma_table_info('{entity_name}'))""").fetchone()
                    data_frames[entity_name] = duckdb.query(f"""SELECT * FROM '{data_folder}/{entity}'""")
                    print(f"[LOADED] {entity_name} in {mode} mode. Shape: {rows, cols}. Columns: {rel.columns}")
        return data_frames

    @staticmethod
    def clean_up_duckdb_views():
        """Cleans up all duckdb views created during the data loading process."""
        views = duckdb.query("SHOW VIEWS").fetchall()
        for view in views:
            view_name = view[0]
            duckdb.query(f"DROP VIEW IF EXISTS {view_name}")
            print(f"[DROPPED VIEW]: {view_name}")

    @classmethod
    def load_country_pc_distribution_table_from_csv(cls, config_file_path: str = "datasets.json") -> pd.DataFrame:
        """Load country PC distribution data from a CSV file.
        Args:
            data_folder (str): The path to the folder containing the country PC distribution CSV file.

        Returns:
            pd.DataFrame: A data frame containing country PC distribution data.
        """
        datasets_metadata = cls.load_json_config(config_file_path)
        data_folder = datasets_metadata['country_distribution']['folder']
        for entity in os.listdir(data_folder):
            if os.path.isfile(f"{data_folder}/" + entity) and Path(entity).suffix == ".csv":
                print(f"[FOUND]: {entity}")
                entity_name = Path(entity).stem
                data_frame = pd.read_csv(f"{data_folder}/{entity}").reset_index(drop=True)
                data_frame["country"] = data_frame["country"].str.strip().str.lower()
                data_frame["country"] = data_frame["country"].str.replace(" ", "_")
                data_frame["number"] = data_frame["number"].astype(int)
                data_frame.drop_duplicates(subset=["country"], keep='first', inplace=True)
                shape = data_frame.shape
                columns = data_frame.columns.tolist()
                print(f"[LOADED] {entity_name}. Shape: {shape}. Columns: {columns}")   
        return data_frame

    @classmethod
    def load_cities_location_and_population_table_from_csv(cls, config_file_path: str = "datasets.json", dry: bool = True) -> pd.DataFrame:
        """Load city location and population data from a CSV file.
        Args:
            data_folder (str): The path to the folder containing the city location and population CSV file
            dry (bool): If True, only stores country,location_name,latitude,longitude columns and renames to country,city,latitude,longitude.
        Returns:
            pd.DataFrame: A data frame containing city location and population data.
        """
        map_of_countries_corrections = {
            "Venezuela, Bolivarian Rep. of": "venezuela",
            "Moldova, Republic of": "moldova",
            "Brunei Darussalam": "brunei",
            "Taiwan, China": "taiwan",
            "Russian Federation": "russia",
            "Libyan Arab Jamahiriya": "libya",
            "Hong Kong, China": "hong_kong",
            "Korea, Republic of": "south_korea",
            "Cape Verde": "cabo_verde",
            "Viet Nam": "vietnam",
        }

        datasets_metadata = cls.load_json_config(config_file_path)
        data_folder = datasets_metadata['cities_population_and_location']['folder']
        for entity in os.listdir(data_folder):
            if os.path.isfile(f"{data_folder}/" + entity) and Path(entity).suffix == ".csv":
                print(f"[FOUND]: {entity}")
                entity_name = Path(entity).stem
                data_frame = pd.read_csv(f"{data_folder}/{entity}").reset_index(drop=True)
                if dry:
                    data_frame = data_frame[["Country name EN", "ASCII Name", "Population", "Latitude", "Longitude"]]
                    data_frame.columns = ["country", "city", "population", "latitude", "longitude"]

                    for wrong_country_name, correct_country_name in map_of_countries_corrections.items():
                        data_frame['country'] = data_frame['country'].str.replace(wrong_country_name, correct_country_name)

                    data_frame["country"] = data_frame["country"].str.strip().str.lower()
                    data_frame["country"] = data_frame["country"].str.replace(" ", "_")
                    data_frame["city"] = data_frame["city"].str.strip().str.lower()
                    data_frame["city"] = data_frame["city"].str.replace(" ", "_")
                    data_frame["latitude"] = data_frame["latitude"].astype(float)
                    data_frame["longitude"] = data_frame["longitude"].astype(float)
                    data_frame["population"] = data_frame["population"].astype(int)

                shape = data_frame.shape
                columns = data_frame.columns.tolist()
                print(f"[LOADED] {entity_name}. Shape: {shape}. Columns: {columns}")   
        return data_frame
    
    @classmethod
    def prepare_city_tables(cls) -> pd.DataFrame:
        country_mal_distr = cls.load_country_pc_distribution_table_from_csv()
        cities_stats = cls.load_cities_location_and_population_table_from_csv()

        countries = set(country_mal_distr['country'].tolist())
        cities_stats = cities_stats[cities_stats['country'].isin(countries)].reset_index(drop=True)
        cities_population_sum = cities_stats['population'].sum()
        cities_fraction = cities_stats['population'] / cities_population_sum
        
        # 1) merge
        # cities_stats = cities_stats.merge(
        #     country_mal_distr,
        #     on="country",
        #     how="left",
        #     validate="many_to_one"   # one country → many cities
        # )
        # 2) sum population over available cities per country
        # cities_stats["country_total_pop"] = cities_stats.groupby("country")["population"].transform("sum")
        # 3) distribute
        # cities_stats["city_users_float"] = (
        #     cities_stats["population"] / cities_stats["country_total_pop"] * cities_stats["number"]
        # )
        # 4) balanced rounding
        # cities_stats = cls.fix_rounding(cities_stats)
        # cols_to_drop = ["number", "city_users_float", "population", "country_total_pop", "base", "frac", "diff", "country_total_base"]
        # 5) remove temporary columns
        # cities_stats = cities_stats.drop(
        #     columns=[c for c in cols_to_drop if c in cities_stats.columns],
        #     errors="ignore"
        #     )
        # 6) remove cities with 0 users
        # cities_stats = cities_stats[cities_stats["city_users"] > 0]
        return cities_stats

    @staticmethod
    def fix_rounding(df):
        """
        Adjust rounded values so that their total sum equals a given target.

        This function performs integer rounding on an array of numeric values
        and then fixes the rounding error by distributing ±1 adjustments
        to elements with the largest fractional parts (for positive correction)
        or smallest fractional parts (for negative correction).

        Parameters
        ----------
        values : array-like of float
            Original fractional values whose sum is approximately equal to `target_sum`.
        target_sum : int
            The desired integer sum after rounding.

        Returns
        -------
        list of int
            Integer values whose sum is exactly `target_sum`, obtained by:
            1. Standard rounding
            2. Minimal redistribution of ±1 to correct total error

        Notes
        -----
        - Guarantees exact integer total equal to `target_sum`.
        - Minimizes L1 deviation from naïve rounding.
        - At most |target_sum − sum(round(values))| elements are modified.
        - Each modified value differs from round(x) by ±1.

        Examples
        --------
        >>> fix_rounding([3.4, 2.6, 1.2, 1.0], 9)
        [3, 4, 1, 1]

        """
        # Round, but keep the fractional part
        df["base"] = np.floor(df["city_users_float"]).astype(int)
        df["frac"] = df["city_users_float"] - df["base"]

        # How much is missing after rounding down
        df["country_total_base"] = df.groupby("country")["base"].transform("sum")
        df["diff"] = df["number"] - df["country_total_base"]

        result = []

        for country, sub in df.groupby("country"):
            sub = sub.copy()
            k = int(sub["diff"].iloc[0])  # how many to add
            # Sort by descending fractional part
            sub = sub.sort_values("frac", ascending=False)

            # Add 1 to the first k cities
            sub["city_users"] = sub["base"]
            if k > 0:
                sub.iloc[:k, sub.columns.get_loc("city_users")] += 1

            result.append(sub)

        return pd.concat(result).sort_index()
