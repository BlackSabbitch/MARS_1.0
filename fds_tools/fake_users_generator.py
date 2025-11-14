import os
import pandas as pd
from pathlib import Path
import duckdb
import numpy as np
from dataclasses import dataclass
import logging
from common_tools import CommonTools, PandasTools, log


@dataclass
class Constants:
    male_prob: float = 0.722
    female_prob: float = 0.278
    age_mean: float = 23.64
    age_std: float = 6.04
    infinum_age = 10
    supremum_age = 80


class FakeUsersGenerator:
    @staticmethod
    def load_country_pc_distribution_table_from_csv() -> pd.DataFrame:
        """Load country PC distribution data from a CSV file.
        Args:
            data_folder (str): The path to the folder containing the country PC distribution CSV file.

        Returns:
            pd.DataFrame: A data frame containing country PC distribution data.
        """
        purpose = "myanimelist_countries_distribution"
        file_path = CommonTools.get_paths()[purpose]
        data_frame = PandasTools.load_path(file_path).reset_index(drop=True)
        data_frame["country"] = data_frame["country"].str.strip().str.lower()
        data_frame["country"] = data_frame["country"].str.replace(" ", "_")
        data_frame["number"] = data_frame["number"].astype(int)
        data_frame.drop_duplicates(subset=["country"], keep='first', inplace=True)
        shape = data_frame.shape
        columns = data_frame.columns.tolist()
        log(f"Loaded {purpose}. Shape: {shape}. Columns: {columns}", tag="LOADED", level=logging.INFO)
        return data_frame

    @staticmethod
    def load_cities_location_and_population_table_from_csv() -> pd.DataFrame:
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

        purpose = "cities_population_and_location"
        file_path = CommonTools.load_csv_files_pathes()[purpose]
        data_frame = PandasTools.load_path(file_path).reset_index(drop=True)
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
        log(f"Loading {purpose}. Shape: {shape}. Columns: {columns}", tag="LOADED", level=logging.INFO)   
        return data_frame
    
    @classmethod
    def prepare_city_stats(cls) -> pd.DataFrame:
        temporary_columns = []

        country_mal_distr = cls.load_country_pc_distribution_table_from_csv()
        # compute country fraction among all users
        country_mal_distr["country_impact"] = country_mal_distr['number'].values / country_mal_distr['number'].sum()
        temporary_columns.append("number")
        log("Table 'myanimelist_countries_distribution' has been processed with new columns", tag="PROCESS", level=logging.DEBUG)

        cities_stats = cls.load_cities_location_and_population_table_from_csv()
        # keep only countries present in the MyAnimeList distribution table
        countries = set(country_mal_distr['country'].tolist())
        cities_stats = cities_stats[cities_stats['country'].isin(countries)].reset_index(drop=True)
        # compute city fraction among population of countries
        cities_stats["country_total_pop"] = cities_stats.groupby("country")["population"].transform("sum")
        cities_stats["city_pop_impact"] = cities_stats["population"] / cities_stats["country_total_pop"]
        temporary_columns.extend(["country_total_pop", "population"])
        log("Table 'cities_population_and_location' has been processed with new columns", tag="PROCESS", level=logging.DEBUG)
        # 1) merge
        cities_stats = cities_stats.merge(
            country_mal_distr,
            on="country",
            how="left",
            validate="many_to_one"   # one country → many cities
        )
        log("Tables have been merged", tag="PROCESS", level=logging.DEBUG)

        cities_stats["city_fraction_among_users"] = cities_stats["city_pop_impact"] * cities_stats["country_impact"]
        temporary_columns.extend(["city_pop_impact", "country_impact"])
        # clean up temporary columns
        cities_stats.drop(columns=temporary_columns, inplace=True)
        log("Target table has been prepared", tag="PROCESS", level=logging.DEBUG)

        return cities_stats

    @staticmethod
    def get_distinct_user_ids() -> pd.Series:
        table_names_pathes = CommonTools.load_csv_files_pathes()
        animelist_path = table_names_pathes['animelist']
        con = duckdb.connect()
        query = f"SELECT DISTINCT user_id FROM '{animelist_path}'"
        user_ids = con.execute(query).fetchdf()['user_id']
        log("Distinct user id's have been fetched", tag="PROCESS", level=logging.DEBUG)
        return user_ids

    @classmethod
    def generate_fake_user_profiles(cls, save: bool = False, save_rel_path: str = "", random_state: int = 42) -> pd.DataFrame:
        """
        Generate a table of fake user profiles with probabilistic city assignment,
        age and sex distributions.

        Parameters
        ----------
        random_state : int
            Seed for reproducibility.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['user_id', 'country', 'city', 'age', 'sex', 'location'].
        """
        # series of unique user IDs (pd.Series).
        user_ids = cls.get_distinct_user_ids()
        # world probability of every city (pd.DataFrame).
        # contain columns ['country', 'city', 'latitude', 'longitude', 'city_fraction_among_users']
        cities_stats = cls.prepare_city_stats()

        np.random.seed(random_state)
        N = len(user_ids)

        # --- sex assignment ---
        sexes = np.random.choice(['male', 'female'], size=N, p=[Constants.male_prob, Constants.female_prob])

        # --- age assignment ---
        ages = np.random.normal(loc=Constants.age_mean, scale=Constants.age_std, size=N)
        ages = np.clip(ages, a_min=Constants.infinum_age, a_max=Constants.supremum_age)  # при желании ограничиваем разумный диапазон
        ages = ages.astype(int)
        # --- city assignment ---
        # prepare arrays
        cities = cities_stats['city'].values
        countries = cities_stats['country'].values
        latitudes = cities_stats['latitude'].values
        longitudes = cities_stats['longitude'].values
        probs = cities_stats['city_fraction_among_users'].values
        # probs = probs / probs.sum()  # don't needed, in our case probabilities are already normalized

        # выбор индексов городов для всех пользователей
        city_idx = np.random.choice(len(cities), size=N, p=probs)

        assigned_cities = cities[city_idx]
        assigned_countries = countries[city_idx]
        assigned_latitudes = latitudes[city_idx]
        assigned_longitudes = longitudes[city_idx]

        # --- собираем DataFrame ---
        profiles = pd.DataFrame({
            'user_id': user_ids,
            'country': assigned_countries,
            'city': assigned_cities,
            'age': ages,
            'sex': sexes,
            'latitude': assigned_latitudes,
            'longitude': assigned_longitudes
        })
        log(f"Users profiles were updated with fake columns 'sex', 'age', 'latitude', 'longitude'", tag="GENERATED", level=logging.INFO)
        if save:
            save_rel_path = save_rel_path if save_rel_path else 'users'
            output_dir = Path("data") / save_rel_path
            output_dir.parent.mkdir(parents=True, exist_ok=True)

            output_file = output_dir / "profiles.csv"
            profiles.to_csv(output_file, index=False)
            print(f"Saved to {output_file.resolve()}")
            log(f"Users profiles were saved to {output_file}", tag="GENERATED", level=logging.INFO)
        return profiles
