import os
import pandas as pd
from pathlib import Path
import duckdb
import json
import numpy as np
from dataclasses import dataclass
from common_tools import CommonTools


@dataclass
class Constants:
    male_prob: float = 0.722
    female_prob: float = 0.278
    age_mean: float = 23.64
    age_std: float = 6.04
    infinum_age = 10
    supremum_age = 80


class FDSTools:
    @staticmethod
    def dump_the_file(file_purpose, config_file_path: str = "datasets.json"):
        datasets_metadata = CommonTools.load_json_config(config_file_path)
        data_folder = datasets_metadata[f"{file_purpose}"]['folder']
        content = os.listdir(data_folder)
        assert len(content) == 1, "Error: More then 1 MAL distribution file!"
        entity = content[0]
        if os.path.isfile(f"{data_folder}/" + entity) and Path(entity).suffix == ".csv":
            print(f"[FOUND]: {entity}")
            return data_folder, entity
        else:
            raise ValueError(f"No source file for {file_purpose}") 

    @classmethod
    def load_country_pc_distribution_table_from_csv(cls, config_file_path: str = "datasets.json", file_purpose: str = "country_distribution") -> pd.DataFrame:
        """Load country PC distribution data from a CSV file.
        Args:
            data_folder (str): The path to the folder containing the country PC distribution CSV file.

        Returns:
            pd.DataFrame: A data frame containing country PC distribution data.
        """
        data_folder, source_file = cls.dump_the_file(file_purpose, config_file_path)
        entity_name = Path(source_file).stem
        data_frame = pd.read_csv(f"{data_folder}/{source_file}").reset_index(drop=True)
        data_frame["country"] = data_frame["country"].str.strip().str.lower()
        data_frame["country"] = data_frame["country"].str.replace(" ", "_")
        data_frame["number"] = data_frame["number"].astype(int)
        data_frame.drop_duplicates(subset=["country"], keep='first', inplace=True)
        shape = data_frame.shape
        columns = data_frame.columns.tolist()
        print(f"[LOADED] {entity_name}. Shape: {shape}. Columns: {columns}")   
        return data_frame

    @classmethod
    def load_cities_location_and_population_table_from_csv(cls, config_file_path: str = "datasets.json", file_purpose: str = "cities_population_and_location") -> pd.DataFrame:
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

        data_folder, source_file = cls.dump_the_file(file_purpose, config_file_path)
        entity_name = Path(source_file).stem
        data_frame = pd.read_csv(f"{data_folder}/{source_file}").reset_index(drop=True)
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
    def prepare_city_stats(cls) -> pd.DataFrame:
        temporary_columns = []

        country_mal_distr = cls.load_country_pc_distribution_table_from_csv()
        # compute country fraction among all users
        country_mal_distr["country_impact"] = country_mal_distr['number'].values / country_mal_distr['number'].sum()
        temporary_columns.append("number")

        cities_stats = cls.load_cities_location_and_population_table_from_csv()
        # keep only countries present in the MyAnimeList distribution table
        countries = set(country_mal_distr['country'].tolist())
        cities_stats = cities_stats[cities_stats['country'].isin(countries)].reset_index(drop=True)
        # compute city fraction among population of countries
        cities_stats["country_total_pop"] = cities_stats.groupby("country")["population"].transform("sum")
        cities_stats["city_pop_impact"] = cities_stats["population"] / cities_stats["country_total_pop"]
        temporary_columns.extend(["country_total_pop", "population"])
        # 1) merge
        cities_stats = cities_stats.merge(
            country_mal_distr,
            on="country",
            how="left",
            validate="many_to_one"   # one country → many cities
        )

        cities_stats["city_fraction_among_users"] = cities_stats["city_pop_impact"] * cities_stats["country_impact"]
        temporary_columns.extend(["city_pop_impact", "country_impact"])
        # clean up temporary columns
        cities_stats.drop(columns=temporary_columns, inplace=True)

        return cities_stats

    @classmethod
    def get_distinct_user_ids(cls):
        table_names_pathes = CommonTools.load_csv_files_pathes()
        animelist_path = table_names_pathes['animelist']
        con = duckdb.connect()
        query = f"SELECT DISTINCT user_id FROM '{animelist_path}'"
        user_ids = con.execute(query).fetchdf()['user_id']
        return user_ids

    @classmethod
    def generate_fake_user_profiles(cls, random_state: int = 42) -> pd.DataFrame:
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
        assigned_locations = list(zip(latitudes[city_idx], longitudes[city_idx]))

        # --- собираем DataFrame ---
        profiles = pd.DataFrame({
            'user_id': user_ids,
            'country': assigned_countries,
            'city': assigned_cities,
            'age': ages,
            'sex': sexes,
            'location': assigned_locations
        })

        return profiles

    @classmethod
    def find_not_honest_users(cls):
        # Пути к файлам
        table_names_pathes = cls.load_csv_files_pathes()
        animelist_path = table_names_pathes['animelist']
        anime_path = table_names_pathes['anime']
        rating_complete_path = table_names_pathes['rating_complete']

        # подключаемся к временной in-memory базе
        con = duckdb.connect(database=':memory:')

        # Загружаем только нужные поля (всё остальное не читаем)
        query_find_bad = f"""
        SELECT 
            l.user_id,
            l.anime_id,
            l.rating,
            l.watched_episodes,
            a.Episodes
        FROM read_csv_auto('{animelist_path}', 
                        columns={{'user_id':'BIGINT', 'anime_id':'BIGINT', 'rating':'INT', 'watching_status':'INT', 'watched_episodes':'INT'}})
                AS l
        JOIN read_csv_auto('{anime_path}', 
                        columns={{'MAL_id':'BIGINT', 'Episodes':'INT'}})
                AS a
        ON l.anime_id = a.MAL_id
        WHERE 
            l.watching_status = 2
            AND l.rating != 0
            AND l.watched_episodes < a.Episodes
        """

        # Загружаем только нужные строки
        df_incomplete_votes = con.execute(query_find_bad).fetchdf()
        return df_incomplete_votes
    
    @classmethod
    def clean_for_not_honest_users(cls):
        df_incomplete_votes = cls.find_not_honest_users()
        # Загружаем rating_complete (тут уже можно pandas, он небольшой)
        table_names_pathes = cls.load_csv_files_pathes()
        rating_complete_path = table_names_pathes['rating_complete']
        df_rating_complete = pd.read_csv(rating_complete_path)

        # Формируем множество (user_id, anime_id) для удаления
        pairs_to_remove = set(zip(df_incomplete_votes.user_id, df_incomplete_votes.anime_id))

        # Фильтруем
        mask = ~df_rating_complete.apply(lambda x: (x.user_id, x.anime_id) in pairs_to_remove, axis=1)
        df_clean = df_rating_complete[mask].reset_index(drop=True)

        # Сохраняем
        df_clean.to_csv('rating_complete_cleaned.csv', index=False)
        print(f"Удалено строк: {len(df_rating_complete) - len(df_clean)}")
