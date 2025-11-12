import os
import pandas as pd
from pathlib import Path
import duckdb
import json
import numpy as np
from dataclasses import dataclass


@dataclass
class Constants:
    male_prob: float = 0.722
    female_prob: float = 0.278
    age_mean: float = 23.64
    age_std: float = 6.04
    infinum_age = 10
    supremum_age = 80


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
