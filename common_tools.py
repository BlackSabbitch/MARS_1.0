import os
import pandas as pd
from pathlib import Path
import duckdb


class CommonTools:
    @staticmethod
    def load_tables_from_csv(data_folder: str, mode='duckdb') -> dict:
        """Load all CSV files from a specified folder into a dictionary of data frames.
        Args:
            data_folder (str): The path to the folder containing CSV files.
            mode (str): The mode of loading data frames, either 'pandas' or 'duckdb'.
        Returns:
            dict: A dictionary where keys are file names (without extension) and values are data frames or duckdb objects.
        """
        data_frames = {}
        for entity in os.listdir(data_folder):
            if os.path.isfile(f"{data_folder}/" + entity) and Path(entity).suffix == ".csv":
                # if entity == "animelist.csv":
                #     continue
                print(f"[FOUND]: {entity}")
                entity_name = Path(entity).stem
                if mode == 'pandas':
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
