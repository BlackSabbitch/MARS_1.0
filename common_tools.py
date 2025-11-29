# deprecated

import os
import pandas as pd
from pathlib import Path
import duckdb
import json
import logging
from typing import Dict, Optional, Any

DATASET_CONFIG_FILE = "datasets.json"
# ============================================================
#               Logging Setup with TAG Support  QQQ
# ============================================================

class TagFormatter(logging.Formatter):
    """
    Custom formatter that supports a `tag` attribute.
    If a tag is present, logs messages as:  [TAG] message
    Otherwise:                              [LEVEL] message
    """

    def format(self, record: logging.LogRecord) -> str:
        tag = getattr(record, "tag", None)
        prefix = f"[{tag}]" if tag else f"[{record.levelname}]"
        record.msg = f"{prefix} {record.msg}"
        return super().format(record)

logger = logging.getLogger("data_tools")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = TagFormatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def log(msg: str, *, tag: str | None = None, level=logging.INFO):
    """
    EXAMPLES:
    log(f"Loading config from {config_file_path}", tag="FOUND", level=logging.DEBUG)    | [DEBUG] --> [FOUND]
    log(f"Config loaded: {config_file_path}", tag="LOADED") | [INFO] --> [LOADED]
    log(f"{purpose} is too large for pandas → skipped", tag="SKIPPED", level=logging.WARNING)   | [WARNING] --> [SKIPPED]
    """
    logger.log(level, msg, extra={"tag": tag})

# ============================================================
#                     Common Tools
# ============================================================

class CommonTools:
    """
    Reads datasets.json and provides easy access to dataset paths.
    """
    
    @staticmethod
    def load_config(config_file_path: str = DATASET_CONFIG_FILE) -> dict:
        log(f"Loading config: {config_file_path}", tag="LOADING", level=logging.DEBUG)
        with open(config_file_path) as f:
            config = json.load(f)
        log("Config loaded", tag="LOADED", level=logging.DEBUG)
        return config

    @classmethod
    def load_csv_files_pathes(cls, config_file_path: str = DATASET_CONFIG_FILE) -> dict[str, str]:
        """
        Returns {purpose: absolute_path_to_csv}.
        """
        config = cls.load_config(config_file_path)
        out = {}

        for section in config.values():
            folder = section["folder"]
            log(f"Scanning folder for CSV: {folder}", tag="SCAN", level=logging.DEBUG)
            for entity in os.listdir(folder):
                if entity.endswith(".csv"):
                    purpose = Path(entity).stem
                    path = str(Path(folder) / entity)
                    out[purpose] = path
                    log(f"Found CSV: {purpose} -> {path}", tag="FOUND", level=logging.DEBUG)

        log(f"Files: {list(out.keys())}", tag="FOUND", level=logging.DEBUG)
        return out

    @classmethod
    def anime_paths(cls, config_file_path: str = DATASET_CONFIG_FILE) -> dict[str, str]:
        needed = {'anime', 'animelist', 'rating_complete', 'watching_status', 'anime_with_synopsis'}
        paths = cls.load_csv_files_pathes(config_file_path)
        anime_paths = {k: v for k, v in paths.items() if k in needed}
        log(f"Anime files: {list(anime_paths.keys())}", tag="FOUND", level=logging.DEBUG)
        return anime_paths


# ============================================================
#                     Pandas Tools
# ============================================================

class PandasTools:
    """
    High-level pandas CSV loader.
    """

    TOO_BIG_FOR_PANDAS = {"animelist", "rating_complete"}

    @classmethod
    def load(cls, purpose: str, *, config_file_path: str = DATASET_CONFIG_FILE) -> pd.DataFrame | None:
        paths = CommonTools.load_csv_files_pathes(config_file_path)

        if purpose in cls.TOO_BIG_FOR_PANDAS:
            log(f"{purpose} is marked as too large for pandas — skipped.", tag="SKIPPED", level=logging.WARNING)
            return None

        file_path = paths[purpose]
        return cls.load_path(file_path)

    @staticmethod
    def load_path(file_path: str) -> pd.DataFrame:
        log(f"Loading with pandas: {Path(file_path).stem}", tag="LOADING", level=logging.DEBUG)
        df = pd.read_csv(file_path)
        log(f"PANDAS loaded {Path(file_path).stem}: shape={df.shape}", tag="LOADED", level=logging.DEBUG)
        return df
    
    @classmethod
    def load_anime_data(cls, *, config_file_path: str = DATASET_CONFIG_FILE) -> dict[str, pd.DataFrame]:
        paths = CommonTools.anime_paths(config_file_path)
        out = {}
        log(f"Pandas loading anime tables", tag="LOADING", level=logging.INFO)

        for purpose, path in paths.items():
            if purpose in cls.TOO_BIG_FOR_PANDAS:
                log(f"{purpose} is marked as too large for pandas — skipped.", tag="SKIPPED", level=logging.WARNING)
                continue

            out[purpose] = cls.load_path(path)

        log(f"Pandas loaded {len(out)} tables", tag="LOADED", level=logging.INFO)
        return out

# ============================================================
#                     DuckDB Tools
# ============================================================

class DuckDBTools:
    """
    Creates DuckDB views for CSV files. 
    Useful for heavy tables.
    """

    @classmethod
    def load(cls, purpose: str, *, config_file_path: str = DATASET_CONFIG_FILE) -> str:
        paths = CommonTools.load_csv_files_pathes(config_file_path)
        file_path = paths[purpose]
        return cls.load_path(file_path)

    @staticmethod
    def load_path(file_path: str) -> str:
        view_name = Path(file_path).stem
        log(f"Creating DuckDB view: {view_name}", tag="LOADING", level=logging.DEBUG)

        duckdb.query(f"""
            CREATE OR REPLACE VIEW {view_name} AS 
            SELECT * FROM read_csv_auto('{file_path}')
        """)

        log(f"DUCKDB view created: {view_name}", tag="LOADED", level=logging.DEBUG)
        return view_name

    @classmethod
    def load_anime_data(cls, *, config_file_path: str = DATASET_CONFIG_FILE) -> dict[str, str]:
        paths = CommonTools.anime_paths(config_file_path)
        out = {}

        log("Loading ALL anime tables (duckdb)...", tag="LOADING", level=logging.INFO)

        for purpose, file_path in paths.items():
            out[purpose] = cls.load_path(file_path)

        log(f"DuckDB created {len(out)} views", tag="LOADED", level=logging.INFO)
        return out

    @classmethod
    def cleanup(cls) -> None:
        logger.info("Cleaning up DuckDB views...")
        views = duckdb.query("SHOW VIEWS").fetchall()

        if not views:
            log("No DuckDB views to drop.", tag="DUCKDB CLEANUP", level=logging.DEBUG)
            return

        for (view_name,) in views:
            cls.drop_view(view_name)

        log("DuckDB cleanup completed.", tag="DUCKDB CLEANUP", level=logging.INFO)

    @staticmethod
    def drop_view(view_name: str) -> None:
        duckdb.query(f"DROP VIEW IF EXISTS {view_name}")
        log(f"Dropped DuckDB view: {view_name}", tag="DUCKDB CLEANUP", level=logging.DEBUG)
