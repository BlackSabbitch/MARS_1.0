import os
import re
import math
from typing import Any, Dict, List, Optional

from pymongo import MongoClient, UpdateOne
from pymongo.collection import Collection
from pymongo.errors import BulkWriteError


class MongoTools:
    """
    Generic MongoDB tools:
    - Connection and collection access
    - Bulk write helper
    - Index creation
    - All shared helper functions used by data loaders
    """
    uri = "mongodb+srv://msamosudova:Duckling@mars-cluster.8ruotdw.mongodb.net/?appName=MARS-Cluster"
    db_name = "anime_db"
    collection = "animes"

    def __init__(self, uri: str, db_name: str, collection: str):
        self.uri = uri
        self.db_name = db_name
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collection = collection

    # ---------------------------------------------------------------------
    # DB ACCESS
    # ---------------------------------------------------------------------
    def get_collection(self, collection_name: str) -> Collection:
        """Return a MongoDB collection."""
        return self.db[collection_name]

    # ---------------------------------------------------------------------
    # BULK WRITE HELPER
    # ---------------------------------------------------------------------
    def safe_bulk_write(self, collection: Collection, ops: list) -> dict:
        """
        Execute bulk_write safely with ordered=False.
        Returns result summary.
        """
        if not ops:
            return {"modified": 0, "upserts": 0}

        try:
            result = collection.bulk_write(ops, ordered=False)
            return {
                "modified": result.modified_count or 0,
                "upserts": result.upserted_count or 0,
            }
        except BulkWriteError as bwe:
            print("BulkWriteError sample:", bwe.details.get("writeErrors", [])[:3])
            raise

    # ---------------------------------------------------------------------
    # INDEX CREATION (separate method, as requested)
    # ---------------------------------------------------------------------
    def create_indexes(self, collection: Collection, index_specs: list):
        """
        Create multiple indexes:
        index_specs = [
            [("name", "text"), ("synopsis", "text")],
            [("genres", 1)],
            [("stats.score", -1)]
        ]
        """
        for spec in index_specs:
            try:
                collection.create_index(spec)
            except Exception as e:
                print(f"Index creation failed for {spec}: {e}")

    # ---------------------------------------------------------------------
    # HELPER FUNCTIONS (moved from your original scripts)
    # ---------------------------------------------------------------------
    @staticmethod
    def clean_str(s: Any) -> Optional[str]:
        if s is None:
            return None
        s = str(s).strip()
        return s if s else None

    @staticmethod
    def to_int(s: Any) -> Optional[int]:
        try:
            if s is None or str(s).strip() == "":
                return None
            return int(s)
        except:
            return None

    @staticmethod
    def to_float2(s: Any) -> Optional[float]:
        try:
            if s is None or str(s).strip() == "":
                return None
            return round(float(s), 2)
        except:
            return None

    @staticmethod
    def lower_keys(d: Dict[Any, Any]) -> Dict[Any, Any]:
        """Return dict with lowercase string keys."""
        return {
            (k.lower() if isinstance(k, str) else k): v
            for k, v in d.items()
        }

    @staticmethod
    def split_list(cell: Any) -> List[str]:
        """
        Split comma/pipe/semicolon separated values into normalized string list.
        Removes duplicates, trims whitespace.
        """
        if cell is None:
            return []

        s = str(cell).strip()
        if not s or s.upper() in {"UNKNOWN", "NONE", "NULL", "N/A"}:
            return []

        parts = re.split(r"[|;,]", s)
        seen, out = set(), []
        for p in parts:
            v = p.strip()
            if not v:
                continue
            if v not in seen:
                seen.add(v)
                out.append(v)
        return out

    @staticmethod
    def merge_synonyms(list1: List[str], list2: List[str]) -> List[str]:
        """Merge synonym lists, removing duplicates."""
        seen, out = set(), []
        for x in (list1 or []) + (list2 or []):
            if not x:
                continue
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    # Additional tool if needed
    @staticmethod
    def get_first_nonempty(d: dict, candidate_keys: list) -> Optional[str]:
        """Return first nonempty string among provided keys."""
        for key in candidate_keys:
            v = d.get(key.lower())
            if v is None:
                continue
            s = str(v).strip()
            if s and s.upper() not in {"NULL", "NONE", "N/A", "UNKNOWN"}:
                return s
        return None