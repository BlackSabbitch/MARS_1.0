import pandas as pd
from typing import Any, Dict, List, Optional
from pymongo.collection import Collection
from pymongo.errors import BulkWriteError
from database_connection_manager import DatabaseConnectionManager

class MongoTools:
    """
    Generic MongoDB tools:
    - Bulk write helper
    - Data enrichment (Federation logic)
    """
    
    def __init__(self, db_manager: DatabaseConnectionManager):
        self.db_manager = db_manager

    # DB access
    def get_mongo_collection(self, collection_name: str = None) -> Collection:
        """
        Return a MongoDB collection. 
        Delegates the request to the DatabaseConnectionManager.
        """
        return self.db_manager.get_mongo_collection(collection_name)
    
    # Federation / batch reading / enrichment
    def enrich_with_synopsis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Application-Level Join (Federation):
        Takes a Pandas DataFrame containing a 'MAL_ID' column.
        Fetches 'synopsis' from MongoDB for these IDs and merges it.
        """
        if df is None or df.empty:
            return df

        if "MAL_ID" not in df.columns:
            print("Warning: DataFrame missing 'MAL_ID'. Cannot enrich from Mongo.")
            return df

        try:
            # Optimization: batched read
            # Instead of iterating through the DataFrame and querying MongoDB 
            # for each ID, we collect all IDs and execute a single query using the '$in' operator.
            
            # 1. Collect IDs for batching
            mal_ids = df["MAL_ID"].astype(int).tolist()
            
            collection = self.get_mongo_collection("animes")
            
            # 2. Execute Batch Read
            # Minimize data transfer by projecting only required fields.
            docs = collection.find(
                {"_id": {"$in": mal_ids}}, 
                {"_id": 1, "synopsis": 1}
            )
            
            # Optimization: in-memory mapping
            # Converting list (has access time O(N)) to dict provides O(1) access time during the merge.
            synopsis_map = {doc["_id"]: doc.get("synopsis", "") for doc in docs}
            
            df["synopsis"] = df["MAL_ID"].map(synopsis_map)
            df["synopsis"] = df["synopsis"].fillna("Synopsis not available")
            return df

        except Exception as e:
            print(f"MongoDB Federation failed ({e}). Returning SQL data only.")
            df["synopsis"] = "Content unavailable (Mongo Error)"
            return df

    # Bulk write
    def safe_bulk_write(self, collection: Collection, ops: list) -> dict:
        """Execute bulk_write with ordered=False."""
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

    # Search anime IDs by text in 'synopsis'
    def search_anime_ids_by_text(self, keywords: str, limit: int = 100) -> List[int]:
        """
        Executes a Full-Text Search in MongoDB using the text index.
        Returns a list of MAL_IDs relevant to the provided keywords.
        """
        try:
            collection = self.get_mongo_collection("animes")
            
            # $text search requires a text index on 'synopsis' field
            cursor = collection.find(
                {"$text": {"$search": keywords}},
                {"score": {"$meta": "textScore"}, "_id": 1}
            ).sort([("score", {"$meta": "textScore"})]).limit(limit)

            return [doc["_id"] for doc in cursor]
            
        except Exception as e:
            print(f"Mongo Text Search failed: {e}")
            return []

    # Utils
    @staticmethod
    def lower_keys(d: Dict[str, Any]) -> Dict[str, Any]:
        """Converts dictionary keys to lowercase for standardized access."""
        return {k.lower(): v for k, v in d.items()}

    @staticmethod
    def split_list(s: Any) -> Optional[List[str]]:
        """Splits a semicolon/comma separated string into a cleaned list."""
        if s is None or str(s).strip() == "":
            return None
        s = str(s).replace(';', ',').strip()
        return [item.strip() for item in s.split(',') if item.strip()]

    @staticmethod
    def to_float2(s: Any) -> Optional[float]:
        """Converts a string to a float, rounding to 2 decimal places."""
        try:
            val = float(s)
            return round(val, 2)
        except: 
            return None

    @staticmethod
    def get_first_nonempty(d: Dict[str, Any], keys: List[str]) -> Optional[str]:
        """Returns the first non-None, non-empty string value for a list of keys."""
        for key in keys:
            val = d.get(key)
            if val is not None and str(val).strip() != "":
                return str(val).strip()
        return None

    @staticmethod
    def merge_synonyms(a: List[str], b: List[str]) -> List[str]:
        """Merges two lists and returns a de-duplicated, sorted list."""
        merged = set(a) | set(b)
        return sorted(list(merged))
    
    @staticmethod
    def clean_str(s: Any) -> Optional[str]:
        if s is None: return None
        s = str(s).strip()
        return s if s else None

    @staticmethod
    def to_int(s: Any) -> Optional[int]:
        try:
            return int(s) if s is not None and str(s).strip() != "" else None
        except: return None