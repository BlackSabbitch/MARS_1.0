from pymongo import ASCENDING, DESCENDING, TEXT
from pymongo.errors import OperationFailure

class MongoIndexManager:
    """
    Manages MongoDB indexes.
    """

    def __init__(self, mongo_tools):
        self.mongo_tools = mongo_tools
        self.collection = self.mongo_tools.get_mongo_collection("animes")

        # Dictionary of index definitions
        self.index_definitions = {
            "idx_score": [("score", DESCENDING)],
            "idx_genres": [("genres", ASCENDING)],
            "idx_synopsis_text": [("synopsis", TEXT)] 
        }

    def create_specific_index(self, index_name):
        """Creates a specific index by name."""
        if index_name not in self.index_definitions:
            print(f"Error: Unknown index definition '{index_name}'")
            return

        fields = self.index_definitions[index_name]
        print(f"Mongo creating index {index_name}...")

        try:
            # Special handling for Text Index (MongoDB allows only one per collection)
            if self._is_text_index(fields):
                self._ensure_text_index_safe(index_name, fields)
            else:
                self.collection.create_index(fields, name=index_name)
                print(f"Mongo created {index_name}.")

        except Exception as e:
            print(f"Mongo failed to create {index_name}: {e}")

    def drop_specific_index(self, index_name):
        """Drops a specific index by name."""
        print(f"Mongo dropping index {index_name}...")
        try:
            self.collection.drop_index(index_name)
            print("Dropped.")
        except OperationFailure as e:
            if "index not found" in str(e):
                print("Index did not exist, skipping.")
            else:
                print(f"Error dropping index: {e}")
        except Exception as e:
            print(f"Error: {e}")

    def ensure_all_indexes(self):
        """Restores ALL defined indexes."""
        print("\nMongo restoring indexes")
        for name in self.index_definitions:
            self.create_specific_index(name)
        print("Done\n")

    # Helpers
    def _is_text_index(self, fields):
        """Checks if the index definition contains a TEXT field."""
        return any(ftype == TEXT for _, ftype in fields)

    def _ensure_text_index_safe(self, name, fields):
        """
        Creates a text index. 
        If a conflicting text index exists, it drops it first.
        """
        try:
            # Check existing indexes
            existing_indexes = self.collection.list_indexes()
            for idx in existing_indexes:
                if "weights" in idx: # This is a text index
                    existing_name = idx["name"]
                    
                    if existing_name == name:
                        print("Text index already exists.")
                        return
                    
                    print(f"Found conflicting text index '{existing_name}'. Dropping it...")
                    self.collection.drop_index(existing_name)
            
            # Create the new one
            self.collection.create_index(fields, name=name)
            print(f"Created text index '{name}'.")
        except Exception as e:
             print(f"Error in text index creation: {e}")