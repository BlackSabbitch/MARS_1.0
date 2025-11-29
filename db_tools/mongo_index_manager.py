from pymongo import ASCENDING, DESCENDING

class MongoIndexManager:

    def __init__(self, db):
        self.db = db
        self.collection = self.db.get_mongo_collection("animes")

        self.indexes = {
            "idx_score": [("score", DESCENDING)],
            "idx_genres": [("genres", ASCENDING)],
            "idx_genre_score": [("genres", ASCENDING), ("score", DESCENDING)],
            "idx_type": [("typeID", ASCENDING)],
            "idx_source": [("sourceID", ASCENDING)],
            "idx_age_rating": [("age_ratingID", ASCENDING)],
        }

    # ------------------------------------
    # Create one index
    # ------------------------------------
    def create_index(self, name, fields):
        print(f"Creating Mongo index: {name} = {fields}")
        self.collection.create_index(fields, name=name)

    # ------------------------------------
    # Text index for synopsis
    # ------------------------------------
    def ensure_text_index(self):
        """Ensures a valid text index exists for synopsis."""
        print("Checking existing text indexes...")

        indexes = list(self.collection.list_indexes())
        existing_text_index = None

        for idx in indexes:
            if idx.get("weights"):  # text index always has weights
                existing_text_index = idx
                break

        # ------ CASE 1: Text index exists ------
        if existing_text_index:
            name = existing_text_index["name"]
            weights = existing_text_index["weights"]

            print(f"Found existing text index: {name}, weights={weights}")

            # Does it include synopsis?
            if "synopsis" in weights:
                print("Existing text index already includes 'synopsis'. Nothing to do.")
                return

            print("Existing text index does NOT include 'synopsis'.")
            print("MongoDB allows only ONE text index per collection.")

            print("You need to manually drop and recreate a combined text index.")
            print("Suggested definition:")
            print("""
                    db.animes.dropIndex("<EXISTING_INDEX_NAME>");
                    db.animes.createIndex(
                        { name: "text", synopsis: "text" },
                        { name: "name_synopsis_text" }
                    );
                            """)
            return

        # ------ CASE 2: No text index → create ------
        print("No text index found → creating index on synopsis...")
        self.collection.create_index(
            [("synopsis", "text")],
            name="idx_synopsis_text"
        )
        print("Created text index on synopsis.")


    # ------------------------------------
    # Create all indexes
    # ------------------------------------
    def ensure_all_indexes(self):
        print("\n=== Creating MongoDB Indexes ===")

        for name, fields in self.indexes.items():
            self.create_index(name, fields)

        self.ensure_text_index()
        print("=== DONE ===\n")

    # ------------------------------------
    # Drop all indexes
    # ------------------------------------
    def drop_all_indexes(self):
        print("\n=== DROPPING ALL MONGO INDEXES ===")
        self.collection.drop_indexes()
        print("=== DONE ===\n")

    # ------------------------------------
    # Recreate indexes
    # ------------------------------------
    def recreate_indexes(self):
        self.drop_all_indexes()
        self.ensure_all_indexes()
