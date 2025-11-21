class IndexManager:
    """Creates and checks MySQL indexes."""

    def __init__(self, db):
        self.db = db

    def create_index(self, table, index_name, columns_csv):
        conn = self.db.get_mysql_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT 1
            FROM information_schema.statistics
            WHERE table_schema = DATABASE()
              AND table_name = %s
              AND index_name = %s
            LIMIT 1
        """, (table, index_name))

        if cursor.fetchone() is None:
            cursor.execute(
                f"ALTER TABLE {table} ADD INDEX {index_name} ({columns_csv})"
            )
            conn.commit()
            print(f"Created index {index_name} on {table}({columns_csv})")
        else:
            print(f"Index {index_name} already exists on {table}")

        cursor.close()

    def ensure_all_indexes(self):
        """Create all required project indexes."""
        print("\n=== Ensuring required MySQL indexes ===")
        self.create_index("anime", "idx_anime_name", "name")

        self.create_index("anime_genre", "idx_ag_by_genre", "genreID, MAL_ID")
        self.create_index("anime_genre", "idx_ag_by_anime", "MAL_ID, genreID")

        self.create_index("anime_user_rating", "idx_aur_user", "userID")
        self.create_index("anime_user_rating", "idx_aur_anime", "MAL_ID")

        self.create_index("users", "idx_users_country_age", "country, age")

        print("=== Indexes ensured ===\n")
