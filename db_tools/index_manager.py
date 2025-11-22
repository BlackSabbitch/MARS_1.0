class IndexManager:
    """
    Manages all MySQL indexes for the project:
    • create
    • check
    • drop
    • recreate
    • list

    Clean, modular, and extensible.
    """

    def __init__(self, db):
        self.db = db

        # Central index registry
        # table → list of (index_name, columns)
        self.indexes = {
            "anime": [
                ("idx_anime_name", "name"),
            ],

            "anime_genre": [
                ("idx_ag_by_genre", "genreID, MAL_ID"),
                ("idx_ag_by_anime", "MAL_ID, genreID"),
            ],

            "anime_user_rating": [
                ("idx_aur_user", "userID"),
                ("idx_aur_anime", "MAL_ID"),
            ],

            "users": [
                ("idx_users_country_age", "country, age"),
                ("idx_users_sex", "sex"),               # NEW INDEX
            ],
        }

    # ---------------------------------------------------------
    # Core helpers
    # ---------------------------------------------------------
    def _connect(self):
        """Returns a fresh MySQL connection."""
        return self.db.new_mysql_connection()

    # ---------------------------------------------------------
    # Check if index exists
    # ---------------------------------------------------------
    def index_exists(self, table, index_name):
        conn = self._connect()
        cur = conn.cursor()

        cur.execute("""
            SELECT 1
            FROM information_schema.statistics
            WHERE table_schema = DATABASE()
              AND table_name = %s
              AND index_name = %s
            LIMIT 1
        """, (table, index_name))

        exists = cur.fetchone() is not None
        cur.close()
        conn.close()
        return exists

    # ---------------------------------------------------------
    # Create index
    # ---------------------------------------------------------
    def create_index(self, table, index_name, columns):
        if self.index_exists(table, index_name):
            print(f"✓ Index {index_name} already exists on {table}")
            return

        conn = self._connect()
        cur = conn.cursor()

        cur.execute(
            f"ALTER TABLE `{table}` ADD INDEX `{index_name}` ({columns})"
        )
        conn.commit()

        cur.close()
        conn.close()

        print(f"Created index {index_name} on {table}({columns})")

    # ---------------------------------------------------------
    # Drop a single index
    # ---------------------------------------------------------
    def drop_index(self, table, index_name):
        if not self.index_exists(table, index_name):
            print(f"Index {index_name} does not exist on {table}")
            return

        conn = self._connect()
        cur = conn.cursor()

        cur.execute(f"ALTER TABLE `{table}` DROP INDEX `{index_name}`")
        conn.commit()

        cur.close()
        conn.close()

        print(f"Dropped index {index_name} from {table}")

    # ---------------------------------------------------------
    # Ensure ALL project indexes exist
    # ---------------------------------------------------------
    def ensure_all_indexes(self):
        print("\n=== Ensuring required MySQL indexes ===")
        for table, idx_list in self.indexes.items():
            for index_name, columns in idx_list:
                self.create_index(table, index_name, columns)
        print("=== Indexes ensured ===\n")

    # ---------------------------------------------------------
    # Drop all project indexes
    # ---------------------------------------------------------
    def drop_all_indexes(self):
        print("\n=== DROPPING ALL PROJECT INDEXES ===")
        for table, idx_list in self.indexes.items():
            for index_name, _ in idx_list:
                self.drop_index(table, index_name)
        print("=== All project indexes removed ===\n")

    # ---------------------------------------------------------
    # Recreate (drop + create)
    # ---------------------------------------------------------
    def recreate_all_indexes(self):
        print("\n=== RECREATING ALL PROJECT INDEXES ===")
        self.drop_all_indexes()
        self.ensure_all_indexes()
        print("=== All indexes recreated ===\n")

    # ---------------------------------------------------------
    # List all indexes in DB
    # ---------------------------------------------------------
    def list_indexes(self):
        conn = self._connect()
        cur = conn.cursor()

        cur.execute("""
            SELECT table_name, index_name, column_name, seq_in_index
            FROM information_schema.statistics
            WHERE table_schema = DATABASE()
            ORDER BY table_name, index_name, seq_in_index
        """)

        rows = cur.fetchall()
        cur.close()
        conn.close()

        print("\n=== CURRENT INDEXES IN DATABASE ===")
        for r in rows:
            table, idx, col, seq = r
            print(f"{table:20} | {idx:25} | col={col:20} | pos={seq}")
        print("=====================================\n")

        return rows
