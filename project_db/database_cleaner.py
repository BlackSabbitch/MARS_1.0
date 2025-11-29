class DatabaseCleaner:
    """Clears or resets MySQL tables."""

    def __init__(self, db):
        self.db = db

    def truncate(self, table):
        conn = self.db.get_mysql_connection()
        cursor = conn.cursor()
        cursor.execute(f"TRUNCATE TABLE {table}")
        conn.commit()
        cursor.close()
        print(f"Truncated {table}")

    def truncate_all(self):
        tables = [
            "anime_user_rating", "anime_genre", "anime_studio",
            "anime_producer", "anime_licensor", "users",
            "anime_statistics", "anime"
        ]

        for t in tables:
            self.truncate(t)

        print("All tables truncated.")

    def drop_all(self):
        conn = self.db.get_mysql_connection()
        cursor = conn.cursor()

        cursor.execute("SET FOREIGN_KEY_CHECKS=0")
        cursor.execute("SHOW TABLES")

        for (table,) in cursor.fetchall():
            cursor.execute(f"DROP TABLE IF EXISTS {table}")

        conn.commit()
        cursor.close()
        print("All tables dropped.")
