class SchemaManager:
    """Handles MySQL schema creation (tables, PK, FK, DDL)."""

    def __init__(self, db):
        self.db = db

    def run_sql_file(self, file_path):
        """Runs SQL commands from a .sql file (schema creation)."""
        conn = self.db.get_mysql_connection()
        cursor = conn.cursor()

        with open(file_path, "r", encoding="utf-8") as f:
            sql_commands = f.read()

        for stmt in sql_commands.split(";"):
            stmt = stmt.strip()
            if stmt:
                cursor.execute(stmt)

        conn.commit()
        cursor.close()
        print(f"Schema applied from {file_path}")

    def create_schema(self):
        """Runs default schema.sql in project folder."""
        self.run_sql_file("db_tools/schema.sql")
