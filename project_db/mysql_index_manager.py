import time

from concurrency_rating_test import synchronized_print

class MySQLIndexManager:
    """
    Manages MySQL indexes.
    """

    def __init__(self, db):
        self.db = db
        
        self.index_definitions = {
            "idx_mal_rating": ("anime_user_rating", "MAL_ID, user_rating"),
            
            "idx_users_country_age": ("users", "country, age"),
            "idx_ag_by_genre": ("anime_genre", "genreID, MAL_ID"),
            "idx_score_members": ("anime_statistics", "score DESC, members DESC"),
            "idx_users_lat_lon": ("users", "latitude, longitude"),
            "idx_users_country_gender": ("users", "country, sex")
        }

    def _execute_drop(self, table, index_name, cur):
        """Internal function to safely drop an index, handling errors and FK checks."""
        try:
            # Disable FK checks to allow dropping indexes used by constraints
            cur.execute("SET FOREIGN_KEY_CHECKS = 0")
            
            # Attempt to drop the index
            cur.execute(f"ALTER TABLE {table} DROP INDEX {index_name}")
            
            # Re-enable FK checks
            cur.execute("SET FOREIGN_KEY_CHECKS = 1")
            return True # Success
        except Exception as e:
            # Re-enable FK checks in case of failure
            try:
                cur.execute("SET FOREIGN_KEY_CHECKS = 1")
            except: 
                pass
                
            # Check for errors indicating an essential (PK/FK) index
            error_str = str(e).lower()
            if "cannot drop index" in error_str or "foreign key constraint" in error_str or "primary" in error_str:
                return False # Skipped (Essential index)
            elif "check that column/key exists" in str(e):
                return True # Index didn't exist, success
            else:
                # Log other critical errors
                print(f"Error dropping index {index_name} on {table}: {e}")
                return False

    def create_specific_index(self, index_name):
        """Creates a specific index."""
        if index_name not in self.index_definitions:
            print(f"Error: Unknown index definition '{index_name}'")
            return

        table, columns = self.index_definitions[index_name]
        print(f"MySQL creating index {index_name} on {table}...")
        
        conn = self.db.new_mysql_connection()
        cur = conn.cursor()
        try:
            # Check existence
            cur.execute("""
                SELECT COUNT(*) FROM information_schema.statistics 
                WHERE table_schema = DATABASE() AND index_name = %s
            """, (index_name,))
            if cur.fetchone()[0] > 0:
                print(f"Index {index_name} already exists.")
                return

            start = time.time()
            cur.execute(f"ALTER TABLE {table} ADD INDEX {index_name} ({columns})")
            conn.commit()
            print(f"Done in {time.time() - start:.2f}s")
            
        except Exception as e:
             print(f"Failed to create {index_name}: {e}")
        finally:
            cur.close()
            conn.close()

    def drop_specific_index(self, index_name):
        """Drops a specific index safely by disabling FK checks."""
        if index_name not in self.index_definitions:
            return

        table, _ = self.index_definitions[index_name]
        print(f"MySQL dropping index {index_name} on {table}...")

        conn = self.db.new_mysql_connection()
        cur = conn.cursor()
        try:
            cur.execute("SET FOREIGN_KEY_CHECKS = 0")
            cur.execute(f"ALTER TABLE {table} DROP INDEX {index_name}")
            conn.commit()
            print("Dropped.")
        except Exception as e:
            if "check that column/key exists" in str(e):
                print("Index did not exist, skipping.")
            else:
                print(f"Error dropping index: {e}")
        finally:
            try:
                cur.execute("SET FOREIGN_KEY_CHECKS = 1")
            except: 
                pass
            cur.close()
            conn.close()

    def drop_all_user_defined_indexes(self):
        """
        Drops all non-PRIMARY and non-FK supporting secondary indexes 
        defined either in this class or manually in the database.
        """
        conn = self.db.new_mysql_connection()
        cur = conn.cursor()
        db_name = self.db.mysql_config['database']
        
        # 1. Get ALL secondary indexes in the database (excluding PRIMARY)
        cur.execute(f"""
            SELECT TABLE_NAME, INDEX_NAME 
            FROM information_schema.statistics 
            WHERE TABLE_SCHEMA = %s AND INDEX_NAME != 'PRIMARY'
        """, (db_name,))
        
        all_secondary_indexes = cur.fetchall()
        cur.close()

        print("\nStarting mass drop of secondary indexes...")
        
        dropped_count = 0
        skipped_count = 0
        
        # 2. Iterate and attempt to drop each one
        for table_name, index_name in all_secondary_indexes:
            if index_name.upper() == 'PRIMARY':
                continue
                
            temp_conn = self.db.new_mysql_connection()
            temp_cur = temp_conn.cursor()
            
            # The FK check is handled inside _execute_drop
            if self._execute_drop(table_name, index_name, temp_cur):
                dropped_count += 1
                synchronized_print(f" Dropped: {table_name}.{index_name}")
            else:
                skipped_count += 1
                synchronized_print(f"Kept: {table_name}.{index_name} (FK related)")
                
            temp_conn.commit()
            temp_cur.close()
            temp_conn.close()

        print(f"\nDropped: {dropped_count}, Skipped (FK/PK related): {skipped_count}.")

    def ensure_all_indexes(self):
        for name in self.index_definitions:
            self.create_specific_index(name)