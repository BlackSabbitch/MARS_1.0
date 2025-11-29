class MySQLIndexManager:
    """Manages creation, deletion and recreation of MySQL indexes."""

    def __init__(self, db):
        self.db = db

        self.indexes = {
            "anime": [
                ("idx_anime_name", "name(100)"),
            ],

            "anime_genre": [
                ("idx_ag_by_genre", "genreID, MAL_ID"),
                ("idx_ag_by_anime", "MAL_ID, genreID"),
            ],

            "anime_studio": [
                ("idx_as_by_studio", "studioID, MAL_ID"),
                ("idx_as_by_anime", "MAL_ID, studioID"),
            ],

            "anime_producer": [
                ("idx_ap_by_producer", "producerID, MAL_ID"),
                ("idx_ap_by_anime", "MAL_ID, producerID"),
            ],

            "anime_licensor": [
                ("idx_al_by_licensor", "licensorID, MAL_ID"),
                ("idx_al_by_anime", "MAL_ID, licensorID"),
            ],

            "anime_user_rating": [
                ("idx_aur_user", "userID"),
                ("idx_aur_anime", "MAL_ID"),
                ("idx_aur_anime_rating", "MAL_ID, user_rating"),
                ("idx_aur_user_anime", "userID, MAL_ID, user_rating"),  # main index for optimization
            ],

            "users": [
                ("idx_users_country_age", "country, age"),
                ("idx_users_sex", "sex"),
                ("idx_users_country_sex", "country, sex"),
                ("idx_users_lat_lon", "latitude, longitude"),
            ],

            # --------------------
            # genre
            # --------------------
            "genre": [
                ("idx_genre_name", "genre"),  # optimization — speeds genre lookup
            ],

            # --------------------
            # anime_statistics
            # --------------------
            "anime_statistics": [
                ("idx_members", "members"),  # optimization — speeds WHERE members >= 500
                ("idx_score_members", "score, members"),  # optional
            ],
        }

    def kill_blocking_queries(self, table: str):
        conn = self.db.get_mysql_connection()
        cur = conn.cursor(dictionary=True)
        cur.execute(
            """
            SELECT ID
            FROM INFORMATION_SCHEMA.PROCESSLIST
            WHERE COMMAND = 'Query'
            AND TIME > 30
            AND INFO LIKE %s
            """,
            (f"%{table}%",)
        )
        rows = cur.fetchall()
        cur.close()

        if rows:
            cur2 = conn.cursor()
            for r in rows:
                try:
                    cur2.execute(f"KILL {r['ID']}")
                except:
                    pass
            cur2.close()

        conn.commit()
        conn.close()

    # ------------------------------------
    # Create one index
    # ------------------------------------
    def create_index(self, table: str, index_name: str, columns: str):
        attempts = 3

        for attempt in range(1, attempts + 1):
            conn = None
            cur = None

            try:
                conn = self.db.new_mysql_connection()
                cur = conn.cursor()

                # existence check
                cur.execute("""
                    SELECT COUNT(*)
                    FROM information_schema.statistics
                    WHERE table_schema = DATABASE()
                      AND table_name = %s
                      AND index_name = %s
                """, (table, index_name))

                if cur.fetchone()[0] > 0:
                    return True

                cur.execute(
                    f"ALTER TABLE {table} ADD INDEX {index_name} ({columns})"
                )

                conn.commit()
                return True

            except Exception as e:
                if "Lock wait timeout" in str(e) or "metadata lock" in str(e):
                    self.kill_blocking_queries(table)
                    if attempt == attempts:
                        raise
                else:
                    raise

            finally:
                if cur:
                    try: cur.close()
                    except: pass
                if conn:
                    try: conn.close()
                    except: pass

        return False


    # ------------------------------------
    # Create ALL indexes
    # ------------------------------------
    def ensure_all_indexes(self):
        print("\n=== Creating MySQL Indexes ===")

        for table, idx_list in self.indexes.items():
            for index_name, columns in idx_list:
                self.create_index(table, index_name, columns)

        print("=== DONE ===\n")

    # ------------------------------------
    # DROP all indexes (for benchmarking )
    # ------------------------------------
    def drop_all_indexes(self):
        print("\n=== FORCE DROPPING ALL MYSQL INDEXES ===")
        conn = self.db.get_mysql_connection()
        cur = conn.cursor(buffered=True)

        # Disable FKs for speed
        cur.execute("SET FOREIGN_KEY_CHECKS = 0")

        # --- 1. Kill LOCKING threads ---
        print("Killing metadata lock wait threads…")
        cur.execute("""
            SELECT ID
            FROM information_schema.PROCESSLIST
            WHERE COMMAND != 'Sleep'
            AND STATE LIKE '%metadata lock%'
        """)
        for (pid,) in cur.fetchall():
            print(f"Killing thread {pid}")
            try:
                cur.execute(f"KILL {pid}")
            except Exception as e:
                print("    (skip)", e)

        # --- 2. Drop indexes ---
        for table, idx_list in self.indexes.items():
            for index_name, _ in idx_list:
                print(f"Dropping {index_name} on {table}...")

                # check exists
                cur.execute("""
                    SELECT COUNT(*)
                    FROM information_schema.statistics
                    WHERE table_schema = DATABASE()
                    AND table_name = %s
                    AND index_name = %s
                """, (table, index_name))
                exists = cur.fetchone()[0]
                if exists == 0:
                    print("  -> does not exist, skipping")
                    continue

                try:
                    # --- 3. Force unlock metadata ---
                    cur.execute("SET SESSION lock_wait_timeout = 1")

                    cur.execute(f"ALTER TABLE {table} DROP INDEX {index_name}")
                    conn.commit()
                    print("dropped")

                except Exception as e:
                    print(f"  (retry DROP ONLINE) {e}")

                    # Retry: ONLINE drop
                    try:
                        cur.execute(
                            f"ALTER TABLE {table} DROP INDEX {index_name}, ALGORITHM=INPLACE, LOCK=NONE"
                        )
                        conn.commit()
                        print("dropped (INPLACE)")
                    except Exception as e2:
                        print(f"  (skip) {e2}")

        cur.execute("SET FOREIGN_KEY_CHECKS = 1")
        cur.close()
        conn.commit()
        conn.close()
        print("=== DONE (FORCE SAFE) ===\n")


    # ------------------------------------
    # Recreate (drop -> create)
    # ------------------------------------
    def recreate_indexes(self):
        self.drop_all_indexes()
        self.ensure_all_indexes()
