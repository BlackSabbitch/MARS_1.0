import duckdb
import pandas as pd
from common_tools import CommonTools


class FFF:
    @classmethod
    def find_not_honest_users(cls):
        # Пути к файлам
        table_names_pathes = CommonTools.load_csv_files_pathes()
        animelist_path = table_names_pathes['animelist']
        anime_path = table_names_pathes['anime']
        rating_complete_path = table_names_pathes['rating_complete']

        # подключаемся к временной in-memory базе
        con = duckdb.connect(database=':memory:')

        # Загружаем только нужные поля (всё остальное не читаем)
        query_find_bad = f"""
        SELECT 
            l.user_id,
            l.anime_id,
            l.rating,
            l.watched_episodes,
            a.Episodes
        FROM read_csv_auto('{animelist_path}', 
                        columns={{'user_id':'BIGINT', 'anime_id':'BIGINT', 'rating':'INT', 'watching_status':'INT', 'watched_episodes':'INT'}})
                AS l
        JOIN read_csv_auto('{anime_path}', 
                        columns={{'MAL_id':'BIGINT', 'Episodes':'INT'}})
                AS a
        ON l.anime_id = a.MAL_id
        WHERE 
            l.watching_status = 2
            AND l.rating != 0
            AND l.watched_episodes < a.Episodes
        """

        # Загружаем только нужные строки
        df_incomplete_votes = con.execute(query_find_bad).fetchdf()
        return df_incomplete_votes
    
    @classmethod
    def clean_for_not_honest_users(cls):
        df_incomplete_votes = cls.find_not_honest_users()
        # Загружаем rating_complete (тут уже можно pandas, он небольшой)
        table_names_pathes = CommonTools.load_csv_files_pathes()
        rating_complete_path = table_names_pathes['rating_complete']
        df_rating_complete = pd.read_csv(rating_complete_path)

        # Формируем множество (user_id, anime_id) для удаления
        pairs_to_remove = set(zip(df_incomplete_votes.user_id, df_incomplete_votes.anime_id))

        # Фильтруем
        mask = ~df_rating_complete.apply(lambda x: (x.user_id, x.anime_id) in pairs_to_remove, axis=1)
        df_clean = df_rating_complete[mask].reset_index(drop=True)

        # Сохраняем
        df_clean.to_csv('rating_complete_cleaned.csv', index=False)
        print(f"Удалено строк: {len(df_rating_complete) - len(df_clean)}")

    @classmethod
    def restore_score(cls, df: pd.DataFrame):
        def restore_score_for_row(row):
            votes = []
            for i in range(1, 11):
                count = row[f"Score-{i}"]
                if pd.isna(count):
                    return None
                votes.append((i, count))
            total = sum(c for _, c in votes)
            if total == 0:
                return None
            return sum(i * c for i, c in votes) / total

        df['Score_restored'] = df.apply(restore_score_for_row, axis=1)