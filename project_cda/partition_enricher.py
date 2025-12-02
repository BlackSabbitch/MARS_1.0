import pandas as pd
from collections import Counter

import pandas as pd
import ast
from collections import Counter
import numpy as np

class PartitionEnricher:
    def __init__(self, anime_metadata_path: str = "data/datasets/azathoth42/anime_sterilized.csv"):
        # 1. Загружаем
        self.meta_df = pd.read_csv(anime_metadata_path)
        
        # 2. Парсим жанры (строка "{'Action', 'Comedy'}" -> set({'Action', 'Comedy'}))
        if 'genres' in self.meta_df.columns:
            self.meta_df['genres'] = self.meta_df['genres'].apply(self._parse_set_string)
            
        # 3. Чистим Source
        if 'source' in self.meta_df.columns:
            self.meta_df['source'] = self.meta_df['source'].fillna("Unknown")

    def _parse_set_string(self, val):
        """
        Превращает строку "{'A', 'B'}" в питоновский set.
        """
        if pd.isna(val) or val == "":
            return set()
        
        try:
            # ast.literal_eval безопасно вычисляет структуру данных из строки
            parsed = ast.literal_eval(val)
            if isinstance(parsed, set):
                return parsed
            # Иногда бывает, что там список или кортеж, приведем к сету
            return set(parsed)
        except (ValueError, SyntaxError):
            return set()

    def enrich_partition(self, partition_csv_path: str) -> pd.DataFrame:
        """Склеивает результаты кластеризации с метаданными."""
        df_part = pd.read_csv(partition_csv_path)
        
        enriched_df = df_part.merge(
            self.meta_df, 
            on='anime_id', 
            how='left'
        )
        
        # Заглушка для тайтлов, которых нет в базе
        if 'title' in enriched_df.columns:
            enriched_df['title'] = enriched_df['title'].fillna("Unknown ID " + enriched_df['anime_id'].astype(str))
            
        return enriched_df

    def get_cluster_summaries(self, enriched_df: pd.DataFrame):
        """Генерирует описания для тултипов."""
        summaries = {}
        # Группируем, игнорируя NaN в cluster_id (если вдруг есть)
        groups = enriched_df.dropna(subset=['cluster_id']).groupby(['year', 'cluster_id'])
        
        for (year, c_id), group in groups:
            size = len(group)
            
            # 1. Топ-5 Аниме (по Popularity / Members / Score)
            sort_col = 'anime_id' # Fallback
            for col in ['members', 'scored_by', 'favorites', 'score']: # Ищем лучшую метрику популярности
                if col in group.columns:
                    sort_col = col
                    break
            
            # Берем топ-5
            top_anime_series = group.sort_values(sort_col, ascending=False).head(5)['title']
            top_anime = [str(t) for t in top_anime_series.tolist()]

            # 2. Main Feature (Жанры или Source)
            main_feature_str = "N/A"
            
            # Логика Жанров (теперь это точно сеты)
            if 'genres' in group.columns:
                all_tags = []
                # group['genres'] теперь содержит реальные сеты
                for val in group['genres']:
                    if isinstance(val, set) and val:
                        all_tags.extend(list(val))
                
                if all_tags:
                    # Самый частый тег
                    top_tag, _ = Counter(all_tags).most_common(1)[0]
                    
                    # Считаем покрытие (Coverage)
                    # Проверяем: входит ли тег в сет жанров конкретного аниме
                    count_with_tag = group['genres'].apply(
                        lambda x: top_tag in x if isinstance(x, set) else False
                    ).sum()
                    
                    perc = int((count_with_tag / size) * 100)
                    main_feature_str = f"Genre: {top_tag} ({perc}%)"
            
            # Логика Source (если жанры не сработали или хочется добавить)
            # Если Source важнее, можно поменять местами if
            if main_feature_str == "N/A" and 'source' in group.columns:
                sources = group['source'].replace('Unknown', np.nan).dropna()
                if not sources.empty:
                    top_source = sources.mode()[0]
                    count = sources.value_counts().iloc[0]
                    perc = int((count / size) * 100)
                    main_feature_str = f"Source: {top_source} ({perc}%)"

            # 3. Score
            avg_score = group['score'].mean() if 'score' in group.columns else 0
            
            # HTML
            desc = (
                f"<b>Cluster {int(c_id)} ({year})</b><br>"
                f"Size: {size}<br>"
                f"Avg Score: {avg_score:.2f}<br>"
                f"Main: {main_feature_str}<br>"
                f"<hr>"
                f"<b>Top Anime:</b><br>• " + "<br>• ".join(top_anime)
            )
            summaries[(year, int(c_id))] = desc
            
        return summaries


"""
# 1. Загружаем и чистим мету ОДИН РАЗ
enricher = PartitionEnricher("data/datasets/azathoth42/anime_sterilized.csv")

# 2. Создаем словарь для ClusterEvaluation
# Нам нужен формат: {anime_id: {'genres': {'Action', ...}, 'source': 'Original'}}
anime_info_dict = {}

# Превращаем DataFrame в словарь. Это очень быстро.
# to_dict('records') вернет список строк, нам нужен доступ по ID
temp_records = enricher.meta_df[['anime_id', 'genres', 'source']].to_dict('records')

for row in temp_records:
    anime_info_dict[row['anime_id']] = {
        'genres': row['genres'], # Тут уже set, спасибо Enricher'у
        'source': row['source']
    }

# ... Дальше запускаешь цикл перебора методов ...
# И передаешь этот anime_info_dict в ClusterEvaluation
"""
