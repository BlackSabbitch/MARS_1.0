import pandas as pd
from collections import Counter

import pandas as pd
import ast
from collections import Counter
import numpy as np

class PartitionEnricher:
    def __init__(self, metadata_path: str, key_col: str = 'anime_id', set_cols: list = None):
        """
            Универсальный обогатитель.
            
            Args:
                metadata_path: Путь к CSV с метаданными.
                key_col: Название колонки-ключа ('anime_id' или 'username').
                set_cols: Список колонок, которые нужно парсить как сеты (напр. ['genres']). 
                        Для юзеров передавать None или [].
        """
        self.key_col = key_col
        self.set_cols = set_cols if set_cols else []

        # 1. Загружаем
        self.meta_df = pd.read_csv(metadata_path)

        # 2. Обработка ключа (если это ID и он числовой)
        # Если ключ 'username', он останется строкой, это ок.
        if key_col in self.meta_df.columns and pd.api.types.is_numeric_dtype(self.meta_df[key_col]):
             self.meta_df[key_col] = self.meta_df[key_col].fillna(0).astype(int)

        # 3. Парсинг сетов (только для указанных колонок)
        for col in self.set_cols:
            if col in self.meta_df.columns:
                self.meta_df[col] = self.meta_df[col].apply(self._parse_set_string)

    def _parse_set_string(self, val):
        """
        Превращает строку "{'A', 'B'}" в питоновский set.
        """
        if pd.isna(val) or val == "":
            return set()
        
        try:
            # ast.literal_eval безопасно вычисляет структуру данных из строки
            parsed = ast.literal_eval(val)
            if isinstance(parsed, (set, list, tuple)):
                return set(parsed)
            return {str(parsed)}
        except (ValueError, SyntaxError):
            return {val}

    def enrich_partition(self, partition_csv_path: str) -> pd.DataFrame:
        """Склеивает результаты кластеризации с метаданными."""
        df_part = pd.read_csv(partition_csv_path)

        # Проверка, что ключ есть в обоих файлах
        if self.key_col not in df_part.columns:
            raise ValueError(f"Partition file is missing key column: {self.key_col}")
        
        df_part[self.key_col] = df_part[self.key_col].astype('Int64')
        self.meta_df[self.key_col] = self.meta_df[self.key_col].astype('Int64')

        return df_part.merge(self.meta_df, on=self.key_col, how='left')

    def get_metadata_dict(self) -> dict:
        """
        [НОВОЕ] Возвращает словарь {id: {attr: val}} для ClusterEvaluation.
        """
        # Превращаем DataFrame в словарь, ориентированный по индексу (key_col)
        # orient='index' сделает {id: {'genres': {...}, 'source': '...'}}
        if self.key_col not in self.meta_df.columns:
             return {}
        
        return self.meta_df.set_index(self.key_col).to_dict(orient='index')
