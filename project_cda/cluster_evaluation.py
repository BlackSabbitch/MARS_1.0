import numpy as np
import pandas as pd
import ast
from scipy.stats import entropy
from sklearn.metrics import adjusted_mutual_info_score
from collections import Counter, defaultdict
from project_cda.tag_formatter import log


class ClusterEvaluation:
    def __init__(self, method_name: str, aligned_history: dict, anime_info: dict = None, modularity_dict: dict = None):
        """
        aligned_history: {year: {anime_id: cluster_id}}
        """
        self.method_name = method_name
        self.history = aligned_history
        self.years = sorted(aligned_history.keys())
        self.anime_info = self._sanitize_anime_info(anime_info) if anime_info else None
        self.modularity_dict = modularity_dict if modularity_dict else {}

    def _sanitize_anime_info(self, raw_info: dict) -> dict:
        """Очистка метаданных (строки -> сеты)."""
        clean_info = {}
        for anime_id, attrs in raw_info.items():
            clean_attrs = attrs.copy()
            # Парсинг жанров
            if 'genres' in clean_attrs:
                val = clean_attrs['genres']
                if isinstance(val, str):
                    try:
                        parsed = ast.literal_eval(val)
                        clean_attrs['genres'] = set(parsed) if isinstance(parsed, (list, set, tuple)) else {str(parsed)}
                    except:
                        clean_attrs['genres'] = {val} if val else set()
                elif isinstance(val, (list, tuple)):
                    clean_attrs['genres'] = set(val)
                elif pd.isna(val):
                    clean_attrs['genres'] = set()
            
            # Очистка Source
            if 'studio' in clean_attrs:
                clean_attrs['studio'] = str(clean_attrs['studio']) if not pd.isna(clean_attrs['studio']) else "Unknown"
                     
            clean_info[anime_id] = clean_attrs
        return clean_info

    # --- Метрики мгновенного состояния (Snapshot Metrics) ---

    def _get_gini(self, year) -> float:
        """Считает Джини размеров кластеров для конкретного года."""
        counts = list(Counter(self.history[year].values()).values())
        if len(counts) < 2: return 0.0
        
        array = np.array(counts, dtype=np.float64)
        array = np.sort(array)
        n = array.shape[0]
        index = np.arange(1, n + 1)
        return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

    def _get_entropy(self, year) -> float:
        """Энтропия Шеннона распределения размеров (разнообразие)."""
        counts = np.array(list(Counter(self.history[year].values()).values()))
        if counts.sum() == 0: return 0.0
        probs = counts / counts.sum()
        return entropy(probs)

    def _get_purity(self, year, attribute='genres') -> float:
        """Чистота кластеров в конкретном году."""
        if not self.anime_info: return 0.0
        
        clusters = defaultdict(list)
        for node, c_id in self.history[year].items():
            clusters[c_id].append(node)
        
        total = 0
        correct = 0
        
        for nodes in clusters.values():
            if len(nodes) < 3: continue 
            
            all_values = []
            for n in nodes:
                val = self.anime_info.get(n, {}).get(attribute)
                if val:
                    if isinstance(val, set): all_values.extend(list(val))
                    else: all_values.append(val)
            
            if not all_values: continue
            
            # Доминирующий признак
            most_common, _ = Counter(all_values).most_common(1)[0]
            
            # Сколько узлов его имеют
            match_count = 0
            for n in nodes:
                val = self.anime_info.get(n, {}).get(attribute, set())
                if isinstance(val, set):
                    if most_common in val: match_count += 1
                else:
                    if most_common == val: match_count += 1
            
            correct += match_count
            total += len(nodes)
            
        return correct / total if total > 0 else 0.0

    # --- Метрики переходов (Transition Metrics) ---

    def _get_stability_ami(self, year) -> float:
        """
        AMI текущего года относительно ПРЕДЫДУЩЕГО.
        Для первого года возвращает NaN (или 1.0, по вкусу).
        """
        idx = self.years.index(year)
        if idx == 0: return np.nan # Нет прошлого
        
        prev_year = self.years[idx - 1]
        
        # Пересечение узлов (только выжившие)
        common = list(set(self.history[prev_year].keys()) & set(self.history[year].keys()))
        if not common: return 0.0
        
        labels_prev = [self.history[prev_year][n] for n in common]
        labels_curr = [self.history[year][n] for n in common]
        
        return adjusted_mutual_info_score(labels_prev, labels_curr)

    def get_trajectory_df(self) -> pd.DataFrame:
        """
        Собирает все метрики в единый DataFrame по годам.
        """
        log(f"Calculating trajectory for {self.method_name}...", tag="EVAL", level="INFO")
        rows = []
        for year in self.years:
            row = {
                "Year": year,
                "Method": self.method_name,
                "Gini_Spatial": self._get_gini(year),
                "Entropy_Info": self._get_entropy(year),
                "Stability_AMI": self._get_stability_ami(year),
                "Purity_Source": self._get_purity(year, 'source'),
                "Purity_Genre": self._get_purity(year, 'genres'),
                "Modularity": self.modularity_dict.get(year, np.nan),
                # Добавим количество кластеров просто для справки
                "N_Clusters": len(set(self.history[year].values()))
            }
            rows.append(row)
            
        return pd.DataFrame(rows).set_index("Year")
