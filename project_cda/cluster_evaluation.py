import numpy as np
import pandas as pd
import ast
from scipy.stats import entropy
from sklearn.metrics import adjusted_mutual_info_score
from collections import Counter

class ClusterEvaluation:
    def __init__(self, method_name: str, aligned_history: dict, anime_info: dict = None):
        """
        Args:
            method_name: Название метода.
            aligned_history: Словарь {year: {node_id: cluster_id}}.
            anime_info: Словарь метаданных. 
                        Класс сам попытается распарсить строки-сеты в полях 'genres'.
        """
        self.method_name = method_name
        self.history = aligned_history
        self.years = sorted(aligned_history.keys())
        
        # --- АВТОМАТИЧЕСКАЯ ОЧИСТКА INFO ---
        self.anime_info = self._sanitize_anime_info(anime_info) if anime_info else None

    def _sanitize_anime_info(self, raw_info: dict) -> dict:
        """
        Превращает строковые представления сетов "{'A', 'B'}" в реальные сеты.
        Делает глубокую копию, чтобы не менять оригинал снаружи.
        """
        clean_info = {}
        for anime_id, attrs in raw_info.items():
            clean_attrs = attrs.copy()
            
            # Проверяем поле genres
            if 'genres' in clean_attrs:
                val = clean_attrs['genres']
                if isinstance(val, str) and val.startswith("{") and val.endswith("}"):
                    try:
                        clean_attrs['genres'] = ast.literal_eval(val)
                    except (ValueError, SyntaxError):
                        clean_attrs['genres'] = set()
                elif isinstance(val, str):
                    # Если вдруг там не сет, а просто строка - оборачиваем в сет
                    clean_attrs['genres'] = {val}
            
            # Проверяем поле source
            if 'source' in clean_attrs:
                if pd.isna(clean_attrs['source']):
                     clean_attrs['source'] = "Unknown"
                     
            clean_info[anime_id] = clean_attrs
        return clean_info

    def _calculate_gini(self, array):
        """Расчет коэффициента Джини."""
        array = np.array(array, dtype=np.float64)
        if np.amin(array) < 0: array -= np.amin(array)
        array += 1e-7
        array = np.sort(array)
        index = np.arange(1, array.shape[0] + 1)
        n = array.shape[0]
        return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

    def metric_spatial_gini(self) -> float:
        """Неравенство размеров кластеров (0..1)."""
        gini_values = []
        for year in self.years:
            counts = list(Counter(self.history[year].values()).values())
            if len(counts) > 1:
                gini_values.append(self._calculate_gini(counts))
            else:
                gini_values.append(0.0)
        return float(np.mean(gini_values))

    def metric_temporal_stability_ami(self) -> float:
        """Стабильность во времени (AMI)."""
        ami_scores = []
        for i in range(len(self.years) - 1):
            y1, y2 = self.years[i], self.years[i+1]
            common_nodes = list(set(self.history[y1].keys()) & set(self.history[y2].keys()))
            if not common_nodes: continue
                
            labels1 = [self.history[y1][n] for n in common_nodes]
            labels2 = [self.history[y2][n] for n in common_nodes]
            ami_scores.append(adjusted_mutual_info_score(labels1, labels2))
        return float(np.mean(ami_scores)) if ami_scores else 0.0

    def metric_entropy_mdl_proxy(self) -> float:
        """Энтропия Шеннона (Информативность)."""
        entropy_values = []
        for year in self.years:
            counts = np.array(list(Counter(self.history[year].values()).values()))
            total = counts.sum()
            probs = counts / total
            entropy_values.append(entropy(probs))
        return float(np.mean(entropy_values))

    def metric_cluster_count_volatility(self) -> float:
        """Дисперсия количества кластеров."""
        counts = [len(set(self.history[y].values())) for y in self.years]
        return float(np.std(counts))

    def metric_semantic_purity(self, attribute='genres') -> float:
        """
        Универсальная Purity.
        Работает и с 'source' (строка), и с 'genres' (сет).
        """
        if not self.anime_info: return 0.0
            
        purity_per_year = []
        for year in self.years:
            partition = self.history[year]
            clusters = {}
            for node, c_id in partition.items():
                clusters.setdefault(c_id, []).append(node)
            
            total_nodes = 0
            correct_nodes = 0
            
            for nodes in clusters.values():
                if len(nodes) < 5: continue 
                
                # Собираем значения
                values_list = []
                for n in nodes:
                    val = self.anime_info.get(n, {}).get(attribute, None)
                    if val: values_list.append(val)
                
                if not values_list: continue

                # Ветка 1: Строки (Source)
                if isinstance(values_list[0], str):
                    dom_val, count = Counter(values_list).most_common(1)[0]
                    correct_nodes += count
                    
                # Ветка 2: Сеты/Списки (Genres) - проверяем вхождение
                elif isinstance(values_list[0], (set, list, tuple)):
                    all_tags = []
                    for v in values_list: all_tags.extend(list(v))
                    
                    if all_tags:
                        dom_tag, _ = Counter(all_tags).most_common(1)[0]
                        # Считаем покрытие
                        count = sum(1 for v in values_list if dom_tag in v)
                        correct_nodes += count
                
                total_nodes += len(nodes)
                
            if total_nodes > 0:
                purity_per_year.append(correct_nodes / total_nodes)
                
        return float(np.mean(purity_per_year)) if purity_per_year else 0.0

    def evaluate(self) -> dict:
        """Сводный отчет."""
        print(f"Evaluating method: {self.method_name}...")
        return {
            "Method": self.method_name,
            "Avg_Gini": round(self.metric_spatial_gini(), 4),
            "Avg_Entropy": round(self.metric_entropy_mdl_proxy(), 4),
            "Stability_AMI": round(self.metric_temporal_stability_ami(), 4),
            "Count_Volatility": round(self.metric_cluster_count_volatility(), 2),
            # Считаем оба вида чистоты!
            "Purity_Source": round(self.metric_semantic_purity(attribute='source'), 4),
            "Purity_Genre": round(self.metric_semantic_purity(attribute='genres'), 4),
        }

"""
# 1. Загружаем CSV пандой
df_meta = pd.read_csv("data/datasets/azathoth42/anime_sterilized.csv")

# 2. Делаем простой словарь (без ручного парсинга ast!)
# Просто тупо перегоняем в dict
anime_info_raw = df_meta.set_index('anime_id')[['genres', 'source']].to_dict('index')

# 3. Создаем оценщик
# Он сам увидит, что genres это "{'Comedy'}" и превратит в set({'Comedy'})
evaluator = ClusterEvaluation("Method_Name", aligned_history, anime_info_raw)

print(evaluator.evaluate())
"""