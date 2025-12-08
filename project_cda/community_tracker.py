from __future__ import annotations
import os
import pandas as pd
from collections import defaultdict, Counter
from project_cda.tag_formatter import log

class CommunityTracker:
    def __init__(self, threshold: float = 0.1):
        """
        Args:
            threshold: Минимальный Jaccard index (0.1 = 10%), 
                       чтобы считать новый кластер наследником старого.
        """
        self.threshold = threshold
        self.next_unique_id = 0
        # Память: {Global_Cluster_ID: set(anime_ids)}
        self.prev_clusters_sets = {}

    def track_communities(self, partition_by_year: dict) -> dict:
        """
        Основной метод запуска.
        Args:
            partition_by_year: { 2006: {anime_id: raw_cluster_id}, 2007: ... }
        Returns:
            aligned_history: { 2006: {anime_id: GLOBAL_cluster_id}, ... }
        """
        aligned_history = {}
        sorted_years = sorted(partition_by_year.keys())
        
        # Сброс состояния перед запуском
        self.next_unique_id = 0
        self.prev_clusters_sets = {}

        for year in sorted_years:
            raw_partition = partition_by_year[year]
            
            # 1. Инвертируем: {cluster_id: set(nodes)}
            curr_clusters_sets = defaultdict(set)
            for node, c_id in raw_partition.items():
                curr_clusters_sets[c_id].add(node)

            # 2. Логика первого года
            if not self.prev_clusters_sets:
                mapping = {}
                for raw_id in curr_clusters_sets:
                    mapping[raw_id] = self.next_unique_id
                    self.next_unique_id += 1
                
                # Сохраняем состояние
                self._update_state(curr_clusters_sets, mapping)
                
                # Применяем маппинг
                aligned_history[year] = {n: mapping[pid] for n, pid in raw_partition.items()}
            
            # 3. Логика последующих лет (Matching)
            else:
                mapping = self._match_clusters(curr_clusters_sets)
                
                # Сохраняем состояние
                self._update_state(curr_clusters_sets, mapping)
                
                # Применяем маппинг
                aligned_history[year] = {n: mapping[pid] for n, pid in raw_partition.items()}

            log(f"Year {year}: done. Unique clusters: {len(curr_clusters_sets)}", tag="TRACK", level="INFO")

        return aligned_history

    def _match_clusters(self, curr_clusters_sets):
        """Жадное сопоставление (Greedy Matching)"""
        matches = []
        
        # Считаем скоры для всех пар
        for curr_raw_id, curr_nodes in curr_clusters_sets.items():
            for prev_global_id, prev_nodes in self.prev_clusters_sets.items():
                
                intersection = len(curr_nodes & prev_nodes)
                union = len(curr_nodes | prev_nodes)
                score = intersection / union if union > 0 else 0
                
                if score > self.threshold:
                    matches.append((score, prev_global_id, curr_raw_id))

        # Сортируем: самые похожие забирают имена первыми
        matches.sort(key=lambda x: x[0], reverse=True)
        
        mapping = {}
        used_prev = set()
        used_curr = set()
        
        # Раздаем имена отцов
        for score, prev_id, curr_id in matches:
            if prev_id not in used_prev and curr_id not in used_curr:
                mapping[curr_id] = prev_id
                used_prev.add(prev_id)
                used_curr.add(curr_id)
        
        # Остальным — новые паспорта
        for curr_id in curr_clusters_sets:
            if curr_id not in mapping:
                mapping[curr_id] = self.next_unique_id
                self.next_unique_id += 1
                
        return mapping

    def _update_state(self, curr_clusters_sets, mapping):
        """Обновляет self.prev_clusters_sets для следующей итерации"""
        self.prev_clusters_sets = {}
        for raw_id, nodes in curr_clusters_sets.items():
            global_id = mapping[raw_id]
            self.prev_clusters_sets[global_id] = nodes

    @classmethod
    def save_aligned_history_to_csv(cls, aligned_history: dict, filename: str, id_col_name: str = "anime_id"):
        """Сохраняет детальный файл и файл статистики."""
        partition_rows = []
        stats_rows = []

        for year, mapping in aligned_history.items():
            # Детали
            for node_id, cluster_id in mapping.items():
                partition_rows.append({
                    "year": year, 
                    id_col_name: node_id, 
                    "cluster_id": cluster_id
                })
            
            # Статистика
            counts = Counter(mapping.values())
            for c_id, count in counts.items():
                stats_rows.append({
                    "year": year,
                    "cl_n": c_id,
                    "number_of_nodes": count
                })
                
        # Сохраняем Main
        df_part = pd.DataFrame(partition_rows)
        df_part.to_csv(filename, index=False)
        log(f"Saved partition detail to {filename}", tag="TRACK", level="INFO")

        # Сохраняем Stats
        base, ext = os.path.splitext(filename)
        stats_filename = f"{base}_stats{ext}"
        
        df_stats = pd.DataFrame(stats_rows)
        if not df_stats.empty:
            df_stats = df_stats.sort_values(by=['year', 'number_of_nodes'], ascending=[True, False])
        df_stats.to_csv(stats_filename, index=False)
        log(f"Saved partition stats to  {stats_filename}", tag="TRACK", level="INFO")
