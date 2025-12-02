from __future__ import annotations
from typing import Set, Dict
import igraph
from igraph.clustering import VertexClustering
import pandas as pd
import os
from collections import Counter



AnimeID = int

class CommunityTracker:
    @staticmethod
    def get_membership(graph: igraph.Graph, partition: VertexClustering) -> dict:
        membership = partition.membership
        part_dict = {node: membership[i] for i, node in enumerate(graph.vs['name'])}
        return part_dict

    @classmethod
    def jaccard_sets(cls, set1: Set[AnimeID], set2: Set[AnimeID]) -> float:
        """Вспомогательная функция для сравнения двух множеств узлов."""
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0

    @classmethod
    def track_communities(cls, partition_by_year: Dict[int, dict], threshold=0.1) -> dict:
        """
        Пробрасывает метки кластеров сквозь время.
        
        Args:
            partition_by_year: Словарь {year: {node_id: raw_cluster_id}}.
                            Порядок ключей не важен, функция сама их отсортирует.
            threshold: Порог схожести (Jaccard) для наследования ID.
                    
        Returns:
            Словарь {year: {node_id: consistent_cluster_id}}
        """
        
        # 1. Сортируем года, чтобы время шло линейно
        sorted_years = sorted(partition_by_year.keys())
        
        # Результирующий словарь
        aligned_partitions_by_year = {}
        
        # Глобальный счетчик уникальных ID
        next_unique_id = 0
        
        # Состояние предыдущего шага: {consistent_cluster_id: set_of_nodes}
        prev_clusters_sets = None
        
        for year in sorted_years:
            print(f"Aligning year {year}...")
            
            # Сырое разбиение текущего года: {node: raw_id}
            current_part_dict = partition_by_year[year]
            
            # Преобразуем в сеты: {raw_id: set_of_nodes}
            curr_clusters_sets = {}
            for node, comm_id in current_part_dict.items():
                if comm_id not in curr_clusters_sets:
                    curr_clusters_sets[comm_id] = set()
                curr_clusters_sets[comm_id].add(node)
                
            # --- ЛОГИКА ПЕРВОГО ГОДА ---
            if prev_clusters_sets is None:
                mapping = {} 
                for raw_id in curr_clusters_sets:
                    mapping[raw_id] = next_unique_id
                    next_unique_id += 1
                
                # Сохраняем результат
                aligned_partitions_by_year[year] = {
                    n: mapping[c_id] for n, c_id in current_part_dict.items()
                }
                
                # Готовим state для следующего шага
                prev_clusters_sets = {}
                for raw_id, nodes in curr_clusters_sets.items():
                    prev_clusters_sets[mapping[raw_id]] = nodes
                
                continue
                
            # --- ЛОГИКА ПОСЛЕДУЮЩИХ ЛЕТ ---
            mapping = {} 
            
            # Сортируем текущие кластеры по размеру (крупные забирают названия первыми)
            sorted_curr_ids = sorted(curr_clusters_sets.keys(), 
                                    key=lambda k: len(curr_clusters_sets[k]), 
                                    reverse=True)
            
            for curr_id in sorted_curr_ids:
                curr_nodes = curr_clusters_sets[curr_id]
                
                best_match_id = -1
                best_score = -1
                
                # Ищем "родителя" в прошлом году
                for prev_id, prev_nodes in prev_clusters_sets.items():
                    score = cls.jaccard_sets(curr_nodes, prev_nodes)
                    if score > best_score:
                        best_score = score
                        best_match_id = prev_id
                
                # Решаем: наследовать или создавать новый
                if best_score > threshold:
                    mapping[curr_id] = best_match_id
                else:
                    mapping[curr_id] = next_unique_id
                    next_unique_id += 1
                    
            # Сохраняем результат для текущего года
            aligned_partitions_by_year[year] = {
                n: mapping[c_id] for n, c_id in current_part_dict.items()
            }
            
            # Обновляем state (уже с новыми ID)
            prev_clusters_sets = {}
            for raw_id, nodes in curr_clusters_sets.items():
                new_id = mapping[raw_id]
                if new_id not in prev_clusters_sets:
                    prev_clusters_sets[new_id] = set()
                prev_clusters_sets[new_id].update(nodes)

        return aligned_partitions_by_year

    @classmethod
    def save_aligned_history_to_csv(cls, aligned_history: dict, filename: str):
        """
        Сохраняет два файла:
        1. Основной: year, anime_id, cluster_id
        2. Статистика: year, cl_n, number_of_nodes
        """
        # Списки для накопления данных
        partition_rows = []
        stats_rows = []

        for year, mapping in aligned_history.items():
            # 1. Данные для основного файла (детализация)
            for anime_id, cluster_id in mapping.items():
                partition_rows.append({
                    "year": year, 
                    "anime_id": anime_id, 
                    "cluster_id": cluster_id
                })
            
            # 2. Данные для файла статистики (агрегация)
            # Считаем, сколько узлов в каждом кластере в ЭТОМ году
            counts = Counter(mapping.values())
            
            for c_id, count in counts.items():
                stats_rows.append({
                    "year": year,
                    "cl_n": c_id,            # Твое название колонки
                    "number_of_nodes": count # Твое название колонки
                })
                
        # --- Сохранение основного файла ---
        df_part = pd.read_csv(filename) if os.path.exists(filename) else pd.DataFrame(partition_rows)
        # Лучше создать новый, чтобы не путаться с дописыванием
        df_part = pd.DataFrame(partition_rows)
        df_part.to_csv(filename, index=False)
        print(f"Saved partition detail to {filename}")

        # --- Сохранение файла статистики ---
        # Генерируем имя: result.csv -> result_stats.csv
        base, ext = os.path.splitext(filename)
        stats_filename = f"{base}_stats{ext}"
        
        df_stats = pd.DataFrame(stats_rows)
        # Для красоты можно отсортировать: сначала год, потом самые крупные кластеры
        df_stats = df_stats.sort_values(by=['year', 'number_of_nodes'], ascending=[True, False])
        
        df_stats.to_csv(stats_filename, index=False)
        print(f"Saved partition stats to  {stats_filename}")
