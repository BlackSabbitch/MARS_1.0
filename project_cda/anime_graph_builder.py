import json
import ijson
import pandas as pd
import networkx as nx
from datetime import datetime
from collections import defaultdict
from itertools import combinations
import pickle
import gc
import os
import igraph as ig


class AnimeGraphBuilder:

    def __init__(self, users_csv_path, user_dict_json_path, anime_csv_path):
        self.users_csv_path = users_csv_path
        self.user_dict_json_path = user_dict_json_path
        self.anime_csv_path = anime_csv_path
        # self.anime_info = self._load_anime_info()

    # def _load_anime_info(self) -> dict:
    #     df_anime = pd.read_csv(self.anime_csv_path)

    #     anime_info = {
    #         int(row["anime_id"]): {
    #             "title": row.get("title", ""),
    #             "score": row.get("score", 0),
    #             "source": row.get("source", ""),
    #             "studio": row.get("studio", ""),
    #             "episodes": row.get("episodes", 0)
    #         }
    #         for _, row in df_anime.iterrows()
    #     }
    #     return anime_info

    def get_users_by_year(self, year):
        user_df = pd.read_csv(self.users_csv_path, usecols=["username", "join_date"])
        user_df["join_date_parsed"] = pd.to_datetime(user_df["join_date"], errors="coerce")
        user_df["join_year"] = user_df["join_date_parsed"].dt.year

        users = set(user_df.loc[user_df["join_year"] <= year, "username"])
        print(f"Users joined until {year}: {len(users)}")
        return users

    def build_graph(self, edges, node_counts=None, output_path=None) -> nx.Graph:

        G = nx.Graph()

        for (a1, a2), w in edges.items():
            G.add_edge(a1, a2, weight=w)

        print(f"Graph constructed: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        for node in G.nodes():
            nx.set_node_attributes(G, {node: {"title": str(node)}})   # Если нет информации, просто ставим ID в качестве названия

        # 3. [НОВОЕ] Добавляем популярность (Total Voters)
        if node_counts is not None:
            # node_counts это dict {anime_id: count}
            # Мы записываем это в атрибут 'popularity'
            nx.set_node_attributes(G, node_counts, "popularity")

        if output_path:
            self._save_graph(G, output_path)

        return G

    def _collect_stats(self, year, max_users):
        """
        INTERNAL METHOD: Тяжелая работа по чтению и подсчету.
        Ничего не знает о формулах весов. Просто считает сырые факты.
        """
        users_allowed = self.get_users_by_year(year)
        
        intersection_counts = defaultdict(int)
        node_counts = defaultdict(int)
        count = 0

        with open(self.user_dict_json_path, "r", encoding="utf8") as f:
            parser = ijson.kvitems(f, "")
            
            for username, anime_list in parser:
                if username not in users_allowed:
                    continue

                count += 1
                if count % 5000 == 0: # Чуть реже спамим в консоль
                    print(f"Processed {count} users...")
                if count > max_users:
                    break

                # Твоя оптимизированная фильтрация
                filtered_anime_ids = [
                    anime_id
                    for anime_id, vote, timestamp in anime_list
                    # Быстрая проверка года через срез строки
                    if vote != 0 and 2000 < int(timestamp[:4]) <= year
                ]
                
                # Убираем дубли (на всякий случай) и сортируем для combinations
                unique_anime_ids = sorted(set(filtered_anime_ids))

                # 1. Считаем популярность (нужно и для Jaccard, и для Raw как атрибут)
                for anime_id in unique_anime_ids:
                    node_counts[anime_id] += 1

                # 2. Считаем пересечения (Raw weight)
                for a1, a2 in combinations(unique_anime_ids, 2):
                    intersection_counts[(a1, a2)] += 1
                    
        return intersection_counts, node_counts

    def build_edges(self, year: int, max_users: int = 100000, method: str = "jaccard", threshold: float = 0):   # -> dict, dict
        """
        Публичный метод. Выбирает стратегию расчета весов.
        Arguments:
            method: 'jaccard' | 'raw'
            threshold: минимальный вес ребра (для Jaccard - от 0 до 1, для Raw - кол-во пересечений)
        """
        print(f"Building stats for {year}...")
        
        # 1. Запускаем тяжелый сборщик один раз
        raw_intersections, node_counts = self._collect_stats(year, max_users)
        
        final_edges = {}
        print(f"Calculating weights: {method.upper()} (Threshold: {threshold})...")

        # 2. Быстрый постпроцессинг в памяти
        if method == "raw":
            # Для Raw threshold — это минимальное кол-во общих групп (например, 2 или 5)
            # Если threshold=0, берем всё.
            cutoff = int(threshold) if threshold > 1 else 0

            for pair, count in raw_intersections.items():
                if count > cutoff: 
                    final_edges[pair] = float(count)

        elif method == "jaccard":
            # Для Jaccard threshold — это доля (например, 0.05)
            # Дефолтная защита от шума (хотя бы 0.001), если threshold передан как 0
            min_w = threshold if threshold > 0 else 0.001

            for (u, v), intersection in raw_intersections.items():
                size_u = node_counts[u]
                size_v = node_counts[v]
                
                union = size_u + size_v - intersection
                if union > 0:
                    w = intersection / union
                    if w > min_w: # Отсечка шума
                        final_edges[(u, v)] = w
        
        else:
            raise ValueError(f"Unknown method: {method}")

        print(f"Edges built: {len(final_edges)}")
        
        # Возвращаем и ребра, и популярность (она нужна всегда!)
        return final_edges, node_counts

    def sparsify_backbone(self, G: ig.Graph, weight="weight", alpha=0.05, output_path=None) -> nx.Graph:
        print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        print(f"Running Backbone (alpha={alpha})...")
        B = nx.Graph()
        B.add_nodes_from(G.nodes(data=True))
        
        for u in G.nodes():
            neighbors = list(G[u].items())
            k = len(neighbors)
            if k < 2:
                continue
                
            w_total = sum(d.get(weight, 1.0) for _, d in neighbors)
            if w_total == 0:
                continue
            
            for v, d in neighbors:
                w = d.get(weight, 1.0)
                p = w / w_total
                
                try:
                    alpha_uv = 1 - (k - 1) * (1 - p)**(k - 1)
                except ZeroDivisionError:
                    alpha_uv = 1.0
                
                if alpha_uv < alpha:
                    B.add_edge(u, v, **d)
        
        print(f"Backbone result: {B.number_of_edges()} edges")

      
        if output_path:
            self._save_graph(B, output_path)

        return B

    def sparsify_knn(self, G: ig.Graph, k: int = 20, output_path=None) -> nx.Graph:
        """
        Оставляет Top-K связей по весу для каждого узла.
        Результат — неориентированный граф.
        Ребро (u, v) существует, если v входит в топ-k у u, ИЛИ u входит в топ-k у v.
        """
        print(f"Running KNN (k={k})...")
        # Создаем новый пустой граф. Копировать весь g дорого.
        sparse_g = nx.Graph()
        
        # Копируем узлы (чтобы не потерять изолированные, если такие нужны, или атрибуты)
        sparse_g.add_nodes_from(G.nodes(data=True))
        
        for node in G.nodes():
            if len(G[node]) == 0:
                continue
                
            # Получаем список соседей: [(neighbor, weight), ...]
            # Предполагаем, что вес лежит в атрибуте 'weight'
            neighbors = [
                (neighbor, data['weight']) 
                for neighbor, data in G[node].items()
            ]
            
            # Сортируем по весу (убывание) и берем топ-K
            # Срез [:k] безопасен, даже если соседей меньше k
            top_k_neighbors = sorted(neighbors, key=lambda x: x[1], reverse=True)[:k]
            
            # Добавляем ребра в новый граф.
            # Так как граф неориентированный, повторное добавление ребра (u, v) 
            # просто перезапишет его (или проигнорируется, если без атрибутов), что нам и нужно.
            for neighbor, weight in top_k_neighbors:
                sparse_g.add_edge(node, neighbor, weight=weight)
                
        print(f"KNN result: {sparse_g.number_of_edges()} edges")

        if output_path:
            self._save_graph(sparse_g, output_path)
        
        return sparse_g

    def _save_graph(self, G, path):
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(G, f)
        print(f"Graph saved to: {path}")
