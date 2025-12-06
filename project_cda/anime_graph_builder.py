import json
import ijson
import pandas as pd
from datetime import datetime
from collections import defaultdict
from itertools import combinations
import pickle
import gc
import os
import igraph as ig
from project_cda.tag_formatter import log


class AnimeGraphBuilder:

    def __init__(self, users_csv_path, user_dict_json_path, anime_csv_path=None):
        self.users_csv_path = users_csv_path
        self.user_dict_json_path = user_dict_json_path
        log(f"Anime Graph Builder initialzed for", tag="AGB", level="DEBUG")

    def get_users_by_year(self, year):
        user_df = pd.read_csv(self.users_csv_path, usecols=["username", "join_date"])
        # Быстрый парсинг дат
        user_df["join_date_parsed"] = pd.to_datetime(user_df["join_date"], errors="coerce")
        user_df["join_year"] = user_df["join_date_parsed"].dt.year

        users = set(user_df.loc[user_df["join_year"] <= year, "username"])
        log(f"Users joined until {year}: {len(users)}", tag="AGB", level="DEBUG")
        return users

    def _collect_stats(self, year, max_users):
        """
        Сбор статистики: пересечения (edges) и популярность (nodes).
        Возвращает:
          raw_intersections: dict {(id1, id2): count}
          node_counts: dict {id: count}
        """
        users_allowed = self.get_users_by_year(year)
        
        # Используем int для ключей, это быстрее чем str
        intersection_counts = defaultdict(int)
        node_counts = defaultdict(int)
        count = 0

        with open(self.user_dict_json_path, "r", encoding="utf8") as f:
            # ijson позволяет читать огромный файл потоково
            parser = ijson.kvitems(f, "")
            
            for username, anime_list in parser:
                if username not in users_allowed:
                    continue

                count += 1
                if count % 10000 == 0:
                    log(f"Processed {count} users...", tag="AGB", level="INFO")
                if max_users and count > max_users:
                    break

                filtered_anime_ids = [
                    anime_id
                    for anime_id, vote, timestamp in anime_list
                    # Быстрая проверка года через срез строки
                    if vote != 0 and 2000 < int(timestamp[:4]) <= year
                ]
                
                # Уникальные ID для combinations
                unique_anime_ids = sorted(set(filtered_anime_ids))

                # 1. Считаем популярность
                for anime_id in unique_anime_ids:
                    node_counts[anime_id] += 1

                # 2. Считаем пересечения
                for a1, a2 in combinations(unique_anime_ids, 2):
                    intersection_counts[(a1, a2)] += 1
                    
        return intersection_counts, node_counts

    def build_edges(self, year: int, max_users: int = 100000, method: str = "jaccard", threshold: float = 0):
        """
        Считает веса ребер на основе статистики.
        Возвращает:
          edges_list: list of tuples (source_id, target_id, weight)
          node_counts: dict {anime_id: count}
        """
        log(f"Building stats for {year}...", tag="AGB", level="INFO")
        raw_intersections, node_counts = self._collect_stats(year, max_users)
        
        final_edges = [] # Список кортежей для igraph [(u, v, w), ...]
        log(f"Calculating Jaccard weights: {method.upper()} (Threshold: {threshold})...", tag="AGB", level="INFO")
        if method == "raw":
            cutoff = int(threshold) if threshold > 1 else 0
            for (u, v), count in raw_intersections.items():
                if count > cutoff:
                    final_edges.append((u, v, float(count)))

        elif method == "jaccard":
            min_w = threshold if threshold > 0 else 0.001
            for (u, v), intersection in raw_intersections.items():
                size_u = node_counts[u]
                size_v = node_counts[v]
                
                union = size_u + size_v - intersection
                if union > 0:
                    w = intersection / union
                    if w > min_w:
                        final_edges.append((u, v, w))
        else:
            raise ValueError(f"Unknown method: {method}")

        log(f"Edges prepared: {len(final_edges)}", tag="AGB", level="INFO")
        return final_edges, node_counts

    def build_graph(self, edges_list, node_counts, output_path=None) -> ig.Graph:
        """
        Создает igraph.Graph из списка ребер.
        edges_list: [(u_real_id, v_real_id, weight), ...]
        """
        log("Constructing iGraph...", tag="AGB", level="INFO")        
        # 1. Собираем все уникальные узлы
        # Нам нужно маппить Real ID -> 0..N-1
        unique_nodes = set()
        for u, v, _ in edges_list:
            unique_nodes.add(u)
            unique_nodes.add(v)
        
        # Если есть изолированные узлы в node_counts, которые мы хотим оставить?
        # Обычно в графах похожести изолированные не нужны, берем только тех, кто в ребрах.
        # Но если нужно добавить популярность для всех, можно расширить unique_nodes keys из node_counts.
        
        sorted_nodes = sorted(list(unique_nodes))
        node_map = {real_id: idx for idx, real_id in enumerate(sorted_nodes)}
        
        # 2. Создаем граф
        G = ig.Graph(len(sorted_nodes), directed=False)
        
        # 3. Атрибуты узлов
        # 'name' - стандартный атрибут для лейблов в igraph
        # Преобразуем в строки, так как iGraph иногда капризен с типами в name при сохранении в gml/graphml
        G.vs["name"] = [str(nid) for nid in sorted_nodes] 
        
        # Атрибут популярности
        if node_counts:
            # get(nid, 0) на случай, если вдруг узел попал в ребра, но не в counts (невозможно в текущей логике, но безопасно)
            G.vs["popularity"] = [node_counts.get(nid, 0) for nid in sorted_nodes]

        # 4. Добавляем ребра (Векторизованно!)
        # iGraph очень быстр, если давать ему списки кортежей (id1, id2)
        edges_to_add = []
        weights = []
        
        for u, v, w in edges_list:
            edges_to_add.append((node_map[u], node_map[v]))
            weights.append(w)
            
        G.add_edges(edges_to_add)
        G.es["weight"] = weights

        log(f"Graph constructed: {G.vcount()} nodes, {G.ecount()} edges", tag="AGB", level="INFO")
        if output_path:
            self._save_graph(G, output_path)

        return G

    def sparsify_knn(self, G: ig.Graph, k: int = 20, output_path=None) -> ig.Graph:
        """
        Оставляет Top-K соседей для каждого узла.
        В iGraph это можно сделать намного быстрее, чем в NetworkX.
        """
        log(f"Running KNN (k={k})...", tag="AGB", level="INFO")
        # Если граф маленький, можно делать через матрицу смежности, но для больших лучше итерироваться.
        # Мы создадим список ID ребер, которые нужно ОСТАВИТЬ.
        
        edges_to_keep = set()
        
        # G.vs - итератор по узлам
        # Для ускорения используем внутренние индексы (0..N)
        for v_idx in range(G.vcount()):
            # Получаем инцидентные ребра (их ID)
            edge_ids = G.incident(v_idx, mode="ALL")
            if not edge_ids:
                continue
            
            # Получаем веса этих ребер
            # G.es[edge_ids] возвращает последовательность ребер
            # Получение атрибутов списком быстрее цикла
            weights = G.es[edge_ids]["weight"]
            
            # Нам нужны пары (edge_id, weight)
            # Сортируем по весу убыванию
            ew_pairs = sorted(zip(edge_ids, weights), key=lambda x: x[1], reverse=True)
            
            # Берем топ K ребер
            top_k = ew_pairs[:k]
            
            # Добавляем ID ребер в сет "на сохранение"
            for eid, _ in top_k:
                edges_to_keep.add(eid)
        
        log(f"KNN filtering: keeping {len(edges_to_keep)} out of {G.ecount()} edges", tag="AGB", level="INFO")
        
        # Создаем подграф
        # subgraph_edges создает новый граф, содержащий только указанные ребра и ВСЕ узлы
        knn_G = G.subgraph_edges(list(edges_to_keep), delete_vertices=False)
        
        # Опционально: удалить изолированные узлы, если они появились после обрезки
        # knn_G.vs.select(_degree=0).delete() 
        
        if output_path:
            self._save_graph(knn_G, output_path)
            
        return knn_G

    def sparsify_backbone(self, G: ig.Graph, alpha=0.05, output_path=None) -> ig.Graph:
        """
        Disparity Filter (Backbone).
        Портировано под iGraph.
        """
        log(f"Running Backbone (alpha={alpha})...", tag="AGB", level="INFO")
        edges_to_keep = set()
        
        # Итерируемся по узлам
        for v_idx in range(G.vcount()):
            edge_ids = G.incident(v_idx, mode="ALL")
            k = len(edge_ids)
            if k < 2:
                continue
            
            weights = G.es[edge_ids]["weight"]
            w_total = sum(weights)
            if w_total == 0: continue
            
            # Формула Disparity Filter
            for eid, w in zip(edge_ids, weights):
                p = w / w_total
                try:
                    alpha_uv = 1 - (k - 1) * (1 - p)**(k - 1)
                except (ZeroDivisionError, ArithmeticError): # защита от float issues
                    alpha_uv = 1.0
                
                if alpha_uv < alpha:
                    edges_to_keep.add(eid)
        
        log(f"Backbone filtering: keeping {len(edges_to_keep)} out of {G.ecount()} edges", tag="AGB", level="INFO")

        backbone_G = G.subgraph_edges(list(edges_to_keep), delete_vertices=False)
        
        if output_path:
            self._save_graph(backbone_G, output_path)
            
        return backbone_G

    def _save_graph(self, G: ig.Graph, path):
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            
        # iGraph pickle — это просто пикл объекта Graph.
        # Это самый быстрый и надежный способ для Python-to-Python
        with open(path, "wb") as f:
            pickle.dump(G, f)
        
        # Если нужна совместимость с Gephi, можно сохранить в GML или GraphML
        # G.write_graphml(str(path).replace('.pickle', '.graphml'))

        log(f"Graph saved to: {path}", tag="AGB", level="INFO")

# --- Пример использования ---
# if __name__ == "__main__":
    # builder = AnimeGraphBuilder("users.csv", "users_dict.json")
    # edges, counts = builder.build_edges(2015, method="jaccard", threshold=0.05)
    # g = builder.build_graph(edges, counts, output_path="graphs/2015_full.pickle")
    # g_knn = builder.sparsify_knn(g, k=10, output_path="graphs/2015_knn.pickle")
    # pass