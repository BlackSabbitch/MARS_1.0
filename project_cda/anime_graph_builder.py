import json
import ijson
import pandas as pd
import networkx as nx
from datetime import datetime
from collections import defaultdict
from itertools import combinations
import pickle
import gc


class AnimeGraphBuilder:

    def __init__(self, users_csv_path, user_dict_json_path, anime_csv_path):
        self.users_csv_path = users_csv_path
        self.user_dict_json_path = user_dict_json_path
        self.anime_csv_path = anime_csv_path
        self.anime_info = self._load_anime_info()

    @staticmethod
    def safe_year(timestamp) -> int:
        try:
            return datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f").year
        except:
            return 9999999

    def _load_anime_info(self) -> dict:
        df_anime = pd.read_csv(self.anime_csv_path)

        anime_info = {
            int(row["anime_id"]): {
                "title": row.get("title", ""),
                "score": row.get("score", 0),
                "source": row.get("source", ""),
                "studio": row.get("studio", ""),
                "episodes": row.get("episodes", 0)
            }
            for _, row in df_anime.iterrows()
        }
        return anime_info

    def get_users_by_year(self, year):
        user_df = pd.read_csv(self.users_csv_path, usecols=["username", "join_date"])
        user_df["join_date_parsed"] = pd.to_datetime(user_df["join_date"], errors="coerce")
        user_df["join_year"] = user_df["join_date_parsed"].dt.year

        users = set(user_df.loc[user_df["join_year"] <= year, "username"])
        print(f"Users joined until {year}: {len(users)}")
        return users

    def build_edges(self, year, max_users=100000):
        users_allowed = self.get_users_by_year(year)
        edges = defaultdict(int)
        count = 0

        with open(self.user_dict_json_path, "r", encoding="utf8") as f:
            parser = ijson.kvitems(f, "")
            
            for username, anime_list in parser:

                if username not in users_allowed:
                    continue

                count += 1

                if count%100 == 0:
                    print (f"Processed {count} users")

                if count > max_users:
                    break

                #print(anime_list)

                filtered = [
                    entry[0]
                    for entry in anime_list
                    if entry[1] != 0 and self.safe_year(entry[2]) <= year and self.safe_year(entry[2])>2000     # != 1970
                ]

                for a1, a2 in combinations(sorted(set(filtered)), 2):
                    edges[(a1, a2)] += 1

        print(f"Edges built: {len(edges)}")
        return edges

    def build_jaccard_edges(self, year, max_users=100000):
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
                    if count % 1000 == 0:
                        print(f"Processed {count} users...")
                    if count > max_users:
                        break
                    filtered_anime_ids = [
                        entry[0]
                        for entry in anime_list
                        if entry[1] != 0 and self.safe_year(entry[2]) <= year and self.safe_year(entry[2]) > 2000   # != 1970
                    ]
                    unique_anime_ids = sorted(set(filtered_anime_ids))
                    for anime_id in unique_anime_ids:
                        node_counts[anime_id] += 1
                    for a1, a2 in combinations(unique_anime_ids, 2):
                        intersection_counts[(a1, a2)] += 1
            print(f"Raw intersections built: {len(intersection_counts)}")
            print("Calculating Jaccard weights...")
            final_edges = {}
            for (u, v), intersection in intersection_counts.items():
                size_u = node_counts[u]
                size_v = node_counts[v]                
                union = size_u + size_v - intersection
                if union > 0:
                    weight = intersection / union
                    if weight > 0.001: 
                        final_edges[(u, v)] = weight
            print(f"Jaccard edges built: {len(final_edges)}")
            return final_edges, node_counts

    def build_graph(self, edges, weight_threshold=0, plot=False, output_path="anime_graph.gpickle") -> nx.Graph:

        G = nx.Graph()

        for (a1, a2), w in edges.items():
            if w > weight_threshold:
                G.add_edge(a1, a2, weight=w)

        print(f"Graph edges added: {G.number_of_edges()}")

        for node in G.nodes():
            if node in self.anime_info:
                nx.set_node_attributes(G, {node: self.anime_info[node]})
            else:
                nx.set_node_attributes(G, {node: {
                    "title": "",
                    "score": 0,
                    "source": "",
                    "studio": "",
                    "episodes": 0
                }})

        #degrees = dict(G.degree())
        #nx.set_node_attributes(G, degrees, "degree")

        if plot==True:
            with open(output_path, "wb") as f:
                pickle.dump(G, f)
            #nx.write_gexf(G, output_path)
            print(f"Graph saved to {output_path}")

        return G

    def sparsify_backbone(self, gpickle_path, weight="weight", alpha=0.05, weight_threshold=1, save_path=None) -> nx.Graph:
        
        with open(gpickle_path, "rb") as f:
            G = pickle.load(f)
        print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

       
        B = nx.Graph()
        B.add_nodes_from(G.nodes(data=True))
        
        for u in G.nodes():
            neighbors = [(v, d) for v, d in G[u].items() if d.get(weight, 1.0) > weight_threshold]
            k = len(neighbors)
            if k < 2:
                continue
                
            w_total = sum(d.get(weight, 1.0) for _, d in neighbors)
            
            for v, d in neighbors:
                w = d.get(weight, 1.0)
                p = w / w_total
                
                
                alpha_uv = 1 - (k - 1) * (1 - p)**(k - 1)
                
                if alpha_uv < alpha:
                    B.add_edge(u, v, **d)
        
        print(f"Backbone edges: {B.number_of_edges()}")

      
        if save_path is not None:
            with open(save_path, "wb") as f:
                pickle.dump(B, f)
            print(f"Backbone saved to {save_path}")

        return B

    def sparsify_knn(self, g: nx.Graph, k: int = 20) -> nx.Graph:
        """
        Оставляет Top-K связей по весу для каждого узла.
        Результат — неориентированный граф.
        Ребро (u, v) существует, если v входит в топ-k у u, ИЛИ u входит в топ-k у v.
        """
        # Создаем новый пустой граф. Копировать весь g дорого.
        sparse_g = nx.Graph()
        
        # Копируем узлы (чтобы не потерять изолированные, если такие нужны, или атрибуты)
        sparse_g.add_nodes_from(g.nodes(data=True))
        
        for node in g.nodes():
            if len(g[node]) == 0:
                continue
                
            # Получаем список соседей: [(neighbor, weight), ...]
            # Предполагаем, что вес лежит в атрибуте 'weight'
            neighbors = [
                (neighbor, data['weight']) 
                for neighbor, data in g[node].items()
            ]
            
            # Сортируем по весу (убывание) и берем топ-K
            # Срез [:k] безопасен, даже если соседей меньше k
            top_k_neighbors = sorted(neighbors, key=lambda x: x[1], reverse=True)[:k]
            
            # Добавляем ребра в новый граф.
            # Так как граф неориентированный, повторное добавление ребра (u, v) 
            # просто перезапишет его (или проигнорируется, если без атрибутов), что нам и нужно.
            for neighbor, weight in top_k_neighbors:
                sparse_g.add_edge(node, neighbor, weight=weight)
                
        print(f"Original edges: {g.number_of_edges()}")
        print(f"Sparse edges: {sparse_g.number_of_edges()}")
        
        return sparse_g
