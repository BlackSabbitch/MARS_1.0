import json
import ijson
import pandas as pd
import networkx as nx
from datetime import datetime
from collections import defaultdict
from itertools import combinations
import pickle


class AnimeGraphBuilder:

    def __init__(self, users_csv_path, user_dict_json_path, anime_csv_path):
        self.users_csv_path = users_csv_path
        self.user_dict_json_path = user_dict_json_path
        self.anime_csv_path = anime_csv_path
        self.anime_info = self._load_anime_info()

    @staticmethod
    def safe_year(timestamp):
        try:
            return datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f").year
        except:
            return 9999999

    def _load_anime_info(self):
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
                    if entry[1] != 0 and self.safe_year(entry[2]) <= year and self.safe_year(entry[2])>2000
                ]

                for a1, a2 in combinations(sorted(set(filtered)), 2):
                    edges[(a1, a2)] += 1

        print(f"Edges built: {len(edges)}")
        return edges

    def build_graph(self, edges, weight_threshold=0, plot=False, output_path="anime_graph.gpickle"):

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
    
    def build_disparity_backbone(self, gpickle_path, weight="weight", alpha=0.05, weight_threshold=1, save_path=None):
        
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