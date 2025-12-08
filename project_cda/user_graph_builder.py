import json
import ijson
import pandas as pd
import numpy as np
import igraph as ig
from datetime import datetime
from collections import defaultdict
from itertools import combinations
import pickle


class UserGraphBuilder:

    def __init__(self, user_dict_json_path, anime_dict_json_path):
        self.anime_dict_json_path = anime_dict_json_path
        self.user_dict_json_path = user_dict_json_path

    @staticmethod
    def safe_year(timestamp):
        try:
            return datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f").year
        except:
            return 9999999

    def get_users_in_iqr(self, q_low, q_high):

        user_counts = []

        with open(self.user_dict_json_path, "r", encoding="utf8") as f:
            parser = ijson.kvitems(f, "")
            for user_id, records in parser:
                count = sum(1 for _, score, _ in records if score != 0)
                user_counts.append((user_id, count))

   
        counts_array = np.array([c for _, c in user_counts])
        q1 = np.percentile(counts_array, q_low)
        q3 = np.percentile(counts_array, q_high)
   
        allowed_users = {
            user_id
            for user_id, count in user_counts
            if q1 <= count <= q3
        }

        return allowed_users


    def build_edges(self, year, cut, q_low = 25, q_high = 75):
        edges = defaultdict(int)
        count = 0
        allowed_users = self.get_users_in_iqr(q_low, q_high)

        with open(self.anime_dict_json_path, "r", encoding="utf8") as f:
            parser = ijson.kvitems(f, "")
            
            for anime_id, usernames in parser:
                if len(usernames)>cut:
                    continue

                count += 1


                if count%100 == 0:
                    print (f"Processed {count} anime entries")

                filtered = [
                    entry[0]
                    for entry in usernames
                    if entry[0] in allowed_users 
                        and entry[1] != 0 
                        and self.safe_year(entry[2]) <= year 
                        and self.safe_year(entry[2])>cut
                ]

                
                for a1, a2 in combinations(sorted(set(filtered)), 2):
                    edges[(a1, a2)] += 1

        print(f"Edges built: {len(edges)}")
        return edges

    def build_graph(self, edges, weight_threshold=0, plot=False, output_path="user_graph.gpickle"):

        nodes = set()
        for (u1, u2), w in edges.items():
            if w > weight_threshold:
                nodes.add(u1)
                nodes.add(u2)

        nodes = list(nodes)
        node_index = {node: idx for idx, node in enumerate(nodes)}

        edge_list = []
        edge_weights = []
        for (u1, u2), w in edges.items():
            if w > weight_threshold:
                edge_list.append((node_index[u1], node_index[u2]))
                edge_weights.append(w)

        G = ig.Graph()
        G.add_vertices(len(nodes))
        G.vs["name"] = nodes
        G.add_edges(edge_list)
        G.es["weight"] = edge_weights

        print(f"Graph nodes: {G.vcount()}, edges: {G.ecount()}")

       
        if plot==True:
            with open(output_path, "wb") as f:
                pickle.dump(G, f)
            print(f"Graph saved to {output_path}")

        return G
    
    