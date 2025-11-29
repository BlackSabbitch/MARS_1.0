# data_preparation.py

import pandas as pd
import os
import networkx as nx
import pandas as pd
import os
import networkx as nx

class GraphDataGenerator:

    @staticmethod
    def user_anime_graph_data_generation():

        # === PATH ===
        DATA_PATH = r"C:\MariaSamosudova\Projects\UNIVER\ADB\Project\MARS_1.0\data\anime_azathoth42"
        OUTPUT_PATH = r"C:\MariaSamosudova\Projects\UNIVER\ADB\Project\MARS_1.0\data\data_cda\users_anime"
        os.makedirs(OUTPUT_PATH, exist_ok=True)

        print("Paths are ok")

        # === 1. Загрузка CSV ===
        print("Loading anime_filtered.csv ...")
        anime = pd.read_csv(os.path.join(DATA_PATH, "anime_filtered.csv"))

        print("Loading UserList.csv ...")
        users = pd.read_csv(os.path.join(DATA_PATH, "UserList.csv"))

        print("Loading UserAnimeList.csv ... (big data:)")
        ua = pd.read_csv(os.path.join(DATA_PATH, "UserAnimeList.csv"))

        print("Files has been loaded.")
        print(f"All rows in UserAnimeList: {len(ua):,}")

        # === 2. Filter actual scores ===
        print("Filter scores: keep only real scores (my_score > 0)...")
        ua = ua[ua['my_score'] > 0]
        print(f"Rows after filter by real scores: {len(ua):,}")

        # === 3. Keep only completed ===
        print("Filter records: keep only completed (my_status == 2)...")
        ua = ua[ua['my_status'] == 2]
        print(f"Rows after filter by completed: {len(ua):,}")

        # === 4. Filter of active users >=150 ===
        print("Calculate amount of completed anime by user...")
        count_by_user = ua.groupby("username")["anime_id"].count()

        print("Choose users with >= 150 anime completed...")
        active_users = count_by_user[count_by_user >= 150].index

        print(f"Found active users: {len(active_users):,}")

        print("Filter table user_anime_list by active users...")
        ua_filtered = ua[ua["username"].isin(active_users)]
        print(f"Rows after filtering in ua_filtered: {len(ua_filtered):,}")

        # === 5. Filter anime list ===
        print("Build list of used anime...")
        filtered_anime_ids = ua_filtered['anime_id'].unique()

        anime_filtered = anime[anime['anime_id'].isin(filtered_anime_ids)]
        print(f"Number of anime after filtration: {len(anime_filtered):,}")

        # === 6. Prepare list of anime nodes ===
        print("Prepare list of anime nodes...")

        anime_nodes = anime_filtered[['anime_id', 'title', 'score', 'genre', 'studio']].copy()
        anime_nodes.rename(columns={
            'anime_id': 'id',
            'title': 'label',
            'genre': 'genres',
            'studio': 'studios'
        }, inplace=True)
        anime_nodes['type'] = 'anime'

        print(f"Anime nodes: {len(anime_nodes):,}")

        # === 7. User nodes with attributes ===
        print("Prepare user nodes...")

        users_filtered = users[users["username"].isin(active_users)]

        user_nodes = users_filtered[[
            "username",
            "gender",
            "location",
            "stats_mean_score",
            "stats_episodes",
            "user_completed"
        ]].copy()

        user_nodes.rename(columns={
            "username": "id",
            "location": "country_city",
            "stats_mean_score": "mean_score",
            "stats_episodes": "episodes_watched"
        }, inplace=True)

        user_nodes["label"] = user_nodes["id"]
        user_nodes["type"] = "user"

        print(f"User nodes: {len(user_nodes):,}")

        # === 8. Join all nodes ===
        nodes = pd.concat([anime_nodes, user_nodes], ignore_index=True)
        print(f"All nodes in graph: {len(nodes):,}")

        # === 9. Edges (user → anime), weight = my_score ===
        print("Prepare edges (user → anime), weight = my_score...")

        edges = ua_filtered[['username', 'anime_id', 'my_score']].copy()
        edges.rename(columns={
            'username': 'source',
            'anime_id': 'target',
            'my_score': 'weight'
        }, inplace=True)
        edges['type'] = 'directed'

        print(f"All edges in graph: {len(edges):,}")

        # === 10. Save CSV ===
        print("Save nodes.csv and edges.csv ...")

        nodes_file = os.path.join(OUTPUT_PATH, "nodes.csv")
        edges_file = os.path.join(OUTPUT_PATH, "edges.csv")

        nodes.to_csv(nodes_file, index=False)
        edges.to_csv(edges_file, index=False)

        print("CSV files saved:")
        print(nodes_file)
        print(edges_file)

        # === 11. Export to GEXF ===
        print("Create GEXF graph...")

        G = nx.DiGraph()

        # Add nodes
        G.add_nodes_from(
            (row['id'], row.drop('id').to_dict())
            for _, row in nodes.iterrows()
        )

        # Add edges
        G.add_edges_from(
            (row['source'], row['target'], row.drop(['source', 'target']).to_dict())
            for _, row in edges.iterrows()
        )

        gexf_file = os.path.join(OUTPUT_PATH, "graph.gexf")
        nx.write_gexf(G, gexf_file, version="1.2draft")

        print("GEXF file saved:")
        print(gexf_file)

        print("\n=== DONE DONE ===")

