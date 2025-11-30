import pickle
import igraph as ig
import numpy as np
import os
import networkx as nx
import pandas as pd

from project_cda.graph_io import GraphIO


class GraphActionsIG:
    """
    Unified class for weighted graph analytics using igraph.
    Supports:
        • weighted diameter
        • weighted radius
        • weighted eccentricity
        • weighted closeness
        • weighted betweenness
        • weighted eigenvector
        • weighted clustering
        • weighted transitivity
        • k-core
        • connected components
    """

    @staticmethod
    def load_graph(path: str):
        """
        Load any supported graph format using GraphIO.
        Always returns a NetworkX graph.
        """
        return GraphIO.load(path)

    @staticmethod
    def convert_nx_to_igraph(G_nx: nx.Graph):
        """
        Convert a NetworkX graph (with edge weights) to igraph.
        """

        nodes = list(G_nx.nodes())
        mapping = {node: idx for idx, node in enumerate(nodes)}

        edges = []
        weights = []

        for u, v, attrs in G_nx.edges(data=True):
            edges.append((mapping[u], mapping[v]))
            weights.append(float(attrs.get("weight", 1.0)))

        g = ig.Graph()
        g.add_vertices(len(nodes))
        g.add_edges(edges)
        g.es["weight"] = weights

        # copy node attributes
        for attr in G_nx.nodes[nodes[0]].keys():
            g.vs[attr] = [G_nx.nodes[n].get(attr, None) for n in nodes]

        g.vs["nx_id"] = nodes  # preserve original node IDs

        return g, mapping
   
    @staticmethod
    def get_basic_metrics(g: ig.Graph) -> dict:
        degrees = g.strength(weights=g.es["weight"]) if "weight" in g.es.attribute_names() else g.degree()

        return {
            "Number of nodes": g.vcount(),
            "Number of edges": g.ecount(),
            "Average degree (weighted)": float(np.mean(degrees)),
            "Degree std (weighted)": float(np.std(degrees)),
            "Connected components": len(g.components()),
            "LCC size": max(len(c) for c in g.components())
        }

    @staticmethod
    def get_path_metrics(g: ig.Graph) -> dict:
        comps = g.components()
        lcc_vertices = comps.giant().vs.indices
        g_lcc = g.subgraph(lcc_vertices)

        try:
            diam = g_lcc.diameter(weights="weight")
        except:
            diam = None

        try:
            rad = g_lcc.radius(weights="weight")
        except:
            rad = None

        try:
            ecc = g_lcc.eccentricity(weights="weight")
            ecc_mean = float(np.mean(ecc))
        except:
            ecc_mean = None

        try:
            apl = g_lcc.average_path_length(weights="weight")
        except:
            apl = None

        return {
            "Weighted diameter": diam,
            "Weighted radius": rad,
            "Mean eccentricity (weighted)": ecc_mean,
            "Avg path length (weighted LCC)": apl
        }

   
    @staticmethod
    def get_centrality_metrics(g: ig.Graph) -> dict:
        try:
            closeness = g.closeness(weights="weight")
            closeness_mean = float(np.mean(closeness))
        except:
            closeness_mean = None

        try:
            betw = g.betweenness(weights="weight", directed=False)
            betw_mean = float(np.mean(betw))
        except:
            betw_mean = None

        try:
            eig = g.eigenvector_centrality(weights="weight")
            eig_mean = float(np.mean(eig))
        except:
            eig_mean = None

        return {
            "Mean closeness (weighted)": closeness_mean,
            "Mean betweenness (weighted)": betw_mean,
            "Mean eigenvector (weighted)": eig_mean
        }

    @staticmethod
    def get_structural_metrics(g: ig.Graph) -> dict:
        try:
            clustering = g.transitivity_local_undirected(weights=g.es["weight"], mode="zero")
            clustering_mean = float(np.mean(clustering))
        except:
            clustering_mean = None

        try:
            global_trans = g.transitivity_undirected(weights=g.es["weight"])
        except:
            global_trans = None

        try:
            core_num = g.coreness()
            core_mean = float(np.mean(core_num))
        except:
            core_mean = None

        return {
            "Clustering coefficient mean (weighted)": clustering_mean,
            "Transitivity (weighted)": global_trans,
            "Core number mean": core_mean
        }


    @staticmethod
    def get_full_metrics_dict(g: ig.Graph, graph_name: str = "Graph") -> dict:
        metrics = {"Graph": graph_name}

        metrics.update(GraphActionsIG.get_basic_metrics(g))
        metrics.update(GraphActionsIG.get_path_metrics(g))
        metrics.update(GraphActionsIG.get_centrality_metrics(g))
        metrics.update(GraphActionsIG.get_structural_metrics(g))

        return metrics

    @staticmethod
    def print_graph_metrics(metrics: dict):
        print("\n======= Weighted Graph metrics summary =======")
        for key, value in metrics.items():
            label = f"{key:<40}"

            if value is None:
                print(f"{label}: N/A")
                continue

            if isinstance(value, str):
                print(f"{label}: {value}")
                continue

            try:
                v = float(value)
                if np.isnan(v):
                    print(f"{label}: N/A")
                elif abs(v - int(v)) < 1e-12:
                    print(f"{label}: {int(v)}")
                else:
                    print(f"{label}: {v:.4f}")
            except:
                print(f"{label}: {value}")
        print("=====================================\n")

    @staticmethod
    def detect_community_leiden(path: str, save_format="pkl"):
        G_nx = GraphActionsIG.load_graph(path)
        g, mapping = GraphActionsIG.convert_nx_to_igraph(G_nx)

        cl = g.community_leiden(weights="weight")
        g.vs["community"] = cl.membership

        base = os.path.splitext(path)[0]

        if save_format.lower() == "pkl":
            out_path = base + "_leiden.pkl"
            GraphIO.save_igraph_to_pickle(g, out_path)
        else:
            out_path = base + "_leiden.gexf"
            GraphIO.save_igraph_to_gexf(g, out_path)

        print(f"Leiden communities saved to: {out_path}")
        return cl, out_path

    @staticmethod
    def detect_community_infomap(path: str, save_format="pkl"):
        G_nx = GraphActionsIG.load_graph(path)
        g, mapping = GraphActionsIG.convert_nx_to_igraph(G_nx)

        cl = g.community_infomap(edge_weights="weight")
        g.vs["community"] = cl.membership

        base = os.path.splitext(path)[0]

        if save_format.lower() == "pkl":
            out_path = base + "_infomap.pkl"
            GraphIO.save_igraph_to_pickle(g, out_path)
        else:
            out_path = base + "_infomap.gexf"
            GraphIO.save_igraph_to_gexf(g, out_path)

        print(f"Infomap communities saved to: {out_path}")
        return cl, out_path
    
    @staticmethod
    def run_community_detection_for_all(graphs_dir):
        print("\n=== Community detection pipeline ===\n")

        # find all backbone graphs (gpickle)
        files = [
            f for f in os.listdir(graphs_dir)
            if f.endswith("_backbone.gpickle")
        ]

        if not files:
            print("No backbone gpickle graphs found.")
            return

        for fname in sorted(files):
            full_path = os.path.join(graphs_dir, fname)
            print(f"\n--- Processing graph: {fname} ---")

            base = os.path.splitext(full_path)[0]

            # Output names
            out_leiden = base + "_leiden.pkl"
            out_infomap = base + "_infomap.pkl"

            # Skip already processed
            if os.path.exists(out_leiden) and os.path.exists(out_infomap):
                print("Leiden + Infomap results already exist, skipping.")
                continue

            # Leiden
            if not os.path.exists(out_leiden):
                print("Running Leiden...")
                cl_leiden, leiden_path = GraphActionsIG.detect_community_leiden(
                    path=full_path,
                    save_format="pkl"
                )
                print(f"Leiden saved: {leiden_path}")
            else:
                print("Leiden already exists -> skipping")

            # Infomap
            if not os.path.exists(out_infomap):
                print("Running Infomap...")
                cl_infomap, infomap_path = GraphActionsIG.detect_community_infomap(
                    path=full_path,
                    save_format="pkl"
                )
                print(f"Infomap saved: {infomap_path}")
            else:
                print("Infomap already exists -> skipping")

        print("\n=== Community detection pipeline completed ===\n")

    @staticmethod
    def count_communities_in_folder(graphs_dir: str, save_csv=True):
        """
        Scans folder for *_leiden.pkl and *_infomap.pkl graphs,
        loads each graph, counts number of communities,
        returns a DataFrame with statistics.

        Output CSV: community_summary.csv in the same folder.
        """

        results = []

        for fname in sorted(os.listdir(graphs_dir)):
            if fname.endswith("_leiden.pkl") or fname.endswith("_infomap.pkl"):
                full_path = os.path.join(graphs_dir, fname)

                # ---- Load graph ----
                with open(full_path, "rb") as f:
                    G = pickle.load(f)

                if not isinstance(G, nx.Graph):
                    print(f"Skipped (not a graph): {fname}")
                    continue

                # ---- Count communities ----
                communities = []

                for n, attrs in G.nodes(data=True):
                    communities.append(attrs.get("community"))

                # filter out None
                communities = [c for c in communities if c is not None]

                if len(communities) == 0:
                    num_comms = 0
                else:
                    num_comms = len(set(communities))

                # ---- Extract year and method ----
                # Example filename: anime_graph_2006_backbone_leiden.pkl
                method = "leiden" if "_leiden" in fname else "infomap"

                # extract year: find 4-digit number
                year = None
                for token in fname.split("_"):
                    if token.isdigit() and len(token) == 4:
                        year = int(token)
                        break

                results.append({
                    "filename": fname,
                    "year": year,
                    "method": method,
                    "num_communities": num_comms
                })

        # ---- Results DataFrame ----
        df = pd.DataFrame(results).sort_values(["year", "method"])

        print("\n=== COMMUNITY COUNT SUMMARY ===")
        print(df.to_string(index=False))

        # ---- Save to CSV ----
        if save_csv:
            out_path = os.path.join(graphs_dir, "community_summary.csv")
            df.to_csv(out_path, index=False, encoding="utf-8")
            print(f"\nSaved to: {out_path}")

        return df


