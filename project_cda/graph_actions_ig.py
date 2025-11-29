import igraph as ig
import numpy as np
import os
import networkx as nx


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
    def load_gexf_as_igraph(path: str) -> ig.Graph:
        """
        Load GEXF using NetworkX and convert to igraph
        """
        # Load in NetworkX
        G_nx = nx.read_gexf(path)

        # Convert to igraph
        g = ig.Graph()
        g.add_vertices(list(G_nx.nodes()))

        edges = list(G_nx.edges())
        g.add_edges(edges)

        # Add weights
        if "weight" in next(iter(G_nx.edges(data=True)))[2]:
            weights = [G_nx[u][v].get("weight", 1.0) for u, v in G_nx.edges()]
        else:
            weights = [1.0] * len(edges)

        g.es["weight"] = weights

        return g

 
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
