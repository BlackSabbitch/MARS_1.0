import networkx as nx
import numpy as np

class GraphActionsNX:

    @staticmethod
    def get_graph(path):
        """Load weighted graph from GEXF."""
        return nx.read_gexf(path)

    @staticmethod
    def add_distance_attribute(G, weight_attr="weight"):
        """
        Convert weight to distance.
        If weight = strength, then distance = 1/weight.
        If weight is already a distance, skip.
        """
        G = G.copy()
        for u, v, d in G.edges(data=True):
            w = d.get(weight_attr, 1.0)
            # avoid division by zero
            if w > 0:
                d["dist"] = 1.0 / w
            else:
                d["dist"] = 1e9  # huge distance fallback
        return G

    @staticmethod
    def get_lcc(G):
        """Return the largest connected component as a subgraph."""
        if G.number_of_nodes() == 0:
            return None

        if nx.is_connected(G):
            return G
        else:
            lcc_nodes = max(nx.connected_components(G), key=len)
            return G.subgraph(lcc_nodes).copy()

    @staticmethod
    def weighted_degree_stats(G, weight_attr="weight"):
        """
        Compute weighted degree (node strength).
        degree[node] = sum of weights on incident edges.
        """
        if G.number_of_nodes() == 0:
            return np.nan, np.nan

        strengths = [G.degree(n, weight=weight_attr) for n in G.nodes()]
        strengths = np.array(strengths, dtype=float)

        avg_strength = strengths.mean()
        std_strength = strengths.std()

        return avg_strength, std_strength

    @staticmethod
    def weighted_path_metrics(G_lcc):
        """
        Compute weighted shortest-path metrics.
        Uses 'dist' attribute, not 'weight'.
        """
        try:
            avg_path = nx.average_shortest_path_length(G_lcc, weight="dist")
        except Exception:
            avg_path = np.nan

        try:
            diameter = nx.diameter(G_lcc, e=None, usebounds=False)   # WARNING: NetworkX ignores weight
            # NetworkX *does not support weighted diameter* → computes UNWEIGHTED diameter
        except Exception:
            diameter = np.nan

        try:
            radius = nx.radius(G_lcc)   # WARNING: also UNWEIGHTED (NetworkX limitation)
        except Exception:
            radius = np.nan

        try:
            ecc = nx.eccentricity(G_lcc)   # WARNING: eccentricity is UNWEIGHTED in NetworkX
            ecc_mean = np.mean(list(ecc.values()))
        except Exception:
            ecc_mean = np.nan

        return avg_path, diameter, radius, ecc_mean

    @staticmethod
    def clustering_metrics(G):
        """
        NetworkX supports ONLY unweighted clustering and transitivity.
        Weighted clustering coefficient is NOT implemented.
        """
        try:
            avg_cluster = nx.average_clustering(G)
        except Exception:
            avg_cluster = np.nan

        try:
            trans = nx.transitivity(G)
        except Exception:
            trans = np.nan

        return avg_cluster, trans

    # Weighted centralities
    @staticmethod
    def get_closeness(G):
        try:
            return np.mean(list(nx.closeness_centrality(G, distance="dist").values()))
        except Exception:
            return np.nan

    @staticmethod
    def get_betweenness(G):
        try:
            return np.mean(list(nx.betweenness_centrality(G, weight="dist").values()))
        except Exception:
            return np.nan

    @staticmethod
    def get_eigenvector(G):
        try:
            return np.mean(list(nx.eigenvector_centrality(G, max_iter=2000, weight="weight").values()))
        except Exception:
            return np.nan

    @staticmethod
    def get_pagerank(G):
        try:
            return np.mean(list(nx.pagerank(G, weight="weight").values()))
        except Exception:
            return np.nan

    # Networkx does not suport weighted assortativity
    @staticmethod
    def degree_assortativity(G):
        """
        NetworkX supports ONLY unweighted degree assortativity.
        """
        try:
            return nx.degree_assortativity_coefficient(G)
        except Exception:
            return np.nan

    # Full metrics dictionary
    @staticmethod
    def get_metrics_dict(G, graph_name="Graph"):
        """Compute full metric dictionary for weighted or unweighted graphs."""

        metrics = {"Graph": graph_name}

        # -------------------------------
        # BASIC SIZE METRICS
        # -------------------------------
        metrics["Number of nodes"] = G.number_of_nodes()
        metrics["Number of edges"] = G.number_of_edges()

        # Degrees (weighted and unweighted)
        degrees = np.array([deg for n, deg in G.degree()])
        metrics["Average degree"] = float(np.mean(degrees))
        metrics["Degree std"] = float(np.std(degrees))

        # -------------------------------
        # CONNECTIVITY METRICS
        # -------------------------------
        try:
            metrics["Connected components"] = nx.number_connected_components(G)
        except Exception:
            metrics["Connected components"] = None

        G_lcc = GraphActions.get_lcc(G)
        metrics["LCC size"] = G_lcc.number_of_nodes() if G_lcc else None

        # -------------------------------
        # PATH-BASED METRICS (unweighted)
        # NetworkX does not support weighted APSP for diameter/radius.
        # -------------------------------
        if G_lcc is None:
            metrics["Avg path length (LCC)"] = None
            metrics["Diameter (LCC)"] = None
            metrics["Radius (LCC)"] = None
            metrics["Eccentricity mean (LCC)"] = None
        else:
            try:
                metrics["Avg path length (LCC)"] = nx.average_shortest_path_length(G_lcc)
            except Exception:
                metrics["Avg path length (LCC)"] = None

            try:
                metrics["Diameter (LCC)"] = nx.diameter(G_lcc)
            except Exception:
                metrics["Diameter (LCC)"] = None

            try:
                metrics["Radius (LCC)"] = nx.radius(G_lcc)
            except Exception:
                metrics["Radius (LCC)"] = None

            try:
                ecc = nx.eccentricity(G_lcc)
                metrics["Eccentricity mean (LCC)"] = float(np.mean(list(ecc.values())))
            except Exception:
                metrics["Eccentricity mean (LCC)"] = None

        # -------------------------------
        # CLUSTERING (weighted version exists)
        # -------------------------------
        try:
            metrics["Average clustering"] = nx.average_clustering(G, weight="weight")
        except Exception:
            metrics["Average clustering"] = None

        # Global transitivity (unweighted only)
        try:
            metrics["Transitivity"] = nx.transitivity(G)
        except Exception:
            metrics["Transitivity"] = None

        # -------------------------------
        # ASSORTATIVITY (unweighted only)
        # -------------------------------
        try:
            metrics["Degree assortativity"] = nx.degree_assortativity_coefficient(G)
        except Exception:
            metrics["Degree assortativity"] = None

        # -------------------------------
        # CENTRALITY (weighted supported)
        # -------------------------------

        # closeness weighted = OK
        try:
            closeness_vals = nx.closeness_centrality(G, distance="weight").values()
            metrics["Mean closeness"] = float(np.mean(list(closeness_vals)))
        except Exception:
            metrics["Mean closeness"] = None

        # betweenness weighted = OK
        try:
            betw_vals = nx.betweenness_centrality(G, weight="weight", normalized=True).values()
            metrics["Mean betweenness"] = float(np.mean(list(betw_vals)))
        except Exception:
            metrics["Mean betweenness"] = None

        # eigenvector weighted = OK
        try:
            eig_vals = nx.eigenvector_centrality(G, weight="weight", max_iter=500).values()
            metrics["Mean eigenvector"] = float(np.mean(list(eig_vals)))
        except Exception:
            metrics["Mean eigenvector"] = None

        return metrics
    
    @staticmethod
    def print_graph_metrics(G, metrics: dict):
        """
        Print all graph metrics contained in a dictionary.
        Keys = metric names, values = metric values.
        """

        print("\n======= GRAPH METRICS SUMMARY =======")

        for key, value in metrics.items():
            label = f"{key:<35}"

            if value is None:
                print(f"{label}: N/A")
                continue

            # String output (for graph name)
            if isinstance(value, str):
                print(f"{label}: {value}")
                continue

            # Numeric values
            try:
                v = float(value)
                if np.isnan(v):
                    print(f"{label}: N/A")
                elif abs(v - int(v)) < 1e-12:
                    print(f"{label}: {int(v)}")
                else:
                    print(f"{label}: {v:.4f}")
            except Exception:
                print(f"{label}: {value}")

        print("=====================================\n")


