import igraph as ig
import numpy as np

class GraphMetricsIG:
    """
    Unified class for weighted graph analytics using igraph.
    Self-contained: automatically handles weight inversion for path metrics.
    Includes console logging for progress tracking.
    """

    @staticmethod
    def _ensure_distance_weights(g: ig.Graph):
        """
        Internal helper to invert weights for path-based algorithms.
        """
        if "distance" in g.es.attribute_names():
            return

        if "weight" not in g.es.attribute_names():
            return

        weights = np.array(g.es["weight"])
        epsilon = 1e-6
        safe_weights = np.where(weights <= epsilon, epsilon, weights)
        g.es["distance"] = 1.0 / safe_weights

    # =========================================================================
    # METRIC CALCULATIONS
    # =========================================================================

    @staticmethod
    def get_basic_metrics(g: ig.Graph) -> dict:
        print("  ... Calculating Basic Metrics (Nodes, Edges, Degree)...")
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
        print("  ... Calculating Path Metrics (Diameter, Radius, Eccentricity)...")
        # Ensure distances exist before calculation
        GraphMetricsIG._ensure_distance_weights(g)

        comps = g.components()
        lcc_vertices = comps.giant().vs.indices
        g_lcc = g.subgraph(lcc_vertices)

        try:
            print("      > Diameter...")
            # diam = g_lcc.diameter(weights="distance")
        except:
            diam = None

        try:
            print("      > Radius...")
            # rad = g_lcc.radius(weights="distance")
        except:
            rad = None

        try:
            print("      > Eccentricity...")
            # Eccentricity is the max distance from a node to all other nodes
            # ecc = g_lcc.eccentricity(weights="distance")
            # ecc_mean = float(np.mean(ecc))
        except:
            ecc_mean = None

        try:
            print("      > Average Path Length...")
            apl = g_lcc.average_path_length(weights="distance")
        except:
            apl = None

        return {
            "Weighted diameter (distance-based)": diam,
            "Weighted radius (distance-based)": rad,
            "Mean eccentricity (distance-based)": ecc_mean,
            "Avg path length (weighted LCC)": apl
        }
    
    @staticmethod
    def get_centrality_metrics(g: ig.Graph) -> dict:
        print("  ... Calculating Centrality Metrics (Closeness, Betweenness, Eigenvector)...")
        # Ensure distances exist before calculation
        GraphMetricsIG._ensure_distance_weights(g)

        # 1. Closeness (Distance based)
        try:
            print("      > Closeness Centrality...")
            closeness = g.closeness(weights="distance")
            closeness_mean = float(np.mean(closeness))
        except:
            closeness_mean = None

        # 2. Betweenness (Distance based)
        try:
            print("      > Betweenness Centrality (this may take time)...")
            # Added cutoff=3.0 to prevent hanging on massive graphs
            betw = g.betweenness(weights="distance", directed=False, cutoff=3.0)
            betw_mean = float(np.mean(betw))
        except:
            betw_mean = None

        # 3. Eigenvector (Strength based)
        try:
            print("      > Eigenvector Centrality...")
            eig = g.eigenvector_centrality(weights="weight", scale=True)
            eig_mean = float(np.mean(eig))
        except:
            eig_mean = None

        return {
            "Mean closeness (weighted distance)": closeness_mean,
            "Mean betweenness (weighted distance)": betw_mean,
            "Mean eigenvector (weighted strength)": eig_mean
        }

    @staticmethod
    def get_structural_metrics(g: ig.Graph) -> dict:
        print("  ... Calculating Structural Metrics (Clustering, Transitivity, Coreness)...")
        
        # 1. Clustering (Local Transitivity) - Strength based
        try:
            clustering = g.transitivity_local_undirected(weights=g.es["weight"], mode="zero")
            clustering_mean = float(np.mean(clustering))
        except:
            clustering_mean = None

        # 2. Global Transitivity
        try:
            global_trans = g.transitivity_undirected(weights=g.es["weight"])
        except:
            global_trans = None

        # 3. K-Core (Coreness)
        try:
            core_num = g.coreness()
            core_mean = float(np.mean(core_num))
            core_max = float(np.max(core_num)) if core_num else 0.0
        except:
            core_mean = None
            core_max = None
            
        # 4a. Assortativity (Degree)
        try:
            assortativity = g.assortativity_degree(directed=False)
            if np.isnan(assortativity): assortativity = None
        except:
            assortativity = None

        # 4b. Weighted Assortativity (Strength)
        try:
            strengths = g.strength(weights="weight")
            assortativity_weighted = g.assortativity(strengths, strengths, directed=False)
            if np.isnan(assortativity_weighted): assortativity_weighted = None
        except:
            assortativity_weighted = None
            
        # 5. Density
        try:
            density = g.density()
        except:
            density = None

        return {
            "Clustering coefficient mean (weighted)": clustering_mean,
            "Transitivity (weighted)": global_trans,
            "Core number mean": core_mean,
            "Max Core number": core_max,
            "Assortativity (degree)": assortativity,
            "Assortativity (strength/weighted)": assortativity_weighted,
            "Graph Density": density
        }

    @staticmethod
    def get_full_metrics_dict(g: ig.Graph, graph_name: str = "Graph") -> dict:
        print(f"\n--- Starting Full Metrics Calculation for {graph_name} ---")
        # SAFETY CHECK: Ensure the graph has the 'distance' attribute.
        GraphMetricsIG._ensure_distance_weights(g)

        metrics = {"Graph": graph_name}

        metrics.update(GraphMetricsIG.get_basic_metrics(g))
        metrics.update(GraphMetricsIG.get_path_metrics(g))
        metrics.update(GraphMetricsIG.get_centrality_metrics(g))
        metrics.update(GraphMetricsIG.get_structural_metrics(g))

        print("--- Metrics Calculation Complete ---")
        return metrics

    @staticmethod
    def print_anime_graph_metrics(metrics: dict):
        """
        Prints a formatted summary of graph metrics with descriptions.
        """
        graph_name = metrics.get("Graph", "Unknown Graph")
        
        descriptions = {
            "Number of nodes": "Total anime titles",
            "Number of edges": "Total connections (shared audience)",
            "Average degree (weighted)": "Avg connectivity (popularity)",
            "Degree std (weighted)": "Variance in popularity",
            "Connected components": "Isolated groups",
            "LCC size": "Size of main component",
            "Weighted diameter (distance-based)": "Longest shortest path (taste gap)",
            "Weighted radius (distance-based)": "Min eccentricity in LCC",
            "Mean eccentricity (distance-based)": "Avg max distance to others",
            "Avg path length (weighted LCC)": "Avg steps to reach any anime",
            "Mean closeness (weighted distance)": "Access speed to other nodes",
            "Mean betweenness (weighted distance)": "Bridge roles (connecting genres)",
            "Mean eigenvector (weighted strength)": "Influence (connected to popular)",
            "Clustering coefficient mean (weighted)": "Local cohesion (genre bubbles)",
            "Transitivity (weighted)": "Global triangle probability",
            "Core number mean": "Avg depth of embedding",
            "Max Core number": "Deepest 'mainstream' core level",
            "Assortativity (degree)": "Link correlation (-1 to 1)",
            "Assortativity (strength/weighted)": "Popularity correlation (-1 to 1)",
            "Graph Density": "Network saturation (0 to 1)"
        }

        print(f"\n{'='*15} Analysis Report: {graph_name} {'='*15}")
        print(f"{'Metric':<40} | {'Value':<12} | {'Description'}")
        print("-" * 95)

        for key, value in metrics.items():
            if key == "Graph": continue

            if value is None:
                val_str = "N/A"
            elif isinstance(value, str):
                val_str = value
            elif isinstance(value, (int, float, np.number)):
                if np.isnan(value):
                    val_str = "N/A"
                elif abs(value - int(value)) < 1e-12:
                    val_str = f"{int(value)}"
                else:
                    val_str = f"{value:.4f}" if abs(value) < 1000 else f"{value:.2f}"
            else:
                val_str = str(value)

            desc = descriptions.get(key, "")
            print(f"{key:<40} | {val_str:<12} | {desc}")

        print("=" * 95 + "\n")