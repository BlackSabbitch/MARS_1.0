import os
import igraph as ig
import numpy as np
import matplotlib.pyplot as plt

from graph_actions_ig import GraphActionsIG

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

    # METRIC CALCULATIONS
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
        print("  ... Calculating Path Metrics (Skipping slow ones)...")
        # Ensure distances exist before calculation
        GraphMetricsIG._ensure_distance_weights(g)

        # Pre-set all variables to None so the return statement is safe
        diam = None
        rad = None
        ecc_mean = None
        apl = None

        # Calculating LCC is fast, so we keep this to establish the subgraph
        comps = g.components()
        lcc_vertices = comps.giant().vs.indices
        g_lcc = g.subgraph(lcc_vertices)

        # --- SLOW METRICS: COMMENTED OUT OR DISABLED ---
        
        try:
            print("      > Diameter (Longest shortest path)...")
            diam = g_lcc.diameter(weights="distance")
        except:
            diam = None

        # try:
        #     print("      > Radius...")
        #     rad = g_lcc.radius(weights="distance")
        # except:
        #     rad = None

        # try:
        #     print("      > Eccentricity...")
        #     ecc = g_lcc.eccentricity(weights="distance")
        #     ecc_mean = float(np.mean(ecc))
        # except:
        #     ecc_mean = None

        try:
            print("      > Average Path Length...")
            apl = g_lcc.average_path_length(weights="distance")
        except:
            apl = None

        return {
            "Weighted diameter - longest shortest path (distance-based)": diam,
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
            "Weighted diameter (distance-based)": "Longest shortest path (eg.: taste gap)",
            "Weighted radius (distance-based)": "Min eccentricity in LCC",
            "Mean eccentricity (distance-based)": "Avg max distance to others",
            "Avg path length (weighted LCC)": "Avg steps to reach any anime",
            "Mean closeness (weighted distance)": "Access speed to other nodes",
            "Mean betweenness (weighted distance)": "Bridge roles (connecting groups/clusters)",
            "Mean eigenvector (weighted strength)": "Influence (connected to popular)",
            "Clustering coefficient mean (weighted)": "Local cohesion (bubbles)",
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

    @staticmethod
    def collect_metric_history(graph_paths_by_year: dict) -> dict:
        """
        Iterates through all years and collects scalar metrics for time-series plotting.
        Returns a dictionary: {'clustering': {2006: 0.5, ...}, 'diameter': {2006: 5, ...}, ...}
        """
        from graph_metrics import GraphMetricsIG

        # Initialize storage for the 3 time-series metrics you asked for
        history = {
            "avg_path_len": {},
            "clustering": {},
            "diameter": {}
        }

        sorted_years = sorted(graph_paths_by_year.keys())
        print(f"\n--- Collecting Metric History for {len(sorted_years)} years ---")

        for i, year in enumerate(sorted_years, 1):
            print(f"  > [{i}/{len(sorted_years)}] Analyzing year {year}...", flush=True)
            path = graph_paths_by_year[year]
            
            try:
                # 1. Load
                g = GraphActionsIG.get_graph(path)
                
                # 2. Calculate Metrics
                # We need specific dictionaries from your GraphMetrics class
                path_metrics = GraphMetricsIG.get_path_metrics(g)
                struct_metrics = GraphMetricsIG.get_structural_metrics(g)

                # 3. Store the specific values you requested
                # Note: These keys must match what 'get_path_metrics' returns
                history["avg_path_len"][year] = path_metrics.get("Avg path length (weighted LCC)")
                history["diameter"][year] = path_metrics.get("Weighted diameter (distance-based)") # Or "Weighted diameter - longest shortest path" 
                history["clustering"][year] = struct_metrics.get("Clustering coefficient mean (weighted)")
                
            except Exception as e:
                print(f"! Error on year {year}: {e}")

        return history
    
    @staticmethod
    def get_degree_data(g: ig.Graph, metric_type: str = "degree") -> list:
        """
        Collects data for distribution plotting.
        
        Args:
            g: The igraph object.
            metric_type: 
                - "degree": Counts connections (Standard for Scale-Free checks).
                - "strength": Sums edge weights (Intensity of activity).
        """
        # Ensure we are working with the right graph type
        if metric_type == "strength":
            if "weight" in g.es.attribute_names():
                return g.strength(weights="weight")
            else:
                # Fallback if no weights exist
                print("Warning: Graph is unweighted. Returning standard degree.")
                return g.degree()
        
        # Default to simple degree (topology)
        return g.degree()
    
    
    @staticmethod
    def plot_metric_evolution(history_data, metric_key, title, ylabel, save_dir):
        """
        Plots a time-series line chart and saves it to a file.
        """
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Extract data
        data_map = history_data.get(metric_key, {})
        
        years = []
        values = []
        for year, val in sorted(data_map.items()):
            if val is not None:
                years.append(str(year))
                values.append(val)
                
        if not years:
            print(f"Skipping plot for {metric_key}: No valid data found.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(years, values, marker='o', linestyle='-', color='b', linewidth=2)
        
        plt.title(title, fontsize=14)
        plt.xlabel("Year", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Construct filename based on metric key (e.g., "history_clustering.png")
        safe_name = metric_key.replace(" ", "_").lower()
        filename = f"history_{safe_name}.png"
        save_path = os.path.join(save_dir, filename)
        
        plt.savefig(save_path, dpi=300)
        print(f"Saved plot to: {save_path}")
        plt.close() # Close memory to prevent leaks

    def plot_distribution_dynamics(data_by_year, metric_name, save_dir):
        """
        Plots distributions and saves to file.
        """
        os.makedirs(save_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 7))
        
        # Sort years for consistent coloring
        years = sorted(data_by_year.keys())
        # Use a colormap
        colors = plt.cm.viridis(np.linspace(0, 1, len(years)))

        has_data = False
        for year, color in zip(years, colors):
            values = data_by_year[year]
            if not values: continue
            
            # Filter zeroes
            values = [v for v in values if v > 0]
            if not values: continue
            
            has_data = True

            # Dynamic Binning
            min_val = min(values)
            max_val = max(values)
            bins = np.logspace(np.log10(min_val), np.log10(max_val), num=30)
            
            hist, bin_edges = np.histogram(values, bins=bins, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            plt.plot(bin_centers, hist, marker='o', linestyle='-', 
                    label=f"{year}", color=color, alpha=0.7, markersize=4)

        if not has_data:
            print(f"Skipping distribution plot: No valid data.")
            plt.close()
            return

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(f'{metric_name} (k)', fontsize=12)
        plt.ylabel('P(k) - Probability Density', fontsize=12)
        plt.title(f'{metric_name} Distribution (Log-Log)', fontsize=14)
        plt.legend(title="Year")
        plt.grid(True, which="both", ls="--", alpha=0.2)
        
        # Construct filename (e.g., "distribution_degree_2018.png")
        # If multiple years are plotted, it names it "distribution_degree_multi.png"
        if len(years) == 1:
            suffix = str(years[0])
        else:
            suffix = "evolution"
            
        safe_metric = metric_name.split(" ")[0].lower() # "Degree (2018)" -> "degree"
        filename = f"distribution_{safe_metric}_{suffix}.png"
        save_path = os.path.join(save_dir, filename)
        
        plt.savefig(save_path, dpi=300)
        print(f"Saved plot to: {save_path}")
        plt.close()