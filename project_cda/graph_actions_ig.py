import pickle
import igraph as ig
from matplotlib import pyplot as plt
import numpy as np
import os
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# from graph_metrics import GraphMetricsIG 
from graph_io import GraphIO
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
    def get_graph(path: str) -> ig.Graph:
        """
        Universal graph loader from pickle-file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No file with graph: {path}")

        print(f"Loading file with graph: {os.path.basename(path)}...", flush=True)

        try:
            with open(path, "rb") as f:
                loaded_obj = pickle.load(f)
        except Exception as e:
            raise ValueError(f"Error reading pickle file: {e}")

        if isinstance(loaded_obj, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
            print(f"NetworkX graph ({type(loaded_obj).__name__}). Converting to iGraph...", flush=True)
            
            g_ig, _ = GraphActionsIG.convert_nx_to_igraph(loaded_obj)
            
        elif isinstance(loaded_obj, ig.Graph):
            print(f"Object is iGraph", flush=True)
            g_ig = loaded_obj
            
        else:
            raise TypeError(f"Not supported object type: {type(loaded_obj)}")

        if "weight" not in g_ig.es.attribute_names():
            print("No weight attribute in iGraph, metrics can be calculated incorrectly", flush=True)
        else:
            w = g_ig.es["weight"]
            print(f"Edges: {g_ig.ecount()}. Weights (min/max): {min(w):.4f} / {max(w):.4f}", flush=True)

        return g_ig
    

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
    def collect_metric_history(graph_paths_by_year: dict) -> dict:
        """
        Iterates through all years and collects scalar metrics for time-series plotting.
        Includes DEBUG prints to fix missing data issues.
        """
        from project_cda.graph_metrics import GraphMetricsIG

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
                g = GraphActionsIG.get_graph(path)
                
                # path_metrics = GraphMetricsIG.get_path_metrics(g)
                struct_metrics = GraphMetricsIG.get_structural_metrics(g)

                # --- DEBUG: Print keys for the first year so we see what's wrong ---
                if i == 1:
                    print(f"    [DEBUG] Available Path Metric Keys: {list(path_metrics.keys())}")

                # 1. Clustering
                history["clustering"][year] = struct_metrics.get("Clustering coefficient mean (weighted)")

                # 2. Avg Path Length
                # history["avg_path_len"][year] = path_metrics.get("Avg path length (weighted LCC)")

                # 3. Diameter (Try BOTH possible names)
                # diam = path_metrics.get("Weighted diameter (distance-based)")
                # if diam is None:
                #     # Try the longer name you might have used
                #     diam = path_metrics.get("Weighted diameter - longest shortest path (distance-based)")
                
                # history["diameter"][year] = diam

            except Exception as e:
                print(f"! Error on year {year}: {e}")

        return history
    
    def plot_distribution_dynamics(data_by_year, metric_name="Degree"):
        """
        Plots distributions (Degree or Strength) for multiple years on a Log-Log plot.
        Handles both Integers (Degree) and Floats (Strength).
        """
        plt.figure(figsize=(10, 7))
        
        years = sorted(data_by_year.keys())
        colors = plt.cm.viridis(np.linspace(0, 1, len(years)))

        for year, color in zip(years, colors):
            values = data_by_year[year]
            if not values: continue
            
            # Filter out zeroes (cannot log-plot 0)
            values = [v for v in values if v > 0]
            if not values: continue

            # Dynamic Binning
            min_val = min(values)
            max_val = max(values)
            
            # Create log-spaced bins. 
            # If min_val < 1 (e.g. Jaccard strength), this still works.
            bins = np.logspace(np.log10(min_val), np.log10(max_val), num=30)
            
            # Compute histogram
            hist, bin_edges = np.histogram(values, bins=bins, density=True)
            
            # Calculate centers
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Plot
            plt.plot(bin_centers, hist, marker='o', linestyle='-', 
                    label=f"{year}", color=color, alpha=0.7, markersize=4)

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(f'{metric_name} (k)', fontsize=12)
        plt.ylabel('P(k) - Probability Density', fontsize=12)
        plt.title(f'Evolution of {metric_name} Distribution (Log-Log)', fontsize=14)
        plt.legend(title="Year")
        plt.grid(True, which="both", ls="--", alpha=0.2)
        
        plt.show()

    @staticmethod
    def collect_degree_dynamics(graph_paths_by_year: dict, metric_type: str = "degree") -> dict:
        """
        Iterates through multiple graph files and collects degree (or strength) data.
        (Progress bar removed)
        """
        # Import moved here to avoid circular dependency, but tqdm is removed
        from graph_metrics import GraphMetricsIG 

        collected_data = {}
        sorted_years = sorted(graph_paths_by_year.keys())
        
        print(f"\n--- Collecting '{metric_type}' dynamics for {len(sorted_years)} years ---")

        for i, year in enumerate(sorted_years, 1):
            file_path = graph_paths_by_year[year]
            
            # Simple progress print
            print(f"  > [{i}/{len(sorted_years)}] Processing year {year}...", flush=True)
            
            try:
                g = GraphActionsIG.get_graph(file_path)
                data = GraphMetricsIG.get_degree_data(g, metric_type=metric_type)
                collected_data[year] = data
            except Exception as e:
                print(f"!! Error processing year {year}: {e}")
                collected_data[year] = []

        return collected_data
    
    @staticmethod
    def plot_degree_counts(data_by_year: dict, metric_name: str, save_dir: str):
        """
        Plots the Exact Count of nodes for each degree (Scatter Plot).
        X-Axis: Degree (k)
        Y-Axis: Number of Nodes having that degree
        """
        import os
        import matplotlib.pyplot as plt
        import numpy as np
        from collections import Counter
        
        # Ensure output directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        plt.figure(figsize=(12, 8))
        
        years = sorted(data_by_year.keys())
        colors = plt.cm.turbo(np.linspace(0, 1, len(years)))
        has_data = False
        
        for year, color in zip(years, colors):
            values = data_by_year[year]
            if not values: continue
            
            # Filter zeroes just in case
            values = [v for v in values if v > 0]
            if not values: continue
            
            has_data = True

            # 1. Count exact frequency: {Degree: Count}
            counts = Counter(values)
            
            # 2. Sort for plotting
            sorted_degrees = sorted(counts.keys())
            sorted_counts = [counts[k] for k in sorted_degrees]
            
            # 3. Plot as Scatter (dots) to show every single data point
            plt.plot(sorted_degrees, sorted_counts, 
                     marker='o', linestyle='None', 
                     label=f"{year}", color=color, alpha=0.6, markersize=3)

        if not has_data:
            print(f"Skipping plot for {metric_name}: No valid data.")
            plt.close()
            return

        # Log-Log scale is required to see the "Power Law" tail
        plt.xscale('log')
        plt.yscale('log')
        
        plt.xlabel(f'{metric_name} (k)', fontsize=12)
        plt.ylabel('Number of Nodes', fontsize=12)
        plt.title(f'{metric_name} Distribution', fontsize=14)
        plt.legend(title="Year", markerscale=3)
        plt.grid(True, which="both", ls="--", alpha=0.2)
        
        # Construct filename
        safe_metric = metric_name.split(" ")[0].lower()
        suffix = years[0] if len(years) == 1 else "evolution"
        filename = f"distribution_counts_{safe_metric}_{suffix}.png"
        save_path = os.path.join(save_dir, filename)
        
        plt.savefig(save_path, dpi=300)
        print(f"Saved plot to: {save_path}")
        plt.close()

    @staticmethod
    def generate_strength_distribution_plot(file_path: str, year: int, save_dir: str):
        """
        Loads a graph, calculates Weighted Degree (Strength), 
        rounds values to the nearest integer, and plots the distribution.
        """
        # Import inside to avoid circular dependency
        from project_cda.graph_metrics import GraphMetricsIG

        print(f"\n--- Processing Strength Distribution for {year} ---")

        # 1. Load the graph
        g = GraphActionsIG.get_graph(file_path)

        # 2. Calculate Strength (Sum of Weights) -> Returns Floats
        raw_strength = GraphMetricsIG.get_degree_data(g, metric_type="strength")

        # 3. Round to Integers
        # We must round because strength is continuous (e.g. 12.4, 12.5).
        # Rounding allows us to group similar users together for the frequency plot.
        rounded_strength = [int(round(s)) for s in raw_strength]

        # 4. Prepare dictionary for the plotter
        data_map = {
            year: rounded_strength
        }

        # 5. Call the existing plotter
        GraphActionsIG.plot_degree_counts(data_map, "Node Strength Distribution", save_dir)

    @staticmethod
    def get_user_metrics_light(g, name):
        """
        Calculates ONLY lightweight metrics (safe for large graphs).
        Skips: Betweenness, Closeness, Diameter, Path Lengths.
        """
        metrics = {}
        
        # 1. Basic Scale
        metrics['Number of nodes'] = g.vcount()
        metrics['Number of edges'] = g.ecount()
        metrics['Graph Density'] = g.density()
        
        # 2. Degree Statistics (Popularity)
        degrees = g.degree()
        metrics['Average degree'] = np.mean(degrees)
        metrics['Degree std'] = np.std(degrees)
        
        # 3. Components (Fragmentation)
        components = g.clusters()
        metrics['Connected components'] = len(components)
        metrics['LCC size'] = components.giant().vcount()
        
        # 4. Clustering (Social Cohesion) - These are usually safe
        # Global Transitivity (Ratio of triangles to triplets)
        metrics['Transitivity'] = g.transitivity_undirected() 
        # Avg Local Clustering (Avg probability that a user's friends are friends)
        metrics['Clustering coefficient mean'] = g.transitivity_avglocal_undirected()
        
        # 5. K-Core (Core Social Groups) - Fast O(m)
        core_nums = g.coreness()
        metrics['Core number mean'] = np.mean(core_nums)
        metrics['Max Core number'] = max(core_nums) if core_nums else 0
        
        # 6. Assortativity (Homophily)
        metrics['Assortativity (degree)'] = g.assortativity_degree()
        
        return metrics
    
    @staticmethod
    def print_user_graph_metrics(metrics, graph_name="User Graph"):
        """
        Prints a clean report with descriptions tailored to SOCIAL NETWORKS (Users).
        """
        row_config = [
            ("Number of nodes", "Total Active Users", "Unique users in the network"),
            ("Number of edges", "Total Interactions", "Votes for the same anime between users"),
            ("Graph Density", "Network Density", "Saturation (How connected is the community?)"),
            ("Average degree", "Avg Connections", "Avg number of friends per user"),
            ("Degree std", "Connection Variance", "Inequality in popularity"),
            ("Connected components", "Isolated Groups", "Fragmented sub-communities"),
            ("LCC size", "LCC Size", "Users in the main 'giant' community"),
            ("Transitivity", "Global Transitivity", "Probability of triangles (tight circles)"),
            ("Clustering coefficient mean", "Avg Clustering Coeff", "Local social cohesion (cliques)"),
            ("Core number mean", "Avg Core Depth", "Avg embedding in the social web"),
            ("Max Core number", "Max Core Level", "Deepest/Most central social tier"),
            ("Assortativity (degree)", "Homophily (Assortativity)", "Do popular users link to popular users?"),
        ]

        print(f"={' User Analysis: ' + graph_name + ' ':=^80}")
        print(f"{'Metric':<35} | {'Value':<10} | {'Description'}")
        print("-" * 80)

        for key, label, desc in row_config:
            val = metrics.get(key)
            
            if val is None: 
                continue
                
            if isinstance(val, (float, np.floating)):
                val_str = f"{val:.4f}"
            else:
                val_str = str(val)
                
            print(f"{label:<35} | {val_str:<10} | {desc}")
        
        print("=" * 80 + "\n")
    
######################################################################################################









       
    @staticmethod
    def detect_community_leiden(
            graph=None,
            path=None,
            save=False,
            save_format="pkl",
            log_weights=False,
            resolution=0.7,
            beta=0.01,
            objective_function="CPM",
            n_iterations=10,
            weights="weight",
            verbose=True
    ):
        if graph is not None:
            g = graph
            path_label = "<igraph-object>"
        elif path is not None:
            G_nx = GraphActionsIG.load_graph(path)
            g, mapping = GraphActionsIG.convert_nx_to_igraph(G_nx)
            path_label = path
        else:
            raise ValueError("Either `graph` or `path` must be provided.")

        # Prepare weights
        w = g.es[weights] if weights in g.es.attribute_names() else None
        if log_weights and w is not None:
            w = [np.log1p(x) for x in w]

        # Leiden
        cl = g.community_leiden(
            weights=w,
            resolution=resolution,
            beta=beta,
            objective_function=objective_function,
            n_iterations=n_iterations
        )

        g.vs["community"] = cl.membership

        if verbose:
            print("\n=== Leiden Community Detection Results ===")
            print(f"Graph: {path_label}")
            print(f"Communities: {len(set(cl.membership))}")
            print(f"Sizes: {cl.sizes()}")
            try:
                print(f"Modularity: {cl.modularity}")
            except:
                print("Modularity: N/A")
            print("==========================================\n")

        if not save:
            return {
                "clustering": cl,
                "membership": cl.membership,
                "sizes": cl.sizes(),
                "graph": g
            }

        if path is None:
            raise ValueError("Cannot save without path.")

        base = os.path.splitext(path)[0]
        out_path = f"{base}_leiden.{save_format}"

        if save_format.lower() == "pkl":
            GraphIO.save_igraph_to_pickle(g, out_path)
        else:
            GraphIO.save_igraph_to_gexf(g, out_path)

        return cl, out_path


    @staticmethod
    def detect_community_walktrap(
            graph=None,
            path: str = None,
            steps: int = 4,
            weights_attr: str = "weight",
            save: bool = True,
            save_format: str = "pkl",
            verbose: bool = True
        ):
        """
        Walktrap community detection.

        Parameters
        ----------
        graph : ig.Graph or None
            If provided, algorithm will use this graph directly.
        path : str or None
            If provided, graph will be loaded from file (.gexf / .pkl / .gpickle).
        steps : int
            Walktrap steps (default 4).
        weights_attr : str
            Edge weight attribute (default 'weight').
        save : bool
            Save result graph with "community" labels.
        save_format : str
            'pkl' or 'gexf'.
        verbose : bool
            Print detailed logs.

        Returns
        -------
        cl : igraph.clustering.VertexClustering
        out_path : str or None
        """

        # ---------------------------
        # Load or use existing graph
        # ---------------------------

        if graph is not None:
            # Already an igraph object
            if verbose:
                print("\n=== WALKTRAP community detection ===")
                print("Using igraph object provided directly.")

            g = graph
            path_label = "<igraph-object>"

        elif path is not None:
            if verbose:
                print("\n=== WALKTRAP community detection ===")
                print(f"Loading graph from: {path}")

            # Load as NetworkX
            G_nx = GraphActionsIG.load_graph(path)

            # Convert to igraph
            g, mapping = GraphActionsIG.convert_nx_to_igraph(G_nx)

            path_label = path

        else:
            raise ValueError("Either `graph` or `path` must be provided.")

        # ---------------------------
        # Run Walktrap
        # ---------------------------
        if verbose:
            print(f"Graph loaded: {g.vcount()} nodes, {g.ecount()} edges")
            print(f"Walktrap steps = {steps}")

        weights = g.es[weights_attr] if weights_attr in g.es.attribute_names() else None

        wt = g.community_walktrap(steps=steps, weights=weights)
        cl = wt.as_clustering()

        # Assign membership
        g.vs["community"] = cl.membership

        # ---------------------------
        # Print full results
        # ---------------------------
        if verbose:
            sizes = cl.sizes()
            num_comm = len(sizes)

            print("\n--- FULL WALKTRAP REPORT ---")
            print(f"Input graph: {path_label}")
            print(f"Detected communities: {num_comm}")
            print(f"Average community size: {np.mean(sizes):.2f}")
            print(f"Max community size: {np.max(sizes)}")
            print(f"Min community size: {np.min(sizes)}")

            if hasattr(cl, "modularity"):
                print(f"Modularity: {cl.modularity:.4f}")
            else:
                print("Modularity: N/A")

            print("\nCommunity sizes (all communities):")
            for idx, size in enumerate(sizes):
                print(f"  Community {idx}: size {size}")

            print("----------------------------------")

        # ---------------------------
        # Saving result
        # ---------------------------
        out_path = None
        if save and path is not None:  # Only possible if we know where to save
            base = os.path.splitext(path)[0]

            if save_format.lower() == "pkl":
                out_path = base + "_walktrap.pkl"
                GraphIO.save_igraph_to_pickle(g, out_path)
            else:
                out_path = base + "_walktrap.gexf"
                GraphIO.save_igraph_to_gexf(g, out_path)

            if verbose:
                print(f"Saved Walktrap result to: {out_path}")

        elif verbose and not save:
            print("save=False → Not saving results.")

        elif verbose and path is None:
            print("Warning: path=None → cannot save graph to disk.")

        return cl, out_path

    @staticmethod
    def detect_community_infomap(
            path: str,
            save: bool = True,
            save_format: str = "pkl",
            log_weights: bool = False,
            edge_weights: str = "weight",
            trials: int = 30,
            verbose: bool = True
    ):
        """
        Perform Infomap community detection on weighted graph (.gexf or .pkl).
        Supports:
            • log-weight scaling
            • optional saving
            • console reporting
        """

        # 1. Load graph
        G_nx = GraphActionsIG.load_graph(path)

        # 2. Convert to igraph
        g, mapping = GraphActionsIG.convert_nx_to_igraph(G_nx)

        # 3. Optionally scale weights in log-scale
        if log_weights:
            w = np.array(g.es[edge_weights], dtype=float)
            w = np.log1p(w)
            g.es[edge_weights] = w
            if verbose:
                print("Applied log-scale to weights.")

        # 4. Run Infomap (python-igraph supports only: edge_weights, trials)
        if verbose:
            print("\nRunning Infomap...")
            print(f"Trials: {trials}, Log-scale: {log_weights}")

        cl = g.community_infomap(
            edge_weights=edge_weights,
            trials=trials
        )

        # 5. Attach community membership to vertices
        g.vs["community"] = cl.membership

        # 6. Console output
        if verbose:
            print("\n=== Infomap Community Detection Results ===")
            print(f"Graph: {os.path.basename(path)}")
            print(f"Number of communities: {len(cl)}")
            print(f"Community sizes: {cl.sizes()[:50]} ...")
            try:
                print(f"Modularity: {cl.modularity}")
            except:
                print("Modularity: N/A")
            print("==========================================")

        # 7. Optional saving
        if save:
            base = os.path.splitext(path)[0]
            if save_format.lower() == "pkl":
                out_path = base + "_infomap.pkl"
                GraphIO.save_igraph_to_pickle(g, out_path)
            else:
                out_path = base + "_infomap.gexf"
                GraphIO.save_igraph_to_gexf(g, out_path)

            if verbose:
                print(f"Infomap communities saved to: {out_path}")

            return cl, out_path

        # If not saving — return results only
        return cl, None
    
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
    
    @staticmethod
    def save_communities_csv(result, threshold, out_dir, method_name="leiden"):
        """
        Saves:
        • separate CSV files for each community ≥ threshold
        • one combined CSV file for all selected communities
        """

        cl = result["clustering"]
        g = result["graph"]
        sizes = result["sizes"]
        membership = result["membership"]

        year_tag = "2018"

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        selected_ids = []

        # -----------------------------------
        # 1. Save each community separately
        # -----------------------------------
        for comm_id, size in enumerate(sizes):
            if size < threshold:
                continue

            selected_ids.append(comm_id)

            vertices = [i for i, c in enumerate(membership) if c == comm_id]
            sub = g.subgraph(vertices)

            # Convert to DataFrame
            rows = []
            for v in sub.vs:
                row = dict(v.attributes())
                row["community"] = comm_id
                rows.append(row)

            df = pd.DataFrame(rows)

            filename = os.path.join(
                out_dir,
                f"community_{year_tag}_{method_name}_{comm_id}_size_{size}.csv"
            )

            df.to_csv(filename, index=False, encoding="utf-8")

            print(f"Saved CSV: {filename} (rows={len(df)})")

        # -----------------------------------
        # 2. Save combined CSV
        # -----------------------------------
        if selected_ids:

            rows = []
            for idx, c in enumerate(membership):
                if c in selected_ids:
                    row = dict(g.vs[idx].attributes())
                    row["community"] = c
                    rows.append(row)

            df_all = pd.DataFrame(rows)

            filename_all = os.path.join(
                out_dir,
                f"communities_combined_{year_tag}_{method_name}_threshold_{threshold}.csv"
            )

            df_all.to_csv(filename_all, index=False, encoding="utf-8")
            print(f"Saved combined CSV: {filename_all} (rows={len(df_all)})")

        return selected_ids

    


