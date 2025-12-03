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

    


