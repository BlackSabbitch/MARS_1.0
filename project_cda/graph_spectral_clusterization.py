import pickle
import os
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from scipy.sparse import csgraph
from sklearn.cluster import SpectralClustering

class GraphSpectralClusterization:
    def __init__(self, gpickle_path):
        """
        Initializes the clusterizer.
        :param gpickle_path: Path to the .gpickle file containing the graph.
        """
        self.file_path = gpickle_path
        self.G = None
        self.adj_matrix = None
        self.node_list = []  # Keeps track of the fixed order of nodes
        self.labels = None
        self.results_df = None
        
        # Load the graph upon initialization
        self._load_graph()

    def _load_graph(self):
        """
        Loads the graph from the file, handling different NetworkX versions.
        NetworkX 3.0+ removed read_gpickle.
        """
        print(f"[INFO] Loading graph from {self.file_path}...")
        
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File {self.file_path} not found.")

        try:
            # For older NetworkX versions (< 3.0)
            self.G = nx.read_gpickle(self.file_path)
        except (AttributeError, NameError):
            # For newer NetworkX versions (>= 3.0) using standard pickle
            with open(self.file_path, 'rb') as f:
                self.G = pickle.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load graph: {e}")
            raise

        self.node_list = list(self.G.nodes())
        print(f"[SUCCESS] Graph loaded: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges.")

    def preprocess(self, weight_field='weight', min_weight_threshold=0):
            """
            1. Converts graph to sparse matrix.
            2. Filters weak connections.
            3. Keeps only the Giant Component.
            4. Converts indices to int32 to fix sklearn "ValueError: Only sparse matrices with 32-bit integer indices...".
            """
            print("[INFO] Converting graph to sparse matrix...")
            
            # 1. Convert the WHOLE graph to a sparse matrix 
            self.adj_matrix = nx.to_scipy_sparse_array(
                self.G, 
                nodelist=self.node_list, 
                weight=weight_field, 
                format='csr'
            )
            
            # 2. Filter directly inside the matrix 
            if min_weight_threshold > 0:
                print(f"[INFO] Filtering edges with weight <= {min_weight_threshold}...")
                self.adj_matrix.data[self.adj_matrix.data <= min_weight_threshold] = 0
                self.adj_matrix.eliminate_zeros()
                print(f"[INFO] Edges remaining in matrix: {self.adj_matrix.nnz}")

            # 3. CHECK CONNECTIVITY (Fix for "Graph is not fully connected" crash)
            n_components, labels = connected_components(self.adj_matrix, connection='weak', directed=False)
            
            if n_components > 1:
                print(f"[WARNING] Graph split into {n_components} disconnected parts after filtering.")
                print("[INFO] Keeping only the largest connected component (Giant Component)...")
                
                counts = np.bincount(labels)
                largest_comp_id = np.argmax(counts)
                mask = labels == largest_comp_id
                
                self.adj_matrix = self.adj_matrix[mask][:, mask]
                self.node_list = np.array(self.node_list)[mask].tolist()
                
                print(f"[INFO] Nodes remaining after keeping Giant Component: {len(self.node_list)}")
            else:
                print("[INFO] Graph is fully connected. Proceeding...")

            # 4. FIx for scikit-learn value error (int64 vs int32)
            # Scikit-learn's spectral embedding requires 32-bit integer indices.
            # Default numpy/scipy behavior on 64-bit systems is to use int64.
            print("[INFO] Converting matrix indices to int32 for sklearn compatibility...")
            self.adj_matrix.indices = self.adj_matrix.indices.astype(np.int32)
            self.adj_matrix.indptr = self.adj_matrix.indptr.astype(np.int32)

            return self

    # def analyze_eigengap(self, max_k=30, save_plot_path = r"C:\MariaSamosudova\Projects\UNIVER\ADB\Project\MARS_1.0\data\data_cda\plots\eigen_plot.png"):
    #     """
    #     Calculates eigenvalues and saves the plot to the specified path.
    #     Automatically creates the directory if it does not exist.
    #     """
    #     if self.adj_matrix is None:
    #         raise ValueError("Please run .preprocess() before analysis.")
        
    #     # --- NEW: Ensure directory exists ---
    #     # Get the directory name from the provided path
    #     directory = os.path.dirname(save_plot_path)
    #     # If a directory was specified (string is not empty), create it
    #     if directory:
    #         os.makedirs(directory, exist_ok=True)
    #     # ------------------------------------

    #     print(f"[INFO] Calculating top {max_k} eigenvalues...")
        
    #     # Calculate Normalized Laplacian
    #     L = csgraph.laplacian(self.adj_matrix, normed=True)
        
    #     # Calculate eigenvalues (removed ", _" fix included)
    #     eigenvalues = eigsh(L, k=max_k + 1, which='SM', return_eigenvectors=False)
    #     eigenvalues = np.sort(eigenvalues)
        
    #     # Plotting
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(range(len(eigenvalues)), eigenvalues, 'o-', markerfacecolor='red', markersize=5)
    #     plt.title('Eigengap Heuristic (Smallest Eigenvalues)')
    #     plt.xlabel('Index (k)')
    #     plt.ylabel('Eigenvalue')
    #     plt.grid(True, linestyle='--', alpha=0.7)
        
    #     # Gap calculation
    #     diffs = np.diff(eigenvalues)
    #     max_diff_idx = np.argmax(diffs)
    #     suggestion = max_diff_idx + 1
    #     print(f"[SUGGESTION] Based on max gap, optimal clusters (k): {suggestion}")
        
    #     plt.annotate(f'Max Gap (k={suggestion})', 
    #                  xy=(max_diff_idx, eigenvalues[max_diff_idx]), 
    #                  xytext=(max_diff_idx+2, eigenvalues[max_diff_idx]),
    #                  arrowprops=dict(facecolor='black', shrink=0.05))
        
    #     # Save
    #     plt.savefig(save_plot_path)
    #     print(f"[INFO] Plot saved to: {save_plot_path}")
    #     plt.close()

    def analyze_eigengap(self, max_k=30, save_plot_path = r"C:\MariaSamosudova\Projects\UNIVER\ADB\Project\MARS_1.0\data\data_cda\plots\eigen_plot.png"):
        """
        Calculates eigenvalues and saves the plot.
        Includes protection against small graphs (k >= N error).
        """
        if self.adj_matrix is None:
            raise ValueError("Run .preprocess() before analysis.")

        # --- Ensure directory exists ---
        # Only run this if a path is actually provided
        if save_plot_path:
            directory = os.path.dirname(save_plot_path)
            if directory:
                os.makedirs(directory, exist_ok=True)

        # --- SAFETY CHECK: Adjust k if graph is too small ---
        n_nodes = self.adj_matrix.shape[0]
            
        print(f"[INFO] Graph size: {n_nodes} nodes.")
            
        # If graph has 0 or 1 node, we cannot calculate eigenvalues
        if n_nodes < 2:
            print("[ERROR] Graph is too small (less than 2 nodes). Cannot run analysis.")
            return

        # Cannot calculate more eigenvalues than there are nodes
        # 'eigsh' requires k < N strictly
        k_to_calc = max_k + 1
            
        if k_to_calc >= n_nodes:
            print(f"[WARNING] Requested {k_to_calc} eigenvalues, but graph only has {n_nodes} nodes.")
            # Reduce k to be N - 1 (the maximum allowed for sparse solvers)
            k_to_calc = n_nodes - 1
            print(f"[INFO] Adjusted k to {k_to_calc}.")

        print(f"[INFO] Calculating top {k_to_calc} eigenvalues...")
            
        # Calculate Normalized Laplacian
        L = csgraph.laplacian(self.adj_matrix, normed=True)
            
        try:
            # Calculate eigenvalues
            # Note: k must be strictly less than N for 'eigsh'
            eigenvalues = eigsh(L, k=k_to_calc, which='SM', return_eigenvectors=False)
            eigenvalues = np.sort(eigenvalues)
                
            # Plotting
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(eigenvalues)), eigenvalues, 'o-', markerfacecolor='red', markersize=5)
            plt.title('Eigengap Heuristic (Smallest Eigenvalues)')
            plt.xlabel('Index (k)')
            plt.ylabel('Eigenvalue')
            plt.grid(True, linestyle='--', alpha=0.7)
                
            # Gap calculation 
            if len(eigenvalues) > 1:
                diffs = np.diff(eigenvalues)
                max_diff_idx = np.argmax(diffs)
                suggestion = max_diff_idx + 1
                print(f"[SUGGESTION] Based on max gap, optimal clusters (k): {suggestion}")
                    
                plt.annotate(f'Max Gap (k={suggestion})', 
                            xy=(max_diff_idx, eigenvalues[max_diff_idx]), 
                            xytext=(max_diff_idx, eigenvalues[max_diff_idx] + 0.1),
                            arrowprops=dict(facecolor='black', shrink=0.05))
                
            # Save
            plt.savefig(save_plot_path)
            print(f"[INFO] Plot saved to: {save_plot_path}")
            plt.close()
                
        except Exception as e:
            print(f"[ERROR] Failed to calculate eigenvalues: {e}")

    def fit(self, n_clusters=10, random_state=42):
        """
        Executes Spectral Clustering.
        Includes safety checks to prevent crashing if the graph is too small.
        """
        if self.adj_matrix is None:
            # If preprocess wasn't called manually, call it with defaults
            self.preprocess()

        # SAFETY CHECK: Check Graph Size
        n_nodes = self.adj_matrix.shape[0]
        print(f"[INFO] Nodes available for clustering: {n_nodes}")

        # Case 1: Less than 2 nodes (Impossible to cluster)
        if n_nodes < 2:
            print(f"[ERROR] Graph is too small (size={n_nodes}). Cannot perform clustering.")
            print("[TIP] Try lowering the 'min_weight_threshold' in preprocess.")
            # Create dummy labels so the code doesn't crash later
            self.labels = np.zeros(n_nodes, dtype=int)
            return self.labels

        # Case 2: Fewer nodes than requested clusters
        if n_clusters > n_nodes:
            print(f"[WARNING] Requested {n_clusters} clusters, but graph only has {n_nodes} nodes.")
            n_clusters = n_nodes
            print(f"[INFO] Automatically reduced n_clusters to {n_clusters}.")

        print(f"[INFO] Running Spectral Clustering with k={n_clusters}...")
        
        try:
            # affinity='precomputed' is crucial, we pass the adjacency matrix
            sc = SpectralClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                assign_labels='kmeans',
                random_state=random_state,
                n_jobs=-1
            )
            
            self.labels = sc.fit_predict(self.adj_matrix)
            print("[INFO] Clustering complete.")
            
        except Exception as e:
            print(f"[ERROR] Spectral Clustering failed: {e}")
            # Fallback to zeros to prevent downstream errors
            self.labels = np.zeros(n_nodes, dtype=int)

        return self.labels

    def get_results(self, include_attributes=None):
        """
        Combines clustering results with node metadata into a DataFrame.
        
        :param include_attributes: List of node attributes to extract (e.g. ['label', 'genres']).
                                   If None, attempts to extract all available attributes.
        """
        if self.labels is None:
            raise ValueError("Please run .fit() first.")

        print("[INFO] Building result DataFrame...")
        
        data = {
            'node_id': self.node_list,
            'cluster': self.labels
        }
        
        # Check attributes of the first node to see what's available
        if self.node_list:
            first_node = self.node_list[0]
            available_attrs = list(self.G.nodes[first_node].keys())
            
            attrs_to_extract = include_attributes if include_attributes else available_attrs
            
            for attr in attrs_to_extract:
                # Extract attribute for all nodes, use None if missing
                data[attr] = [self.G.nodes[n].get(attr, None) for n in self.node_list]

        self.results_df = pd.DataFrame(data)
        return self.results_df
    
    # Edges with <= 2 shared users are ignored
    # The target number of clusters
    def run_clusterization(path, weight_treshold, num_clusters):
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
               
        # --- 2. Instantiate and Load ---
        try:
            model = GraphSpectralClusterization(path)
        except FileNotFoundError:
            print(f"Error: File '{path}' not found.")
            return

        # --- 3. Preprocessing ---
        # This converts the graph to a matrix and filters noise
        model.preprocess(weight_field='weight', min_weight_threshold=weight_treshold)
        
        # --- 4. (Optional) Eigengap Analysis ---
        # Uncomment the line below to view the plot for optimal cluster number
        model.analyze_eigengap(max_k=30)
        
        # --- 5. Clustering ---
        print(f"\n--- Starting Clustering (Target k={num_clusters}) ---")
        model.fit(n_clusters=num_clusters)
        
        # --- 6. Extract Results ---
        # Fetch specific attributes to contextually understand the clusters
        attributes_to_fetch = ['label', 'genres', 'score', 'studios', 'type']
        df = model.get_results(include_attributes=attributes_to_fetch)
        
        # --- 7. Console Output & Statistics ---
        print("\n" + "="*40)
        print("       CLUSTERING REPORT")
        print("="*40)

        # Calculate statistics
        total_nodes = len(df)
        unique_clusters = df['cluster'].nunique()
        cluster_counts = df['cluster'].value_counts().sort_index()

        print(f"Total Nodes Processed: {total_nodes}")
        print(f"Total Clusters Found:  {unique_clusters}")
        print("-" * 40)
        
        # Print formatted table
        print(f"{'Cluster ID':<12} | {'Size':<8} | {'% of Total':<10}")
        print("-" * 40)
        
        for cluster_id, size in cluster_counts.items():
            percentage = (size / total_nodes) * 100
            print(f"{cluster_id:<12} | {size:<8} | {percentage:<6.2f}%")
        
        print("-" * 40)

        # --- 8. Preview (Optional) ---
        # Show the largest cluster's top items to verify logic
        largest_cluster_id = cluster_counts.idxmax()
        print(f"\nSample from Largest Cluster (ID: {largest_cluster_id}):")
        sample = df[df['cluster'] == largest_cluster_id].head(5)
        
        # Print sample without index for cleaner look
        print(sample[['label', 'genres', 'score']].to_string(index=False))


