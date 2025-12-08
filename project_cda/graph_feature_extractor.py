import os
import pickle
import pandas as pd
import numpy as np

class GraphFeatureExtractor:
    def __init__(self, project_root, partition_file, graph_dir, output_dir, years=None):
       
        self.project_root = project_root
        self.partition_file = os.path.join(project_root, partition_file)
        self.graph_dir = os.path.join(project_root, graph_dir)
        self.output_dir = os.path.join(project_root, output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.years = years if years is not None else list(range(2006, 2019))
        
        self.partition_dict = self._load_partition()
        
    def _load_partition(self):
        
        df = pd.read_csv(self.partition_file)
        df = df.drop_duplicates(subset=["year", "username"])
        partition_dict = df.set_index(["year", "username"])["cluster_id"].to_dict()
        return partition_dict
    
    def _compute_icr_outside(self, g, cluster_arr):
        
        neighbors = [g.neighbors(v) for v in range(g.vcount())]
        icr = np.zeros(g.vcount())
        outside = np.zeros(g.vcount())
        for i in range(g.vcount()):
            neigh = neighbors[i]
            total = len(neigh)
            if total == 0:
                continue
            same = sum(cluster_arr[n] == cluster_arr[i] for n in neigh)
            icr[i] = same / total
            outside[i] = (total - same) / total
        return icr, outside

    def process_year(self, year):
        
        out_path = os.path.join(self.output_dir, f"graph_features_{year}.csv")
        if os.path.exists(out_path):
            print(f"File for {year} already exists, skipping...")
            return
        
        print(f"\n=== Processing year {year} ===")
        gpath = os.path.join(self.graph_dir, f"user_graph_{year}_thr_3_q_low5_cut_1900.gpickle")
        print(f"Loading graph for year {year}")
        
        with open(gpath, "rb") as f:
            g = pickle.load(f)
        
        nodes = g.vs["name"]
        weights = g.es["weight"]
        
        # Cluster mapping
        cluster_list = [self.partition_dict.get((year, n), None) for n in nodes]
        cluster_arr = np.array(cluster_list, dtype=object)
        
        # Metrics
        print("Degree...")
        degree = np.array(g.degree())
        
        print("Strength...")
        strength = np.array(g.strength(weights="weight"))
        
        print("Pagerank...")
        pagerank = np.array(g.pagerank(damping=0.85, weights="weight"))
        
        print("K-core...")
        core = np.array(g.coreness())
        
        print("Weighted clustering...")
        clustering = np.array(g.transitivity_local_undirected(weights="weight"))
        
        print("ICR/outside...")
        icr, outside = self._compute_icr_outside(g, cluster_arr)
        
        # Create DataFrame
        df_year = pd.DataFrame({
            "year": year,
            "username": nodes,
            "cluster_id": cluster_arr,
            "degree": degree,
            "strength": strength,
            "pagerank": pagerank,
            "kcore": core,
            "clustering": [c if not np.isnan(c) else 0 for c in clustering],
            "icr": icr,
            "outside_ratio": outside,
        })
        
        df_year.to_csv(out_path, index=False)
        print(f"File for {year} is saved.")
    
    def process_all_years(self):
        
        for year in self.years:
            self.process_year(year)
