import os
import networkx as nx
import pandas as pd

class GraphIO:

    @staticmethod
    def convert_graph_to_parquet(path: str):
        """
        Converts a GEXF or Pickle graph into Parquet format.
        Saves two Parquet files:
            <name>_nodes.parquet
            <name>_edges.parquet
        in the same directory.

        Supported formats:
            .gexf  — NetworkX GEXF
            .pkl   — Pickle with a NetworkX graph
        """
        # 1. Check extension
        ext = os.path.splitext(path)[1].lower()

        if ext == ".gexf":
            G = nx.read_gexf(path)
        elif ext == ".pkl":
            import pickle
            with open(path, "rb") as f:
                G = pickle.load(f)
        else:
            raise ValueError("Only .gexf and .pkl graph formats are supported.")

        # 2. Build DataFrames
        nodes_df = pd.DataFrame([
            {"id": n, **attrs}
            for n, attrs in G.nodes(data=True)
        ])

        edges_df = pd.DataFrame([
            {"source": u, "target": v, **attrs}
            for u, v, attrs in G.edges(data=True)
        ])

        # 3. Save Parquet files
        folder = os.path.dirname(path)
        base = os.path.splitext(os.path.basename(path))[0]

        nodes_out = os.path.join(folder, f"{base}_nodes.parquet")
        edges_out = os.path.join(folder, f"{base}_edges.parquet")

        nodes_df.to_parquet(nodes_out, index=False)
        edges_df.to_parquet(edges_out, index=False)

        print("Saved:")
        print(" ", nodes_out)
        print(" ", edges_out)

        return nodes_out, edges_out
