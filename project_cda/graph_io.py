import os
import json
import pickle
import pandas as pd
import networkx as nx
from networkx.readwrite import json_graph
import igraph as ig


class GraphIO:

    @staticmethod
    def detect_format(path: str):
        ext = os.path.splitext(path)[1].lower()
        return ext

    @staticmethod
    def load(path: str):
        ext = GraphIO.detect_format(path)

        if ext == ".gexf":
            return nx.read_gexf(path)

        if ext == ".graphml":
            return nx.read_graphml(path)

        if ext in (".pkl", ".pickle", ".gpickle"):
            with open(path, "rb") as f:
                return pickle.load(f)

        if ext == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return json_graph.node_link_graph(data)

        if ext == ".parquet":
            # expects <name>_nodes.parquet + <name>_edges.parquet
            base = path.replace("_nodes.parquet", "")
            nodes = pd.read_parquet(base + "_nodes.parquet")
            edges = pd.read_parquet(base + "_edges.parquet")

            G = nx.Graph()
            for _, row in nodes.iterrows():
                attrs = row.to_dict()
                node_id = str(attrs.pop("id"))
                G.add_node(node_id, **attrs)

            for _, row in edges.iterrows():
                attrs = row.to_dict()
                u = str(attrs.pop("source"))
                v = str(attrs.pop("target"))
                G.add_edge(u, v, **attrs)

            return G

        if ext == ".csv":
            # expects <name>_nodes.csv + <name>_edges.csv
            base = path.replace("_nodes.csv", "")
            nodes = pd.read_csv(base + "_nodes.csv")
            edges = pd.read_csv(base + "_edges.csv")

            G = nx.Graph()
            for _, row in nodes.iterrows():
                attrs = row.to_dict()
                node_id = str(attrs.pop("id"))
                G.add_node(node_id, **attrs)

            for _, row in edges.iterrows():
                attrs = row.to_dict()
                u = str(attrs.pop("source"))
                v = str(attrs.pop("target"))
                G.add_edge(u, v, **attrs)

            return G

        raise ValueError(f"Unsupported file format: {ext}")

    @staticmethod
    def save(G: nx.Graph, path: str):
        ext = GraphIO.detect_format(path)

        # GEXF
        if ext == ".gexf":
            nx.write_gexf(G, path)
            return path

        # GRAPHML
        if ext == ".graphml":
            nx.write_graphml(G, path)
            return path

        # PICKLE / G PICKLE
        if ext in (".pkl", ".pickle", ".gpickle"):
            with open(path, "wb") as f:
                pickle.dump(G, f)
            return path

        # JSON node-link
        if ext == ".json":
            data = json_graph.node_link_data(G)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            return path

        # PARQUET
        if ext == ".parquet":

            base = path.replace(".parquet", "")

            nodes = []
            for n, attrs in G.nodes(data=True):
                nodes.append({"id": n, **attrs})

            edges = []
            for u, v, attrs in G.edges(data=True):
                edges.append({"source": u, "target": v, **attrs})

            pd.DataFrame(nodes).to_parquet(base + "_nodes.parquet", index=False)
            pd.DataFrame(edges).to_parquet(base + "_edges.parquet", index=False)

            return base + "_nodes.parquet"

        # CSV
        if ext == ".csv":

            base = path.replace(".csv", "")

            nodes = []
            for n, attrs in G.nodes(data=True):
                nodes.append({"id": n, **attrs})

            edges = []
            for u, v, attrs in G.edges(data=True):
                edges.append({"source": u, "target": v, **attrs})

            pd.DataFrame(nodes).to_csv(base + "_nodes.csv", index=False)
            pd.DataFrame(edges).to_csv(base + "_edges.csv", index=False)

            return base + "_nodes.csv"

        raise ValueError(f"Cannot save to unsupported format: {ext}")

    @staticmethod
    def convert(path_in: str, path_out: str):
        G = GraphIO.load(path_in)
        return GraphIO.save(G, path_out)

    @staticmethod
    def load_as_igraph(path: str):
        G_nx = GraphIO.load(path)

        nodes = list(G_nx.nodes())
        mapping = {node: i for i, node in enumerate(nodes)}

        edges = []
        weights = []

        for u, v, attrs in G_nx.edges(data=True):
            edges.append((mapping[u], mapping[v]))
            weights.append(float(attrs.get("weight", 1.0)))

        g = ig.Graph()
        g.add_vertices(len(nodes))
        g.add_edges(edges)
        g.es["weight"] = weights
        g.vs["nx_id"] = nodes

        # copy node attributes
        for attr in G_nx.nodes[nodes[0]].keys():
            g.vs[attr] = [G_nx.nodes[n].get(attr, None) for n in nodes]

        return g

    @staticmethod
    def save_igraph(g: ig.Graph, path: str):
        # convert to networkx
        G_nx = nx.Graph()

        for v in g.vs:
            node_id = str(v["nx_id"])
            attrs = {k: v[k] for k in g.vs.attributes() if k != "nx_id"}
            G_nx.add_node(node_id, **attrs)

        for e in g.es:
            u = str(g.vs[e.tuple[0]]["nx_id"])
            v = str(g.vs[e.tuple[1]]["nx_id"])
            G_nx.add_edge(u, v, weight=float(e["weight"]))

        return GraphIO.save(G_nx, path)
    
    @staticmethod
    def save_igraph_to_pickle(g, path_out: str):
        """
        Convert igraph -> NetworkX -> save as .pkl
        """
        G_nx = nx.Graph()

        # nodes
        for v in g.vs:
            nx_id = str(v["nx_id"])
            attrs = {k: v[k] for k in g.vs.attributes() if k != "nx_id"}
            G_nx.add_node(nx_id, **attrs)

        # edges
        for e in g.es:
            u_idx, v_idx = e.tuple
            u_id = str(g.vs[u_idx]["nx_id"])
            v_id = str(g.vs[v_idx]["nx_id"])
            weight = float(e["weight"])
            G_nx.add_edge(u_id, v_id, weight=weight)

        with open(path_out, "wb") as f:
            pickle.dump(G_nx, f)

        return path_out

    @staticmethod
    def save_igraph_to_gexf(g, path_out: str):
        """
        Convert igraph -> NetworkX -> save as .gexf
        """
        G_nx = nx.Graph()

        # nodes
        for v in g.vs:
            nx_id = str(v["nx_id"])
            attrs = {k: v[k] for k in g.vs.attributes() if k != "nx_id"}
            G_nx.add_node(nx_id, **attrs)

        # edges
        for e in g.es:
            u_idx, v_idx = e.tuple
            u_id = str(g.vs[u_idx]["nx_id"])
            v_id = str(g.vs[v_idx]["nx_id"])
            weight = float(e["weight"])
            G_nx.add_edge(u_id, v_id, weight=weight)

        nx.write_gexf(G_nx, path_out)
        return path_out
    
    @staticmethod
    def convert_pickle_to_parquet(path: str):
        """
        Convert a NetworkX .pkl / .gpickle graph into Parquet format.
        Creates two files:
            <base>_nodes.parquet
            <base>_edges.parquet
        """

        ext = os.path.splitext(path)[1].lower()
        if ext not in [".pkl", ".gpickle"]:
            raise ValueError("convert_pickle_to_parquet only accepts .pkl or .gpickle")

        with open(path, "rb") as f:
            G = pickle.load(f)

        if not isinstance(G, nx.Graph):
            raise ValueError("The pickle file does not contain a NetworkX graph")

        base = os.path.splitext(path)[0]

        # ------------------------------
        # NODES -> Parquet
        # ------------------------------
        nodes_data = []
        for n, attrs in G.nodes(data=True):
            d = {"id": str(n)}
            d.update(attrs)
            nodes_data.append(d)

        df_nodes = pd.DataFrame(nodes_data)
        nodes_out = base + "_nodes.parquet"
        df_nodes.to_parquet(nodes_out, index=False)

        # ------------------------------
        # EDGES -> Parquet
        # ------------------------------
        edges_data = []
        for u, v, attrs in G.edges(data=True):
            d = {
                "src": str(u),
                "dst": str(v)
            }
            d.update(attrs)  # includes weights
            edges_data.append(d)

        df_edges = pd.DataFrame(edges_data)
        edges_out = base + "_edges.parquet"
        df_edges.to_parquet(edges_out, index=False)

        print(f"Converted to Parquet:")
        print(f"   Nodes → {nodes_out}")
        print(f"   Edges → {edges_out}")

        return nodes_out, edges_out
    
    
