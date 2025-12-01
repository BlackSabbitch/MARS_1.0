from __future__ import annotations
from datetime import datetime
from typing import List, Tuple, Optional, Set, Iterable, Dict
from project_cda.anime_graph_builder import AnimeGraphBuilder
from time import perf_counter_ns
import networkx as nx
import numpy as np
import gc
from pathlib import Path
from dataclasses import dataclass
from collections import Counter
import pickle
import logging


Node = int  # или str, в зависимости от id в графе
Trail = List[Tuple[Node, datetime]]  # (node, unix_timestamp_seconds)
AnimeId = int
DateString = str


class TagFormatter(logging.Formatter):
    """
    Custom formatter that supports a `tag` attribute.
    If a tag is present, logs messages as:  [TAG] message
    Otherwise:                              [LEVEL] message
    """

    def format(self, record: logging.LogRecord) -> str:
        tag = getattr(record, "tag", None)
        prefix = f"[{tag}]" if tag else f"[{record.levelname}]"
        record.msg = f"{prefix} {record.msg}"
        return super().format(record)

logger = logging.getLogger("walker_logger")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = TagFormatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

def log(msg: str, *, tag: str | None = None, level=logging.INFO):
    """
    EXAMPLES:
    log(f"Loading config from {config_file_path}", tag="FOUND", level=logging.DEBUG)    | [DEBUG] --> [FOUND]
    log(f"Config loaded: {config_file_path}", tag="LOADED") | [INFO] --> [LOADED]
    log(f"{purpose} is too large for pandas → skipped", tag="SKIPPED", level=logging.WARNING)   | [WARNING] --> [SKIPPED]
    """
    logger.log(level, msg, extra={"tag": tag})


@dataclass
class RandomWalkerSettings:
    strategy: str = 'greedy'    # 'greedy' | 'probabilistic'
    prob_decay: float = 2.0   # степень убывания вероятности в стратегии probabilistic
    top_k: Optional[int] = None    # если указан, ограничить выбор первыми k соседями по весу 
    seed: Optional[int] = None


@dataclass
class GraphLoader:
    base_folder: Path
    subfolder: str = "Graphs_cleaned_95_percentile/"
    postfix: str = 'new'

    def get_graph_path(self, year: int) -> Path | None:
            """Ищет файл для конкретного года."""
            search_dir = self.base_folder / self.subfolder
            # Формируем имя файла, как ожидается: например 2015_new.gpickle
            # Если формат имени сложнее, нужно вернуть логику с listdir, но только для поиска пути
            for file_path in search_dir.glob("*.gpickle"):
                parts = file_path.stem.split("_") # stem это имя файла без расширения
                if len(parts) < 2: continue
                
                f_year_str, f_postfix = parts[-2], parts[-1]
                
                if f_postfix == self.postfix and f_year_str == str(year):
                    return file_path
            return None

    def load_one_graph(self, year: int) -> nx.Graph | None:
        path = self.get_graph_path(year)
        if not path:
            log(f"Graph for year {year} not found.", tag="LOADER", level=logging.WARNING)
            return None
            
        log(f"Loading graph: {path.name}", tag="LOADER", level=logging.DEBUG)
        try:
            with open(path, "rb") as f:
                graph = pickle.load(f)
            return graph
        except Exception as e:
            log(f"Error loading {path}: {e}", tag="LOADER", level=logging.ERROR)
            return None


    # def load_graphs(self) -> Dict[int, nx.Graph]:
    #     log(f"Loading graphs", tag="GRAPHTEARMANAGER", level=logging.DEBUG)
    #     graphs = {}
    #     parent_path = os.path.join(self.folder, self.subfolder)
    #     for name in os.listdir(parent_path):
    #         if not name.endswith(".gpickle"):
    #             continue
    #         year, postfix = name.split("_")[-2:]
    #         if postfix.split(".")[0] != self.postfix:
    #             continue
    #         if int(year) not in self.year_set:
    #             continue
    #         filepath = os.path.join(parent_path, name)
    #         with open(filepath, "rb") as f:
    #             graphs[year] = pickle.load(f)

    #     if not graphs:
    #         raise ValueError("No graphs loaded: check folder / postfix args")
    #     log(f"Loaded {len(graphs)} graphs", tag="GRAPHTEARMANAGER", level=logging.DEBUG)
    #     return graphs


class Walker:
    def __init__(self, username: str, anime_history: List[Tuple[AnimeId, int, DateString]], threshold_seconds: int = 3600):
        log(f"{username} Walker is initializing", tag="WALKER", level=logging.DEBUG)
        self.username = username
        self.anime_history = anime_history
        self.threshold_seconds = threshold_seconds

        self.trail: Trail = self._build_trail()
        log(f"{self.username} Trail builded", tag="WALKER", level=logging.DEBUG)
        self._calculate_metrics()
        log(f"{self.username} Cheating factor calculated", tag="WALKER", level=logging.DEBUG)

        self.year_steps = [timestamp.year for _, timestamp in self.trail]
        self.year_set = set(self.year_steps)
        self.nodes_sequence = [anime_id for anime_id, _ in self.trail]

        self.start_node = self.trail[0][0]
        self.start_year = self.trail[0][1].year
        log(f"{self.username} Walker initialized. Start year: {self.start_year}. Start node: {self.start_node}", tag="WALKER", level=logging.DEBUG)

    def _build_trail(self) -> Trail:
        parsed = [(rec[0], datetime.strptime(rec[2], "%Y-%m-%d %H:%M:%S.%f")) for rec in self.anime_history]
        return sorted(parsed, key=lambda tup: tup[1])

    def _calculate_metrics(self):
        if len(self.trail) < 2:
            self.deltas = np.array([], dtype=float)
            self.cheating_factor = 0.0
            return
        
        times = [t for _, t in self.trail]
        self.deltas = [t2 - t1 for t1, t2 in zip(times, times[1:])]

        delta_seconds = [d.total_seconds() for d in self.deltas]
        self.cheating_factor = sum(ds < self.threshold_seconds for ds in delta_seconds) / len(delta_seconds)

    def steps_in_year(self, year: int) -> int:
        """Сколько переходов сделал пользователь в этом году."""
        return sum(y == year for y in self.year_steps)

    @property
    def length(self) -> int:
        return len(self.trail)

    def summary(self) -> dict:
        deltas = np.array(self.deltas)
        return {
            "username": self.username,
            "n": self.length,
            "cheating_factor": self.cheating_factor,
            "median_delta": np.median(deltas) if deltas.size else None,
            "perc90_delta": np.percentile(deltas, 90) if deltas.size else None,
        }
    

class RandomWalker:
    _counter = 0

    def __init__(self,
                 parent_walker: str,
                 current_node: Node,
                 settings: RandomWalkerSettings):
        self.parent_walker = parent_walker
        RandomWalker._counter += 1
        self.id = f"{self.parent_walker}_{RandomWalker._counter:03d}"
        self.current_node = current_node
        self.settings = settings
        self.path: List[Node] = []
        self.visited: Set[Node] = {current_node}
        self.stopped = False
        if settings.seed is not None:
            np.random.seed(settings.seed)
        log(f"{self.id} Random Walker initialized.", tag="RANDOMWALKER", level=logging.DEBUG)

    def walk_through_year(self, year: int, graph: nx.Graph, n_steps: int):
        log(f"{self.id} Random Walker is running through year {year}.", tag="RANDOMWALKER", level=logging.DEBUG)
        for _ in range(n_steps):
            if self.stopped:
                break
            next_node = self._choose_next(graph)
            if next_node is None:
                self.stopped = True
                break
            self.path.append(next_node)
            self.visited.add(next_node)
            self.current_node = next_node
        log(f"{self.id} Random Walker finished year {year}.", tag="RANDOMWALKER", level=logging.DEBUG)

    def _neighbors_sorted_by_weight(self, graph: nx.Graph) -> List[Tuple[Node, float]]:
        """
        Возвращает список (neighbor, weight) отсортированных по весу убыванию.
        Ожидается, что вес ребра хранится в атрибуте 'weight' (если нет — используем 1.0).
        """
        if self.current_node not in graph:
            return []

        res = []
        for nbr in graph[self.current_node]:
            weight = graph[self.current_node][nbr].get('weight', 1.0)
            res.append((nbr, float(weight)))

        res.sort(key=lambda x: x[1], reverse=True)

        if self.settings.top_k:
            return res[:self.settings.top_k]
        return res
    
    def _choose_next_greedy(self, graph: nx.Graph) -> Optional[Node]:
        for nbr, _ in self._neighbors_sorted_by_weight(graph):
            if nbr not in self.visited:
                return nbr
        return None
    
    def _choose_next_prob(self, graph: nx.Graph) -> Node | None:
        neigh = self._neighbors_sorted_by_weight(graph)
        if not neigh:
            return None
        nodes = [n for n, _ in neigh]
        weights = np.array([w for _, w in neigh], dtype=float)
        # применим степенной фильтрации, чтобы p1 >> p2 >> p3
        weights = np.power(weights, self.settings.prob_decay)
        probs = weights / weights.sum()
        # выберем случайно среди них, игнорируя уже посещённых (если все посещены -> None)
        candidates = [(n, p) for n, p in zip(nodes, probs) if n not in self.visited]
        if not candidates:
            return None
        nodes_c, probs_c = zip(*candidates)
        probs_c = np.array(probs_c)
        probs_c = probs_c / probs_c.sum()
        return np.random.choice(nodes_c, p=probs_c)
    
    def _choose_next(self, graph) -> Node | None:
        if self.settings.strategy == "greedy":
            return self._choose_next_greedy(graph)
        elif self.settings.strategy == "probabilistic":
            return self._choose_next_prob(graph)
        else:
            raise ValueError("Unknown strategy")


class Measure:
    @staticmethod
    def simple_overlap(user_nodes: Iterable[Node], walker_nodes: Iterable[Node]) -> int:
        su = set(user_nodes)
        sw = set(walker_nodes)
        return len(su & sw)

    @staticmethod
    def normalized_overlap(user_nodes: Iterable[Node], walker_nodes: Iterable[Node]) -> float:
        # нормируем на длину user trail
        su = set(user_nodes)
        sw = set(walker_nodes)
        if not user_nodes:
            return 0.0
        return len(su & sw) / float(len(su))

    @staticmethod
    def weighted_overlap(user_nodes: List[Node],
                         walker_nodes: List[Node],
                         scale: float = 5.0  # регулирует, насколько быстро падает вклад при смещении позиций. Больше scale — мягче штраф за смещение.
                             ) -> float:
        """
        Weighted Overlap (WO).
        scale — параметр, который регулирует, насколько быстро падает вклад при смещении позиций.
            вклад = 1 / (1 + abs(delta)/scale)
        Чем больше scale — тем мягче штраф за смещение.
        Возвращаем нормированное значение (делим на длину user_nodes).
        """
        if not user_nodes:
            return 0.0
        pos_u = {n: i for i, n in enumerate(user_nodes)}
        pos_w = {n: i for i, n in enumerate(walker_nodes)}
        common = set(pos_u.keys()) & set(pos_w.keys())
        if not common:
            return 0.0
        s = 0.0
        for a in common:
            d = abs(pos_u[a] - pos_w[a])
            s += 1.0 / (1.0 + (d / scale))
        # нормировка: делим на длину user trail, чтобы результат в [0,1] примерно
        return s / float(len(user_nodes))


class RandomCrowd:
    def __init__(self,
                 users: List[Walker],
                 data_folder: str | Path = Path("/home/yaroslav/FCUL/MARS_1.0/data/graphs"),
                 n_walkers_per_user: int = 10,
                 settings: RandomWalkerSettings = RandomWalkerSettings()):
        self.users = users
        self.n_walkers_per_user = n_walkers_per_user
        self.settings = settings
        self.walkers = {}
        self.metrics = {}
        self.loader = GraphLoader(base_folder=Path(data_folder))
        log(f"Random Crowd initialized", tag="CROWD", level=logging.INFO)

    def run(self):
        all_years = sorted({y for u in self.users for y in u.year_set})

        self.walkers = {
            u: [RandomWalker(parent_walker=u.username, current_node=u.start_node, settings=self.settings) for _ in range(self.n_walkers_per_user)]
                for u in self.users
                }
        log(f"Created {self.n_walkers_per_user} walkers per user for {len(self.users)} users", tag="CROWD")

        for year in all_years:
            log(f"Processing year {year}...", tag="CROWD")
            graph = self.loader.load_one_graph(year)
            if graph is None:
                continue

            for u in self.users:
                n_steps = u.steps_in_year(year)
                if n_steps == 0:
                    continue

                for rw in self.walkers[u]:
                    rw.walk_through_year(year, graph, n_steps)

            log(f"Year {year} finished.", tag="CROWD")
            del graph
            gc.collect()
            log(f"Memory cleared for year {year}", tag="CROWD", level=logging.DEBUG)

        log(f"All walks are stopped.", tag="CROWD")        

    def evaluate(self,
                metric: str = "weighted",  # 'weighted' | 'normalized' | 'simple'
                scale: float = 5.0
                ) -> Dict[str, dict]:
        """
        Evaluate similarity between each user and its ensemble of walkers.
        Only averages individual comparisons; no consensus walker is used.

        Returns:
            username -> {
                'mean': float,
                'std': float,
                'raw': list[float],
                'cheating_factor': float
            }
        """

        self.metrics = {}

        for u, walkers in self.walkers.items():
            user_nodes = u.nodes_sequence

            vals = []
            for rw in walkers:
                if metric == "simple":
                    v = Measure.simple_overlap(user_nodes, rw.path)
                elif metric == "normalized":
                    v = Measure.normalized_overlap(user_nodes, rw.path)
                elif metric == "weighted":
                    v = Measure.weighted_overlap(user_nodes, rw.path, scale=scale)
                else:
                    raise ValueError(f"Unknown metric: {metric}")

                vals.append(float(v))

            arr = np.array(vals, dtype=float)
            mean = float(arr.mean()) if arr.size else 0.0
            std = float(arr.std()) if arr.size else 0.0

            self.metrics[u.username] = {
                "mean": mean,
                "std": std,
                "raw": vals,
                "cheating_factor": float(u.cheating_factor)
            }
