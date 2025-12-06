from __future__ import annotations
from datetime import datetime
from typing import List, Tuple, Optional, Set, Iterable, Dict
from project_cda.anime_graph_builder import AnimeGraphBuilder
from time import perf_counter_ns
import igraph as ig
import numpy as np
import gc
import os
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from collections import Counter
import pickle
import logging
from project_cda.tag_formatter import log
from datetime import datetime
from typing import Optional


Node = int  # или str, в зависимости от id в графе
Trail = List[Tuple[Node, datetime]]  # (node, unix_timestamp_seconds)
AnimeId = int
DateString = str


@dataclass
class RandomWalkerSettings:
    strategy: str = 'greedy'    # 'greedy' | 'probabilistic'
    prob_decay: float = 2.0   # степень убывания вероятности в стратегии probabilistic
    top_k: Optional[int] = None    # если указан, ограничить выбор первыми k соседями по весу 
    seed: Optional[int] = None


@dataclass
class GraphLoader:
    base_folder: Path | str

    def load_one_graph(self, year: int) -> ig.Graph | None:
        """
        Ищет любой .pickle файл, содержащий год в названии.
        """
        # Ищем файл по маске *2015*.pickle
        # Это сработает и для 'base_2015.pickle', и для 'sparse_2015_knn.pickle'
        self.base_folder = Path(self.base_folder)
        candidates = list(self.base_folder.glob(f"*{year}*.pickle"))

        if not candidates:
            log(f"Graph for {year} not found in {self.base_folder}", tag="LOADER", level=logging.WARNING)
            return None
        
        # ЗАЩИТА: Если файлов больше одного - это ошибка эксперимента, надо разбираться руками
        if len(candidates) > 1:
            names = [f.name for f in candidates]
            log(f"AMBIGUITY ERROR: Found {len(candidates)} files for year {year}: {names}", tag="LOADER", level='ERROR')
            raise ValueError(f"Too many graph files for year {year} in {self.base_folder}")

        # Берем первый найденный файл (так как предполагаем, что лишних файлов в папке нет)
        path = candidates[0]
            
        log(f"Loading graph: {path.name}", tag="LOADER", level='DEBUG')
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            log(f"Error loading {path.name}: {e}", tag="LOADER", level=logging.ERROR)
            return None


class Walker:
    def __init__(self, username: str, anime_history: List[Tuple[AnimeId, int, DateString]], threshold_seconds: int = 3600):
        log(f"{username} Walker is initializing", tag="WALKER", level='DEBUG')
        self.username = username
        self.anime_history = anime_history
        self.threshold_seconds = threshold_seconds

        self.trail: Trail = self._build_trail()
        log(f"{self.username} Trail builded", tag="WALKER", level='DEBUG')
        self._calculate_metrics()
        log(f"{self.username} Cheating factor calculated", tag="WALKER", level='DEBUG')

        self.year_steps = [timestamp.year for _, timestamp in self.trail]
        self.year_set = set(self.year_steps)
        self.nodes_sequence = [anime_id for anime_id, _ in self.trail]

        self.start_node = self.trail[0][0]
        self.start_year = self.trail[0][1].year
        log(f"{self.username} Walker initialized. Start year: {self.start_year}. Start node: {self.start_node}", tag="WALKER", level='DEBUG')

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
        self.current_node = current_node # Это реальный ID (например, anime_id)
        self.settings = settings
        self.path: List[Node] = []
        self.visited: Set[Node] = {current_node}
        self.stopped = False
        
        if settings.seed is not None:
            self.rng = np.random.seed(settings.seed + RandomWalker._counter)
        else:
            self.rng = np.random.default_rng()
            
        log(f"{self.id} Random Walker initialized.", tag="RW_INIT", level='DEBUG')

    def _get_vertex_index(self, graph: ig.Graph, node_name: Node) -> int | None:
        """Безопасное получение индекса узла в igraph по имени."""
        try:
            # graph.vs.find ищет по атрибуту name (мы задали его как str(anime_id))
            return graph.vs.find(name=str(node_name)).index
        except ValueError:
            return None

    def _respawn(self, graph: ig.Graph):
        """JUMP: Если узел исчез."""
        new_node = self._get_random_node(graph)
        log(f"{self.id} node {self.current_node} vanished. Respawning at {new_node}", tag="JUMP", level='DEBUG')
        self.current_node = new_node
        self.visited.add(new_node)

    def walk_through_year(self, year: int, graph: ig.Graph, n_steps: int):
        # 1. Проверяем, существует ли текущий узел в этом году
        # В igraph имена хранятся строками, поэтому str()
        curr_idx = self._get_vertex_index(graph, self.current_node)
        
        if curr_idx is None:
            self._respawn(graph)
            # После респавна нужно обновить индекс
            curr_idx = self._get_vertex_index(graph, self.current_node)

        # 2. WALKING
        for _ in range(n_steps):
            if self.stopped:
                break
            
            # Передаем индекс, чтобы не искать его каждый раз
            next_node = self._choose_next(graph, curr_idx)
            
            if next_node is None:
                self.stopped = True
                break
            
            self.path.append(next_node)
            self.visited.add(next_node)
            self.current_node = next_node
            
            # Обновляем индекс для следующего шага
            curr_idx = self._get_vertex_index(graph, next_node)

    def _neighbors_sorted_by_weight(self, graph: ig.Graph, node_idx: int) -> List[Tuple[Node, float]]:
        """
        Возвращает (neighbor_real_id, weight).
        Работает через внутренние индексы igraph для скорости.
        """
        # Получаем ID инцидентных ребер
        edge_ids = graph.incident(node_idx, mode="ALL")
        if not edge_ids:
            return []
            
        # Получаем веса и соседей пачкой (векторизация)
        edges = graph.es[edge_ids]
        weights = edges["weight"] if "weight" in edges.attributes() else [1.0] * len(edges)
        
        # Получаем индексы соседей. 
        # graph.es[i].source / .target могут вернуть нас самих, нужно выбрать другого
        neighbor_indices = []
        for e in edges:
            if e.source == node_idx:
                neighbor_indices.append(e.target)
            else:
                neighbor_indices.append(e.source)

        # Превращаем индексы обратно в Real Names (anime_id)
        # graph.vs[indices]["name"] вернет список имен
        neighbor_names = graph.vs[neighbor_indices]["name"]
        
        # Преобразуем имена обратно в int, если исходные ID были int
        # (AnimeGraphBuilder сохраняет name как str)
        try:
            neighbor_names = [int(n) for n in neighbor_names]
        except ValueError:
            pass # Если это были логины юзеров, оставляем str

        res = list(zip(neighbor_names, weights))
        
        # Сортировка по весу (desc)
        res.sort(key=lambda x: x[1], reverse=True)

        if self.settings.top_k:
            return res[:self.settings.top_k]
        return res
    
    def _choose_next(self, graph: ig.Graph, curr_node_idx: int) -> Node | None:
        if self.settings.strategy == "greedy":
            # Передаем индекс, чтобы методы работали быстрее
            neighbors = self._neighbors_sorted_by_weight(graph, curr_node_idx)
            for nbr, _ in neighbors:
                if nbr not in self.visited:
                    return nbr
            nxt = None
            
        elif self.settings.strategy == "probabilistic":
            neighbors = self._neighbors_sorted_by_weight(graph, curr_node_idx)
            if not neighbors:
                nxt = None
            else:
                nodes = [n for n, _ in neighbors]
                weights = np.array([w for _, w in neighbors], dtype=float)
                
                # Степенной распад вероятности
                weights = np.power(weights, self.settings.prob_decay)
                probs = weights / weights.sum()
                
                # Фильтр посещенных
                candidates = [(n, p) for n, p in zip(nodes, probs) if n not in self.visited]
                
                if not candidates:
                    nxt = None
                else:
                    nodes_c, probs_c = zip(*candidates)
                    probs_c = np.array(probs_c)
                    probs_c = probs_c / probs_c.sum() # Нормировка
                    nxt = self.rng.choice(nodes_c, p=probs_c)
        else:
            raise ValueError("Unknown strategy")

        if nxt is None:
            nxt = self._teleport(graph)
            
        return nxt

    def _teleport(self, graph: ig.Graph) -> Node:
        new_node = self._get_random_node(graph)
        log(f"{self.id} stuck. Teleporting to {new_node}", tag="TELEPORT", level='DEBUG')
        return new_node

    def _get_random_node(self, graph: ig.Graph) -> Node:
        """Выбор случайного узла пропорционально степени (igraph version)."""
        # degree() в igraph возвращает список степеней для узлов 0..N
        degrees = np.array(graph.degree(), dtype=float)
        probs = degrees / degrees.sum()
        
        # Выбираем ИНДЕКС
        random_idx = self.rng.choice(len(degrees), p=probs)
        
        # Возвращаем ИМЯ (Real ID)
        val = graph.vs[random_idx]["name"]
        try:
            return int(val)
        except ValueError:
            return val


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
                 data_folder: str | Path = Path("data/graphs"),
                 n_walkers_per_user: int = 10,
                 settings: RandomWalkerSettings = RandomWalkerSettings()):
        self.users = users
        self.n_walkers_per_user = n_walkers_per_user
        self.settings = settings
        self.walkers = {}
        self.metrics = {}
        self.loader = GraphLoader(base_folder=Path(data_folder))
        log(f"Random Crowd initialized", tag="CROWD", level='INFO')

    def run(self):
        all_years = sorted({y for u in self.users for y in u.year_set})

        self.walkers = {
            u: [RandomWalker(parent_walker=u.username, current_node=u.start_node, settings=self.settings) for _ in range(self.n_walkers_per_user)]
                for u in self.users
                }
        log(f"Created {self.n_walkers_per_user} walkers per user for {len(self.users)} users", tag="CROWD", level='INFO')

        for year in all_years:
            log(f"Processing year {year}...", tag="CROWD", level='INFO')
            graph = self.loader.load_one_graph(year)
            if graph is None:
                continue

            for u in self.users:
                n_steps = u.steps_in_year(year)
                if n_steps == 0:
                    continue

                for rw in self.walkers[u]:
                    rw.walk_through_year(year, graph, n_steps)

            log(f"Year {year} finished.", tag="CROWD", level='INFO')
            del graph
            gc.collect()
            log(f"Memory cleared for year {year}", tag="CROWD", level='DEBUG')

        log(f"All walks are stopped.", tag="CROWD", level='INFO')

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

    def save_results(self, filename: str):
        """
        Сохраняет результаты эксперимента в CSV.
        Автоматически находит длину истории для каждого юзера.
        """
        if not self.metrics:
            log("Cannot save. No metrics found. Please run evaluate() first.", tag="CROWD", level="WARNING")
            return

        rows = []
        # self.metrics - это словарь {username: {mean, std, ...}}
        # self.users - это список объектов Walker. Превратим его в dict для быстрого поиска
        user_map = {u.username: u for u in self.users}

        for username, data in self.metrics.items():
            user_obj = user_map.get(username)
            
            row = {
                "username": username,
                "n_history": user_obj.length if user_obj else 0, # Важная метрика!
                "cheating_factor": data['cheating_factor'],
                "mean_similarity": round(data['mean'], 4),
                "std_similarity": round(data['std'], 4),
                # Можно сохранить сырые значения списком, если захочешь потом строить боксплоты
                # "raw_values": str(data['raw']) 
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        
        # Создаем папку, если нет
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        df.to_csv(filename, index=False)
        log(f"Results saved to {filename} ({len(df)} users)", tag="CROWD", level="INFO")

    def generate_crowd_filename(self, ext: str = ".csv") -> str:
        """
        Генерирует сигнатуру файла отчета на основе настроек волкеров.
        Пример: crowd_w10_greedy_kNone_20231025_1430.csv
        Пример: crowd_w10_prob_d2.0_k50_20231025_1430.csv
        """
        parts = ["crowd"]
        
        # 1. Количество волкеров
        parts.append(f"w{self.n_walkers_per_user}")
        
        # 2. Стратегия
        parts.append(self.settings.strategy)  # 'greedy' или 'probabilistic'
        
        # 3. Декей (важен только для probabilistic, но можно писать всегда или условно)
        if self.settings.strategy == 'probabilistic':
            pr = str(self.settings.prob_decay).replace('.', '_')
            parts.append(f"d{pr}")
        
        # 4. Top K (если есть)
        k_str = f"k{self.settings.top_k}" if self.settings.top_k is not None else "kAll"
        parts.append(k_str)

        # 5. Seed (если есть, полезно для воспроизводимости)
        if self.settings.seed is not None:
            parts.append(f"seed{self.settings.seed}")
        
        return "_".join(parts) + ext
