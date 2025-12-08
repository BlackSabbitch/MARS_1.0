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
        self.year_steps = [year for year in self.year_steps if 2005 < year < 2019]
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
        
        self.current_node = current_node # Это реальный ID
        self.settings = settings
        
        # Исправил type hint: там могут быть и int, и str
        self.path: List[Node] = [] 
        
        self.visited_real_ids: Set[Node] = {current_node}
        self.stopped = False
        
        if settings.seed is not None:
            self.rng = np.random.default_rng(settings.seed + RandomWalker._counter)
        else:
            self.rng = np.random.default_rng()
            
        log(f"{self.id} Random Walker initialized.", tag="RW_INIT", level='DEBUG')

    def _respawn(self, graph: ig.Graph):
        """
        JUMP: Возвращает Internal Index случайного узла.
        Выбор случайного ребра эквивалентен выбору узла пропорционально его степени.
        """
        log(f"{self.id} stuck. Respawn.", tag="TELEPORT", level='DEBUG')
        rand_edge = graph.es[self.rng.integers(0, graph.ecount())]
        return rand_edge.source if self.rng.random() < 0.5 else rand_edge.target

    def walk_through_year(self, 
                          graph: ig.Graph, 
                          n_steps: int, 
                          real_to_idx: Dict[Node, int], 
                          idx_to_real: Dict[int, Node]):
        """
        Основной цикл прогулки.
        """
        if self.stopped:
            return

        current_idx = real_to_idx.get(self.current_node)
        
        # Если узла нет в графе этого года — респаун
        if current_idx is None:
            current_idx = self._respawn(graph)
            self.current_node = idx_to_real[current_idx]

        # HOT LOOP
        for _ in range(n_steps):
            edge_ids = graph.incident(current_idx, mode="ALL")
            
            if not edge_ids:
                current_idx = self._respawn(graph)
            else:
                next_idx = self._choose_next(graph, current_idx, edge_ids, idx_to_real)

                if next_idx is None:
                    current_idx = self._respawn(graph)
                else:
                    current_idx = next_idx

            # Сохраняем шаг (Real ID)
            real_node = idx_to_real[current_idx]
            
            self.path.append(real_node) 
            self.visited_real_ids.add(real_node)
            self.current_node = real_node # Обновляем состояние для следующей итерации/года

    def _choose_next(self, 
                     graph: ig.Graph, 
                     curr_idx: int, 
                     edge_ids: List[int],
                     idx_to_real: Dict[int, Node]) -> int | None:
        
        edges = graph.es[edge_ids]
        
        # Проверка наличия весов (на всякий случай)
        if "weight" in graph.es.attribute_names():
            weights = edges["weight"]
        else:
            weights = [1.0] * len(edges)
        
        neighbors = []
        for e in edges:
            neighbors.append(e.target if e.source == curr_idx else e.source)
        
        # --- GREEDY ---
        if self.settings.strategy == 'greedy':
            nw = sorted(zip(neighbors, weights), key=lambda x: x[1], reverse=True)
            
            if self.settings.top_k:
                nw = nw[:self.settings.top_k]
                
            for n_idx, _ in nw:
                if idx_to_real[n_idx] not in self.visited_real_ids:
                    return n_idx
            return None

        # --- PROBABILISTIC ---
        elif self.settings.strategy == 'probabilistic':
            w_arr = np.array(weights, dtype=float)
            n_arr = np.array(neighbors, dtype=int)
            
            valid_mask = [idx_to_real[n] not in self.visited_real_ids for n in n_arr]
            
            if not any(valid_mask):
                return None
                
            n_arr = n_arr[valid_mask]
            w_arr = w_arr[valid_mask]
            
            if self.settings.top_k and len(w_arr) > self.settings.top_k:
                top_indices = np.argpartition(w_arr, -self.settings.top_k)[-self.settings.top_k:]
                n_arr = n_arr[top_indices]
                w_arr = w_arr[top_indices]

            if self.settings.prob_decay != 1.0:
                w_arr = np.power(w_arr, self.settings.prob_decay)
            
            w_sum = w_arr.sum()
            if w_sum == 0:
                return self.rng.choice(n_arr)
                
            probs = w_arr / w_sum
            return self.rng.choice(n_arr, p=probs)

        return None


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
    
    @staticmethod
    def stability_test(users_sample, settings, loader, n_walkers=10):
        """
        Запускает симуляцию дважды и сравнивает результаты (Mean Similarity).
        users_sample: список объектов Walker (лучше взять 20-30 штук)
        """
        print(f"Stability Test (Run 1 vs Run 2): {len(users_sample)} users, walkers per user: {n_walkers})...")

        # --- ПРОГОН 1 ---
        print("Run 1/2...")
        # Важно: создаем новые инстансы RandomCrowd, чтобы не смешивать состояния
        crowd_1 = RandomCrowd(users=users_sample, n_walkers_per_user=n_walkers, settings=settings)
        # Ручной инжект лоадера, чтобы не создавать заново
        crowd_1.loader = loader 
        
        crowd_1.run()
        crowd_1.evaluate(metric="weighted")
        res_1 = {u: data['similarity_mean'] for u, data in crowd_1.metrics.items()}
        
        # Чистим память, хотя на 20 юзерах это не критично
        # del crowd_1
        
        # --- ПРОГОН 2 ---
        print("Run 2/2...")
        crowd_2 = RandomCrowd(users=users_sample, n_walkers_per_user=n_walkers, settings=settings)
        crowd_2.loader = loader
        
        crowd_2.run()
        crowd_2.evaluate(metric="weighted")
        res_2 = {u: data['similarity_mean'] for u, data in crowd_2.metrics.items()}
        # del crowd_2

        # --- СРАВНЕНИЕ ---
        diffs = []
        print("\nResults (Mean Similarity Run1 vs Run2):")
        print(f"{'User':<15} | {'Run 1':<10} | {'Run 2':<10} | {'Diff':<10}")
        print("-" * 55)
        
        for u_obj in users_sample:
            u = u_obj.username
            val1 = res_1.get(u, 0.0)
            val2 = res_2.get(u, 0.0)
            diff = abs(val1 - val2)
            diffs.append(diff)
            
            # Выводим первые 10 для примера
            if len(diffs) <= 10:
                print(f"{u[:15]:<15} | {val1:.4f}     | {val2:.4f}     | {diff:.4f}")

        avg_diff = np.mean(diffs)
        # Корреляция Пирсона (насколько линейно связаны результаты)
        v1 = list(res_1.values())
        v2 = list(res_2.values())
        correlation = np.corrcoef(v1, v2)[0, 1] if len(v1) > 1 else 0.0

        print("-" * 55)
        print(f"Среднее отклонение (MAE): {avg_diff:.5f}")
        print(f"Корреляция между запусками: {correlation:.4f}")
        
        if correlation > 0.9 and avg_diff < 0.01:
            print("[HIGH] High stability")
        elif correlation > 0.7:
            print("[AVERAGE] Acceptable stability.")
        else:
            print("[LOW] Low stability. Need more random walkers.")

        return crowd_1, crowd_2


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

            log(f"Building name/idx mapping for year {year}...", tag="CROWD", level='DEBUG')

            vs_names = graph.vs["name"] 
            
            # Пробуем сконвертировать в int, если имена - это ID аниме
            try:
                # Если имена хранятся как строки '123', '455', конвертируем в int
                real_ids = [int(x) for x in vs_names]
            except ValueError:
                # Если там смешанные данные или логины, оставляем как есть
                real_ids = vs_names

            # 1. Lookup: RealID -> InternalIndex
            real_to_idx = {rid: i for i, rid in enumerate(real_ids)}
            
            # 2. Lookup: InternalIndex -> RealID (просто список)
            idx_to_real = real_ids # Список индексируется напрямую int-ом, это быстрее dict
            log(f"Name/idx mapping for year {year} builded", tag="CROWD", level='DEBUG')            
            # --- END PRE-CALC ---

            active_count = 0
            for u in self.users:
                n_steps = u.steps_in_year(year)
                if n_steps == 0:
                    continue
                
                active_count += 1
                for rw in self.walkers[u]:
                    rw.walk_through_year(
                        graph, 
                        n_steps, 
                        real_to_idx=real_to_idx, 
                        idx_to_real=idx_to_real
                    )

            log(f"Year {year} finished. Active users: {active_count}", tag="CROWD", level='INFO')
            del graph
            del real_to_idx
            del idx_to_real
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
                "length_of_history": u.length,
                "similarity_mean": mean,
                "similarity_std": std,
                "similarity_values": vals,
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
