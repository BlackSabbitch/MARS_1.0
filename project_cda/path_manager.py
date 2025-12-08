import os

class PathManager:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.abbr_map = {
            # edge building methods
            'jaccard': 'jac',
            'raw': 'raw',
            # sparsification methods
            'full': 'full',
            'knn': 'knn',
            'backbone': 'bb',
            # clustering methods
            'leiden_mod': 'ldnMOD',
            'leiden_cpm': 'ldnCPM',
            'infomap': 'inf',
            'walktrap': 'wtp',
            'eigenvector': 'eig',
            'label_propagation': 'lp',
            # parameters abbreviations
            'threshold': 'th',
            'k': 'k',
            'alpha': 'a',
            'resolution': 'res',
            'objective_function': '',
            'clusters': 'c',
            'n_iterations': 'iter',
            'trials': 't',
            'steps': 's',
        }

    def _cfg_to_tag(self, cfg):
        if not cfg or not cfg.get('name'):
            return None
        name = cfg.get('name', '')
        kwargs = cfg.get('kwargs', {})
        short_name = self.abbr_map.get(name, name)

        params_str = ""
        for k, v in sorted(kwargs.items()):
            if k == 'objective_function':
                continue
            # (threshold -> th)
            k_abbr = self.abbr_map.get(k, k)
            if isinstance(v, float):
                # 0.05 -> 005, 1.0 -> 10
                val_str = str(v).replace('.', '')
                # 1.0 -> 10 -> 1
                if val_str.endswith('0') and len(val_str) > 1:
                     val_str = val_str[:-1]
            else:
                val_str = str(v)

            # th005 (threshold=0.05) or k10 (k=10)
            params_str += f"{k_abbr}{val_str}"

        return f"{short_name}{params_str}"

    def get_paths(self, graph_type, build_cfg, sparse_cfg=None, cluster_cfg=None):
        """
        Возвращает словарь с путями на основе конфигов.
        """
        if graph_type not in ['anime', 'users']:
            raise ValueError(f"Unknown graph_type: {graph_type}. Use 'anime' or 'users'.")

        # 1. Генерируем теги
        build_tag = self._cfg_to_tag(build_cfg)
        sparse_tag = self._cfg_to_tag(sparse_cfg)
        cluster_tag = self._cfg_to_tag(cluster_cfg)

        # 2. Формируем путь к ИСХОДНОМУ графу (который будем грузить)
        # Если спарсинга нет, берем raw. Если есть - берем папку sparse.
        
        # Путь к RAW (базовый)
        raw_dir = os.path.join(self.base_dir, 'data', 'graphs', graph_type, 'raw', build_tag)
        
        if sparse_tag:
            # Имя папки sparse = build + sparse
            sparse_folder_name = f"{build_tag}_{sparse_tag}"
            input_graph_dir = os.path.join(self.base_dir, 'data', 'graphs', graph_type, 'sparse', sparse_folder_name)
            experiment_base_name = sparse_folder_name # Для имени эксперимента
        else:
            # Если спарсинга не было, работаем с raw
            input_graph_dir = raw_dir
            experiment_base_name = build_tag

        # 3. Формируем путь к ЭКСПЕРИМЕНТУ (Результаты)
        experiment_dir = None
        if cluster_tag:
            # Имя папки exp = (build + sparse) + cluster
            exp_folder_name = f"{experiment_base_name}_{cluster_tag}"
            experiment_dir = os.path.join(self.base_dir, 'data', 'experiments', graph_type, exp_folder_name)

        return {
            'graph_type': graph_type,
            'raw_dir': raw_dir,             # Куда сохранять, если строим с нуля
            'input_graph_dir': input_graph_dir, # Откуда грузить граф для анализа
            'experiment_dir': experiment_dir,    # Куда класть plots/reports
            'exp_id': os.path.basename(experiment_dir) if experiment_dir else None
        }

    def ensure_dirs(self, paths):
        # """Создает папки experiments/partitions и т.д."""
        if paths['experiment_dir']:
            os.makedirs(paths['experiment_dir'], exist_ok=True)
        #     os.makedirs(os.path.join(paths['experiment_dir'], 'plots'), exist_ok=True)
        #     os.makedirs(os.path.join(paths['experiment_dir'], 'reports'), exist_ok=True)
        if paths['raw_dir']:
            os.makedirs(paths['raw_dir'], exist_ok=True)
        if 'input_graph_dir' in paths and paths['input_graph_dir'] and 'sparse' in paths['input_graph_dir']:
            os.makedirs(paths['input_graph_dir'], exist_ok=True)
