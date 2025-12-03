import pandas as pd
import plotly.graph_objects as go
import hashlib
import ast
import numpy as np
from collections import Counter

class ClusterVisualizer:
    def __init__(self, df_enriched: pd.DataFrame):
        self.df = df_enriched.copy()
        if 'year' not in self.df.columns or 'cluster_id' not in self.df.columns:
            raise ValueError("DataFrame must contain 'year' and 'cluster_id'")
        self.years = sorted(self.df['year'].unique())

    def _string_to_color(self, string_input: str) -> str:
        hash_object = hashlib.md5(str(string_input).encode())
        return '#' + hash_object.hexdigest()[:6]

    def _safe_parse_iterable(self, val):
        """Пытается превратить строку обратно в список/сет."""
        if isinstance(val, (set, list, tuple)):
            return val
        if isinstance(val, str) and (val.startswith('{') or val.startswith('[')):
            try:
                return ast.literal_eval(val)
            except:
                pass
        return val

    def _generate_node_summary(self, 
                               year, c_id, group_df, 
                               name_col, feature_cols, metric_col, sort_col,
                               age_col=None):
        
        size = len(group_df)
        
        # --- 1. СРЕДНЯЯ МЕТРИКА (Score / Stats) ---
        avg_val_str = ""
        if metric_col and metric_col in group_df.columns:
            vals = pd.to_numeric(group_df[metric_col], errors='coerce').dropna()
            if not vals.empty:
                val = vals.mean()
                avg_val_str = f"Avg {metric_col}: {val:.2f}<br>"

        # --- 2. ВОЗРАСТ (FIXED) ---
        age_str = ""
        if age_col and age_col in group_df.columns:
            # Если колонка числовая (например, year_start: 2007)
            if pd.api.types.is_numeric_dtype(group_df[age_col]):
                birth_years = group_df[age_col]
            else:
                # Если колонка дата (например, birth_date: 1990-05-01)
                birth_years = pd.to_datetime(group_df[age_col], errors='coerce').dt.year
            
            # Возраст = Год кластера - Год рождения (выхода)
            ages = year - birth_years
            
            # Фильтр адекватности (от -1 до 100 лет). 
            # -1 допускаем для аниме, которые вышли в будущем году (пре-релиз хайп)
            valid_ages = ages[(ages >= -1) & (ages <= 100)]
            
            if not valid_ages.empty:
                avg_age = valid_ages.mean()
                age_str = f"Avg Age: {avg_age:.1f} y.o.<br>"

        # --- 3. ГЛАВНАЯ ФИЧА (FIXED) ---
        main_feature_str = "N/A"
        
        for col in feature_cols:
            if col not in group_df.columns: continue
            
            # Собираем все значения в плоский список
            all_values = []
            
            # Предварительно обрабатываем столбец, чтобы распаковать строки-сеты
            # Делаем это на копии серии, чтобы не ломать исходный DF
            series_parsed = group_df[col].dropna().apply(self._safe_parse_iterable)
            
            if series_parsed.empty: continue
            
            # Проверяем первый элемент, чтобы понять стратегию
            first_val = series_parsed.iloc[0]

            # СТРАТЕГИЯ А: Это коллекция (set/list)?
            if isinstance(first_val, (set, list, tuple)):
                for val in series_parsed:
                    if val: all_values.extend(list(val))
                
                if all_values:
                    # Находим самый частый тег (например, 'Comedy')
                    top_val, _ = Counter(all_values).most_common(1)[0]
                    
                    # Считаем покрытие: у скольких рядов этот тег есть внутри списка?
                    hits = series_parsed.apply(lambda x: top_val in x if isinstance(x, (set, list, tuple)) else False).sum()
                    perc = int((hits / size) * 100)
                    main_feature_str = f"{col}: {top_val} ({perc}%)"
                    break

            # СТРАТЕГИЯ Б: Это атомарное значение?
            else:
                # Просто мода
                valid_series = series_parsed.replace({'Unknown': np.nan, '': np.nan}).dropna()
                if not valid_series.empty:
                    top_val = valid_series.mode()[0]
                    hits = (valid_series == top_val).sum()
                    perc = int((hits / size) * 100)
                    main_feature_str = f"{col}: {top_val} ({perc}%)"
                    break

        # --- 4. ТОП СПИСОК ---
        if sort_col and sort_col in group_df.columns:
            sorted_df = group_df.sort_values(sort_col, ascending=False)
        else:
            sorted_df = group_df
            
        if name_col in sorted_df.columns:
            top_items = sorted_df[name_col].head(5).tolist()
            top_items_str = "<br>• ".join([str(x) for x in top_items])
        else:
            top_items_str = "N/A"

        # СБОРКА HTML
        tooltip = (
            f"<b>{year} | Cluster {c_id}</b><br>"
            f"Size: {size}<br>"
            f"{avg_val_str}"
            f"{age_str}"
            f"Main: {main_feature_str}<br>"
            f"<hr>"
            f"<b>Top {name_col}:</b><br>• {top_items_str}"
        )
        return tooltip

    def plot_evolution_sankey(self, 
                              filename: str, 
                              key_col: str, 
                              name_col: str = 'id', 
                              feature_cols: list = [], 
                              metric_col: str = None, 
                              sort_col: str = None, 
                              age_col: str = None, 
                              min_link_size: int = 5, 
                              title: str = "Evolution"):
        """
        Строит диаграмму Санки.
        
        Args:
            filename (str): Путь сохранения (.html).
            key_col (str): Колонка-идентификатор для связывания потоков ('anime_id' или 'username').
            name_col (str): Колонка для отображения имен в топе ('title' или 'username').
            feature_cols (list): Список колонок для поиска "Главной фичи" по приоритету. 
                                 Пример: ['genres', 'source'].
            metric_col (str): Колонка для расчета среднего по кластеру (например, 'score').
            sort_col (str): Колонка для сортировки списка топ-5 (например, 'members').
            age_col (str): Колонка для расчета "Возраста" (года) относительно текущего года кластера.
                           Для аниме это 'year_start', для юзеров 'birth_date'.
            min_link_size (int): Минимальная толщина связи для отображения.
            title (str): Заголовок графика.
        """
        
        print(f"Generating Sankey diagram ({title})...")
        node_map = {}
        node_labels, node_colors, custom_data = [], [], []
        counter = 0
        
        # --- NODES ---
        for year in self.years:
            df_year = self.df[self.df['year'] == year]
            clusters = sorted(df_year['cluster_id'].dropna().unique())
            
            for c_id in clusters:
                c_id = int(c_id)
                node_map[(year, c_id)] = counter
                cluster_df = df_year[df_year['cluster_id'] == c_id]
                
                tooltip = self._generate_node_summary(
                    year, c_id, cluster_df,
                    name_col, feature_cols, metric_col, sort_col, age_col
                )
                
                node_labels.append(f"{year}: {c_id}")
                node_colors.append(self._string_to_color(f"cluster_{c_id}")) 
                custom_data.append(tooltip)
                counter += 1
                
        # --- LINKS ---
        sources, targets, values, link_colors = [], [], [], []
        
        for i in range(len(self.years) - 1):
            year_curr, year_next = self.years[i], self.years[i+1]
            df_curr = self.df[self.df['year'] == year_curr][[key_col, 'cluster_id']]
            df_next = self.df[self.df['year'] == year_next][[key_col, 'cluster_id']]
            
            merged = df_curr.dropna().merge(df_next.dropna(), on=key_col, suffixes=('_src', '_tgt'))
            transitions = merged.groupby(['cluster_id_src', 'cluster_id_tgt']).size().reset_index(name='count')
            
            for _, row in transitions.iterrows():
                count = row['count']
                if count < min_link_size: continue
                try:
                    src_idx = node_map[(year_curr, int(row['cluster_id_src']))]
                    tgt_idx = node_map[(year_next, int(row['cluster_id_tgt']))]
                    sources.append(src_idx)
                    targets.append(tgt_idx)
                    values.append(count)
                    link_colors.append("rgba(180, 180, 180, 0.3)") 
                except KeyError: continue

        # --- PLOT ---
        fig = go.Figure(data=[go.Sankey(
            node = dict(
                pad=15, thickness=20, line=dict(color="black", width=0.5),
                label=node_labels, color=node_colors,
                customdata=custom_data, hovertemplate='%{customdata}<extra></extra>'
            ),
            link = dict(source=sources, target=targets, value=values, color=link_colors)
        )])
        fig.update_layout(title_text=title, font_size=12, height=800)
        
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig.write_html(filename)
        print(f"Plot saved to {filename}")