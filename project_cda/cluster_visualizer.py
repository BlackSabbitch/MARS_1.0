import pandas as pd
import os
import plotly.graph_objects as go
import hashlib
import ast
import numpy as np
import plotly.express as px
from collections import Counter
from project_cda.tag_formatter import log


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

    def _get_top_feature(self, series, is_iterable=False):
            """Возвращает (TopVal, Percentage) для серии данных."""
            all_vals = []
            total = len(series)
            if total == 0: return "N/A", 0
            
            valid_series = series.dropna()
            if valid_series.empty: return "N/A", 0

            if is_iterable:
                for x in valid_series:
                    parsed = self._safe_parse_iterable(x)
                    if parsed: all_vals.extend(list(parsed))
            else:
                all_vals = valid_series[valid_series != 'Unknown'].tolist()
                
            if not all_vals: return "N/A", 0
            
            top_val, _ = Counter(all_vals).most_common(1)[0]
            
            # Считаем процент покрытия
            if is_iterable:
                hits = valid_series.apply(lambda x: top_val in self._safe_parse_iterable(x) if x else False).sum()
            else:
                hits = (valid_series == top_val).sum()
                
            return top_val, int((hits / total) * 100)

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

        # --- 3. ГЛАВНЫЕ ФИЧИ (Top-5 с процентами) ---
        main_feature_str = "N/A"
            
        for col in feature_cols:
            if col not in group_df.columns: continue
                
            # Распаковка списков [Action, Comedy] в плоский список
            all_values = []
            series_parsed = group_df[col].dropna().apply(self._safe_parse_iterable)
            if series_parsed.empty: continue

            for val in series_parsed:
                if isinstance(val, (set, list, tuple)):
                    all_values.extend(list(val))
                else:
                    all_values.append(val)
                
            if not all_values: continue

            # Считаем
            total_items = len(series_parsed) 
            counts = Counter(all_values)
                
            labels_to_show = []
                
            # Берем ТОП-5 самых частых. 
            # Не прерываем по сумме, чтобы не потерять второй популярный жанр.
            for tag, count in counts.most_common(5):
                perc = (count / total_items) * 100
                    
                # Фильтр: не показывать совсем редкие (<10%), если это не единственный тег
                if perc < 10 and len(labels_to_show) > 0:
                    continue
                    
                # ВОТ ТУТ ПИШЕМ ПРОЦЕНТЫ
                labels_to_show.append(f"{tag} ({int(perc)}%)")
                
            if labels_to_show:
                main_feature_str = f"<b>{col}:</b> " + ", ".join(labels_to_show)
                break

        # --- 4. ТОП СПИСОК (Names) ---
        # Вернул этот блок, он важен для понимания, что внутри
        if sort_col and sort_col in group_df.columns:
            sorted_df = group_df.sort_values(sort_col, ascending=False)
        else:
            sorted_df = group_df
            
        if name_col in sorted_df.columns:
            top_items = sorted_df[name_col].head(5).astype(str).tolist()
            # Обрезаем длинные названия
            top_items = [t[:30] + "..." if len(t) > 30 else t for t in top_items]
            top_items_str = "<br>• ".join(top_items)
        else:
            top_items_str = "N/A"

        # --- СБОРКА И ВОЗВРАТ (Этого не хватало) ---
        tooltip = (
            f"<b>{year} | Cluster {c_id}</b><br>"
            f"Size: {size}<br>"
            f"{avg_val_str}"
            f"{age_str}"
            f"{main_feature_str}<br>"
            f"<hr>"
            f"<b>Top {name_col}:</b><br>• {top_items_str}"
        )
        return tooltip


    # --- 1. SANKEY DIAGRAM ---
    def plot_sankey(self, 
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

        log(f"Generating Sankey diagram ({title})...", tag="PLOT", level="DEBUG")
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
                    # Получаем цвет узла-источника (Hex)
                    src_hex = node_colors[src_idx] 
                    # Превращаем Hex в RGB
                    r = int(src_hex[1:3], 16)
                    g = int(src_hex[3:5], 16)
                    b = int(src_hex[5:7], 16)
                    # Добавляем прозрачность (0.4 выглядит приятно)
                    link_colors.append(f"rgba({r}, {g}, {b}, 0.4)")
                    # link_colors.append("rgba(180, 180, 180, 0.3)") 
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

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig.write_html(filename)
        log(f"Plot saved to {filename}", tag="PLOT", level="INFO")


# --- 2. STREAMGRAPH (Stacked Area) ---
    def plot_streamgraph(self, filename, feature_col, title="Streamgraph: Cluster Sizes"):
        """
        Строит график изменения размеров кластеров.
        Легенда иерархическая: Группировка по Жанру -> Список Кластеров.
        """
        log(f"Generating Streamgraph ({title})...", tag="PLOT", level="DEBUG")
        
        # 1. Агрегация данных
        stats = []
        for year in self.years:
            df_year = self.df[self.df['year'] == year]
            for c_id in df_year['cluster_id'].unique():
                c_df = df_year[df_year['cluster_id'] == c_id]
                size = len(c_df)
                
                # Ищем главное свойство
                feat_val, _ = self._get_top_feature(c_df[feature_col], is_iterable=True)
                if feat_val == "N/A": 
                    feat_val, _ = self._get_top_feature(c_df[feature_col], is_iterable=False)

                # ХИТРОСТЬ 1: Создаем комбинированный ключ для цвета.
                # Используем редкий разделитель '§§', чтобы потом легко разрезать.
                # Формат: "ЖАНР §§ КЛАСТЕР"
                combined_label = f"{feat_val}§§Cluster {int(c_id)}"
                
                stats.append({
                    'year': year, 
                    'count': size, 
                    'combined_label': combined_label, # Это пойдет в color
                    'cluster_id': c_id,
                    'feature': feat_val # Для сортировки
                })
        
        df_stats = pd.DataFrame(stats)
        
        # ХИТРОСТЬ 2: Сортируем данные. 
        # Это критично для Streamgraph, чтобы слои одного жанра лежали рядом.
        df_stats = df_stats.sort_values(by=['feature', 'cluster_id', 'year'])
        
        # 2. Plot
        fig = px.area(
            df_stats, 
            x="year", y="count", 
            color="combined_label",  # Красим по уникальной связке (Жанр+Кластер)
            line_group="cluster_id", # Чтобы линии не прерывались логически
            title=title,
            labels={"count": "Size (Nodes)"}
        )
        
        # ХИТРОСТЬ 3: Переделываем легенду в иерархическую
        # Проходим по всем созданным трейсам и обновляем их свойства
        fig.for_each_trace(lambda trace: trace.update(
            name=trace.name.split('§§')[1],            # Имя в легенде: "Cluster 1"
            legendgroup=trace.name.split('§§')[0],     # Группа логическая: "Action"
            legendgrouptitle_text=trace.name.split('§§')[0] # Заголовок группы: "Action"
        ) if '§§' in trace.name else None)

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig.write_html(filename)
        log(f"Plot saved to {filename}", tag="PLOT", level="INFO")


    # --- 3. ANIMATED BUBBLE CHART ---
    def plot_bubbles(self, filename, x_col, y_col, size_col='count', title="Dynamic Landscape"):
        """
        Анимированный Scatter plot.
        Args:
            x_col: Колонка для оси X (среднее по кластеру).
            y_col: Колонка для оси Y (среднее по кластеру).
            size_col: 'count' (размер кластера) или имя колонки для усреднения.
        """
        log(f"Generating Bubble Chart ({title})...", tag="PLOT", level="DEBUG")
        
        # Агрегация
        agg_rules = {
            x_col: 'mean',
            y_col: 'mean',
            'anime_id' if 'anime_id' in self.df.columns else 'username': 'count' # ID count
        }
        
        # Если size_col не count, добавляем его в аггрегацию
        if size_col != 'count':
            agg_rules[size_col] = 'mean'
            
        df_grouped = self.df.groupby(['year', 'cluster_id']).agg(agg_rules).reset_index()
        
        # Переименовываем count
        count_col_name = 'anime_id' if 'anime_id' in self.df.columns else 'username'
        df_grouped = df_grouped.rename(columns={count_col_name: 'node_count'})
        
        real_size_col = 'node_count' if size_col == 'count' else size_col
        
        # Plot
        fig = px.scatter(
            df_grouped,
            x=x_col,
            y=y_col,
            animation_frame="year",
            animation_group="cluster_id",
            size=real_size_col,
            color="cluster_id",
            hover_name="cluster_id",
            size_max=60,
            range_x=[df_grouped[x_col].min()*0.9, df_grouped[x_col].max()*1.1],
            range_y=[df_grouped[y_col].min()*0.9, df_grouped[y_col].max()*1.1],
            title=title
        )

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig.write_html(filename)
        log(f"Plot saved to {filename}", tag="PLOT", level="INFO")


    # --- 4. SUNBURST ---
    def plot_sunburst(self, filename, feature_col, size_col='count', title="Hierarchy"):
        """
        Иерархия: Год -> Кластер -> Главная Фича.
        """
        log(f"Generating Sunburst ({title})...", tag="PLOT", level="DEBUG")
        
        # Подготовка данных: нам нужно распаковать (explode) фичи, 
        # но чтобы не перегрузить график, берем только TOP-1 фичу для каждого узла.
        
        # Упрощение: Сгруппируем данные и найдем топ-фичу для каждого узла
        # (Иначе Sunburst на 6000 узлах будет висеть)
        
        # 1. Группировка
        # Для каждого аниме/юзера берем его главную фичу
        temp_df = self.df.copy()
        
        # Функция получения первого жанра/страны
        def get_first(val):
            parsed = self._safe_parse_iterable(val)
            if isinstance(parsed, (set, list, tuple)):
                return list(parsed)[0] if parsed else "Unknown"
            return parsed if parsed else "Unknown"
            
        temp_df['primary_feature'] = temp_df[feature_col].apply(get_first)
        
        # 2. Plot
        fig = px.sunburst(
            temp_df,
            path=['year', 'cluster_id', 'primary_feature'],
            title=title,
            height=800
        )

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig.write_html(filename)
        log(f"Plot saved to {filename}", tag="PLOT", level="INFO")
