import pandas as pd
import plotly.graph_objects as go
import hashlib
import ast
import numpy as np
import plotly.express as px
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

# ==========================================================================================================================

        # # --- 3. ГЛАВНАЯ ФИЧА (FIXED) ---
        # main_feature_str = "N/A"
        
        # for col in feature_cols:
        #     if col not in group_df.columns: continue
            
        #     # Собираем все значения в плоский список
        #     all_values = []
            
        #     # Предварительно обрабатываем столбец, чтобы распаковать строки-сеты
        #     # Делаем это на копии серии, чтобы не ломать исходный DF
        #     series_parsed = group_df[col].dropna().apply(self._safe_parse_iterable)
            
        #     if series_parsed.empty: continue
            
        #     # Проверяем первый элемент, чтобы понять стратегию
        #     first_val = series_parsed.iloc[0]

        #     # СТРАТЕГИЯ А: Это коллекция (set/list)?
        #     if isinstance(first_val, (set, list, tuple)):
        #         for val in series_parsed:
        #             if val: all_values.extend(list(val))
                
        #         if all_values:
        #             # Находим самый частый тег (например, 'Comedy')
        #             top_val, _ = Counter(all_values).most_common(1)[0]
                    
        #             # Считаем покрытие: у скольких рядов этот тег есть внутри списка?
        #             hits = series_parsed.apply(lambda x: top_val in x if isinstance(x, (set, list, tuple)) else False).sum()
        #             perc = int((hits / size) * 100)
        #             main_feature_str = f"{col}: {top_val} ({perc}%)"
        #             break

        #     # СТРАТЕГИЯ Б: Это атомарное значение?
        #     else:
        #         # Просто мода
        #         valid_series = series_parsed.replace({'Unknown': np.nan, '': np.nan}).dropna()
        #         if not valid_series.empty:
        #             top_val = valid_series.mode()[0]
        #             hits = (valid_series == top_val).sum()
        #             perc = int((hits / size) * 100)
        #             main_feature_str = f"{col}: {top_val} ({perc}%)"
        #             break

# ==========================================================================================================================

        # # --- 3. ГЛАВНЫЕ ФИЧИ (Logic: Accumulate up to 80%) ---
        # main_feature_str = "N/A"
        
        # for col in feature_cols:
        #     if col not in group_df.columns: continue
            
        #     # Распаковка всех значений
        #     all_values = []
        #     series_parsed = group_df[col].dropna().apply(self._safe_parse_iterable)
            
        #     if series_parsed.empty: continue
            
        #     # Собираем общий мешок слов
        #     for val in series_parsed:
        #         if isinstance(val, (set, list, tuple)):
        #             all_values.extend(list(val))
        #         else:
        #             all_values.append(val)
            
        #     if not all_values: continue

        #     # Считаем частоты
        #     total_items = len(series_parsed) # Знаменатель - кол-во узлов (аниме), а не всего тегов
        #     counts = Counter(all_values)
            
        #     # Сортируем: от самых частых
        #     sorted_counts = counts.most_common()
            
        #     accumulated_perc = 0
        #     labels_to_show = []
            
        #     for tag, count in sorted_counts:
        #         # Процент покрытия: у скольких аниме встречается этот тег
        #         # (Для точного расчета нужно проходить по series_parsed, но для скорости берем упрощенно count/total_items, 
        #         # если теги уникальны внутри аниме. Если один аниме имеет 2 'Action' (баг данных) - будет погрешность, но не страшно)
                
        #         # Более точный подсчет покрытия (сколько рядов содержит тег):
        #         hits = series_parsed.apply(lambda x: tag in x if isinstance(x, (set, list, tuple)) else tag == x).sum()
        #         perc = (hits / total_items) * 100
                
        #         labels_to_show.append(f"{tag} ({int(perc)}%)")
        #         accumulated_perc += perc
                
        #         # Если набрали 80% "веса" (сумма процентов может быть > 100% для мульти-тегов, 
        #         # но логика "показать самые жирные" сохраняется). 
        #         # Давай ограничимся топ-3 или пока не наберем покрытие.
        #         if accumulated_perc >= 80 or len(labels_to_show) >= 4:
        #             break
            
        #     main_feature_str = f"<b>{col}:</b> " + ", ".join(labels_to_show)
        #     break # Берем первую найденную колонку из feature_cols

        # # --- 4. ТОП СПИСОК ---
        # if sort_col and sort_col in group_df.columns:
        #     sorted_df = group_df.sort_values(sort_col, ascending=False)
        # else:
        #     sorted_df = group_df
            
        # if name_col in sorted_df.columns:
        #     top_items = sorted_df[name_col].head(5).tolist()
        #     top_items_str = "<br>• ".join([str(x) for x in top_items])
        # else:
        #     top_items_str = "N/A"

        # # СБОРКА HTML
        # tooltip = (
        #     f"<b>{year} | Cluster {c_id}</b><br>"
        #     f"Size: {size}<br>"
        #     f"{avg_val_str}"
        #     f"{age_str}"
        #     f"Main: {main_feature_str}<br>"
        #     f"<hr>"
        #     f"<b>Top {name_col}:</b><br>• {top_items_str}"
        # )
        # return tooltip

# ==========================================================================================================================

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

# --- 2. STREAMGRAPH (Stacked Area) ---
    def plot_streamgraph(self, filename, feature_col, title="Streamgraph: Cluster Sizes"):
        """
        Строит график изменения размеров кластеров.
        Легенда подписывается как 'Cluster X (MainFeature)'.
        """
        print(f"Generating Streamgraph ({title})...")
        
        # 1. Агрегация данных
        stats = []
        for year in self.years:
            df_year = self.df[self.df['year'] == year]
            for c_id in df_year['cluster_id'].unique():
                c_df = df_year[df_year['cluster_id'] == c_id]
                size = len(c_df)
                
                # Ищем главное свойство для легенды
                feat_val, _ = self._get_top_feature(c_df[feature_col], is_iterable=True) # Пытаемся как iterable, safety check внутри есть
                if feat_val == "N/A": 
                    # Если не вышло как iterable (жанры), пробуем как строку (страна)
                    feat_val, _ = self._get_top_feature(c_df[feature_col], is_iterable=False)

                label = f"Cl {int(c_id)}: {feat_val}"
                stats.append({'year': year, 'count': size, 'label': label, 'cluster_id': c_id})
        
        df_stats = pd.DataFrame(stats)
        
        # 2. Plot
        fig = px.area(
            df_stats, 
            x="year", y="count", 
            color="label", 
            line_group="cluster_id",
            title=title,
            labels={"count": "Size (Nodes)"}
        )
        
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig.write_html(filename)
        print(f"Saved: {filename}")

    # --- 3. RADAR CHART (Spider) ---
    def plot_radar(self, filename, year, target_clusters, feature_col, title="Cluster DNA Comparison"):
        """
        Сравнивает ТОП-5 фичей (жанров/стран) для выбранных кластеров в виде паутины.
        Args:
            year: Год для анализа.
            target_clusters: Список ID кластеров [0, 5, 2].
            feature_col: Колонка для осей (genres/country).
        """
        print(f"Generating Radar Chart ({title})...")
        df_year = self.df[self.df['year'] == year]
        
        # 1. Собираем Топ фичи для осей (Union всех топов целевых кластеров)
        axis_candidates = Counter()
        cluster_data = {} # {c_id: {feat: count}}
        
        for c_id in target_clusters:
            c_df = df_year[df_year['cluster_id'] == c_id]
            if c_df.empty: continue
            
            # Собираем все теги кластера
            all_tags = []
            for val in c_df[feature_col].dropna():
                parsed = self._safe_parse_iterable(val)
                if isinstance(parsed, (set, list, tuple)): all_tags.extend(list(parsed))
                else: all_tags.append(parsed)
            
            # Считаем частоты
            counts = Counter(all_tags)
            total = len(c_df)
            
            # Нормализуем (процент)
            norm_counts = {k: v/total for k, v in counts.items()}
            cluster_data[c_id] = norm_counts
            
            # Добавляем топ-5 этого кластера в кандидаты на оси
            for k, _ in counts.most_common(5):
                axis_candidates[k] += 1
                
        # Оси радара = Топ-10 самых популярных фичей среди выбранных кластеров
        categories = [k for k, _ in axis_candidates.most_common(10)]
        
        # 2. Plot
        fig = go.Figure()
        
        for c_id in target_clusters:
            if c_id not in cluster_data: continue
            
            # Собираем значения для осей
            values = [cluster_data[c_id].get(cat, 0) for cat in categories]
            # Замыкаем круг
            values += [values[0]]
            categories_closed = categories + [categories[0]]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories_closed,
                fill='toself',
                name=f"Cluster {c_id}"
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title=f"{title} ({year})",
            showlegend=True
        )
        
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig.write_html(filename)
        print(f"Saved: {filename}")

    # --- 4. ANIMATED BUBBLE CHART ---
    def plot_bubbles(self, filename, x_col, y_col, size_col='count', title="Dynamic Landscape"):
        """
        Анимированный Scatter plot.
        Args:
            x_col: Колонка для оси X (среднее по кластеру).
            y_col: Колонка для оси Y (среднее по кластеру).
            size_col: 'count' (размер кластера) или имя колонки для усреднения.
        """
        print(f"Generating Bubble Chart ({title})...")
        
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
        
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig.write_html(filename)
        print(f"Saved: {filename}")

    # --- 5. SUNBURST ---
    def plot_sunburst(self, filename, feature_col, size_col='count', title="Hierarchy"):
        """
        Иерархия: Год -> Кластер -> Главная Фича.
        """
        print(f"Generating Sunburst ({title})...")
        
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
        
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig.write_html(filename)
        print(f"Saved: {filename}")
