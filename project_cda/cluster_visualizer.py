import pandas as pd
import plotly.graph_objects as go
import hashlib


class ClusterVisualizer:
    def __init__(self, df_enriched: pd.DataFrame):
        self.df = df_enriched.copy()
        required = {'year', 'cluster_id', 'anime_id'}
        if not required.issubset(self.df.columns):
            raise ValueError(f"DataFrame должен содержать колонки: {required}")
        self.years = sorted(self.df['year'].unique())

    def _string_to_color(self, string_input: str) -> str:
        hash_object = hashlib.md5(string_input.encode())
        return '#' + hash_object.hexdigest()[:6]

    def _generate_node_summary(self, year, c_id, group_df):
        """
        Генерирует HTML-тултип. 
        Теперь поддерживает и Genres (sets), и Source (strings).
        """
        size = len(group_df)
        avg_score = group_df['score'].mean() if 'score' in group_df.columns else 0
        
        # 1. Топ-5 Аниме
        sort_col = 'anime_id'
        for col in ['members', 'scored_by', 'favorites', 'score']:
            if col in group_df.columns:
                sort_col = col
                break
        
        top_anime = group_df.sort_values(sort_col, ascending=False).head(5)['title'].tolist()
        
        # 2. Main Feature (Genre или Source)
        main_feature_str = "N/A"
        
        # Логика Жанров (Sets)
        if 'genres' in group_df.columns:
            all_tags = []
            for val in group_df['genres']:
                if isinstance(val, (set, list)): # Enricher уже превратил их в сеты
                    all_tags.extend(list(val))
            
            if all_tags:
                top_tag, _ = Counter(all_tags).most_common(1)[0]
                # Считаем покрытие
                count = group_df['genres'].apply(lambda x: top_tag in x if isinstance(x, (set, list)) else False).sum()
                perc = int((count / size) * 100)
                main_feature_str = f"Genre: {top_tag} ({perc}%)"
        
        # Логика Source (Fallback)
        if main_feature_str == "N/A" and 'source' in group_df.columns:
            sources = group_df['source'].replace('Unknown', pd.NA).dropna()
            if not sources.empty:
                top_source = sources.mode()[0]
                count = sources.value_counts().iloc[0]
                perc = int((count / size) * 100)
                main_feature_str = f"Source: {top_source} ({perc}%)"

        tooltip = (
            f"<b>{year} | Cluster {c_id}</b><br>"
            f"Size: {size}<br>"
            f"Avg Score: {avg_score:.2f}<br>"
            f"Main: {main_feature_str}<br>"
            f"<hr>"
            f"<b>Top Content:</b><br>• " + "<br>• ".join([str(t) for t in top_anime])
        )
        return tooltip

    def plot_evolution_sankey(self, filename="anime_evolution.html", min_link_size=5, title="Anime Communities Evolution"):
        print(f"Generating Sankey diagram for {len(self.years)} years...")
        
        node_map = {}
        node_labels, node_colors, custom_data = [], [], []
        counter = 0
        
        # --- 1. Nodes ---
        for year in self.years:
            df_year = self.df[self.df['year'] == year]
            clusters = sorted(df_year['cluster_id'].dropna().unique())
            
            for c_id in clusters:
                c_id = int(c_id)
                node_map[(year, c_id)] = counter
                
                cluster_df = df_year[df_year['cluster_id'] == c_id]
                
                node_labels.append(f"{year}: {c_id}")
                node_colors.append(self._string_to_color(f"cluster_{c_id}")) 
                custom_data.append(self._generate_node_summary(year, c_id, cluster_df))
                
                counter += 1
                
        # --- 2. Links ---
        sources, targets, values, link_colors = [], [], [], []
        
        for i in range(len(self.years) - 1):
            year_curr, year_next = self.years[i], self.years[i+1]
            
            df_curr = self.df[self.df['year'] == year_curr][['anime_id', 'cluster_id']]
            df_next = self.df[self.df['year'] == year_next][['anime_id', 'cluster_id']]
            
            merged = df_curr.dropna().merge(df_next.dropna(), on='anime_id', suffixes=('_src', '_tgt'))
            transitions = merged.groupby(['cluster_id_src', 'cluster_id_tgt']).size().reset_index(name='count')
            
            for _, row in transitions.iterrows():
                count = row['count']
                if count < min_link_size: continue
                    
                src_c, tgt_c = int(row['cluster_id_src']), int(row['cluster_id_tgt'])
                
                try:
                    sources.append(node_map[(year_curr, src_c)])
                    targets.append(node_map[(year_next, tgt_c)])
                    values.append(count)
                    link_colors.append("rgba(180, 180, 180, 0.3)") 
                except KeyError: continue

        # --- 3. Plot ---
        fig = go.Figure(data=[go.Sankey(
            node = dict(
                pad=15, thickness=20, line=dict(color="black", width=0.5),
                label=node_labels, color=node_colors,
                customdata=custom_data, hovertemplate='%{customdata}<extra></extra>'
            ),
            link = dict(source=sources, target=targets, value=values, color=link_colors)
        )])

        fig.update_layout(title_text=title, font_size=12, height=800)
        
        # Создаем папку если надо
        import os
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
             os.makedirs(directory, exist_ok=True)
             
        fig.write_html(filename)
        print(f"Plot saved to {filename}")


"""
# ... (Предыдущий код: Enrichment) ...

# 1. Загрузили и обогатили (или просто взяли, если уже есть title в df)
# enriched_df = enricher.enrich_partition("results/jaccard_history.csv") 

# 2. Визуализация
from ClusterMetrics import ClusterVisualizer # Если сохранил в файл

viz = ClusterVisualizer(enriched_df)

# min_link_size=10 уберет "волоски", оставив только мощные потоки жанров
viz.plot_evolution_sankey("reports/sankey_jaccard_clean.html", min_link_size=10)
"""
