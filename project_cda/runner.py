import leidenalg
import igraph
import gc
import pandas as pd
from project_cda.community_tracker import CommunityTracker
from project_cda.anime_graph_builder import AnimeGraphBuilder
from project_cda.partition_enricher import PartitionEnricher
from project_cda.cluster_evaluation import ClusterEvaluation
from project_cda.cluster_visualizer import ClusterVisualizer
import os


# --- КОНФИГУРАЦИЯ ---
USERS_PATH = "data/users.csv"
USER_DICT_PATH = "data/user_dict.json"
ANIME_META_PATH = "data/anime.csv" # Твой CSV с жанрами "{'Action', ...}"
YEARS = range(2006, 2012) # Выбери нужный диапазон
METHODS = ["jaccard", "raw"]

# 1. Инициализация (Enricher нужен сразу, чтобы подготовить чистую мету)
print("Initializing Enricher & Builder...")
enricher = PartitionEnricher(ANIME_META_PATH)
builder = AnimeGraphBuilder(USERS_PATH, USER_DICT_PATH, ANIME_META_PATH)

# Подготавливаем чистый словарь info для Evaluation (уже с сетами жанров)
anime_info_clean = enricher.meta_df.set_index('anime_id')[['genres', 'source']].to_dict('index')

evaluation_results = []

for method in METHODS:
    print(f"\n\n{'='*20} RUNNING PIPELINE: {method.upper()} {'='*20}")
    
    partition_history = {} # {year: {anime_id: raw_cluster_id}}
    
    for year in YEARS:
        # A. Сбор статистики и весов
        # max_users=None, если хочешь всех. Для теста поставь 10000.
        edges, counts = builder.build_edges(year, max_users=20000, method=method)
        
        # B. Построение графа
        # Raw требует отсечения шума (threshold > 1), Jaccard (threshold > 0.01)
        thresh = 2 if method == "raw" else 0.02
        G = builder.build_graph(edges, node_counts=counts, weight_threshold=thresh)
        
        # C. Прореживание (Backbone)
        G_sparse = builder.sparsify_knn(G, k=20)
        
        # D. Лейден (Leiden)
        if G_sparse.number_of_nodes() > 0:
            h = igraph.Graph.TupleList(G_sparse.edges(), directed=False)
            # Modularity для начала. Можно CPMVertexPartition для более мелких структур.
            partition = leidenalg.find_partition(h, leidenalg.ModularityVertexPartition, n_iterations=-1)
            part_dict = CommunityTracker.get_membership(h, partition)
            partition_history[year] = part_dict
        else:
            print(f"Skipping {year}: Graph is empty.")
        
        # Чистка памяти
        del edges, counts, G, G_sparse
        gc.collect()

    # E. Трекинг сообществ (выравнивание ID)
    print("Tracking communities...")
    aligned_data = CommunityTracker.track_communities(partition_history, threshold=0.1)
    
    # F. Сохранение результатов
    csv_path = f"reports/partitions/{method}_history.csv"
    # Создаем папку reports/partitions если нет
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    CommunityTracker.save_aligned_history_to_csv(aligned_data, csv_path)
    
    # G. Оценка качества (Evaluation)
    print("Evaluating metrics...")
    evaluator = ClusterEvaluation(method, aligned_data, anime_info_clean)
    stats = evaluator.evaluate()
    evaluation_results.append(stats)
    
    # H. Визуализация (Sankey)
    print("Generating Visualization...")
    # Обогащаем (добавляем Title, Genre) для графика
    df_viz = enricher.enrich_partition(csv_path)
    viz = ClusterVisualizer(df_viz)
    
    html_path = f"reports/plots/sankey_{method}.html"
    viz.plot_evolution_sankey(html_path, min_link_size=10, title=f"Evolution ({method})")

# --- ФИНАЛ ---
print("\n\n" + "="*30)
print("FINAL SCOREBOARD")
print("="*30)
df_res = pd.DataFrame(evaluation_results)
print(df_res)
df_res.to_csv("reports/final_comparison.csv", index=False)