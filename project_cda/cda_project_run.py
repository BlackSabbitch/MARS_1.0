import sys
import os
import pandas as pd



# Add project root directory to PYTHONPATH
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from project_cda.graph_io import GraphIO
from project_cda.graph_actions_ig import GraphActionsIG
from project_cda.graph_actions_nx import GraphActionsNX
from project_cda.user_analytics import UserAnalytics
from project_cda.user_data import UserData
from project_cda.user_distribution import UserDistribution

def project_cda_run():
    print("Start project_cda_run()")

    # print("=== STEP 1 — Initialize UserData ===")
    # ud = UserData()

    # # ------------------------------------------------------------------
    # # 1. Select 86 initial users (2004-2006 join year + rating criteria)
    # # ------------------------------------------------------------------
    # print("Selecting initial ~86 users...")
    # selected_86 = ud.get_user_data()
    # print("Found:", len(selected_86), "users")

    # # ------------------------------------------------------------------
    # # 2. Save complete dataset for these 86 users
    # # ------------------------------------------------------------------
    # print("Saving full dataset for selected 86 users...")
    # ud.extract_full_user_info(selected_86)

    # # ------------------------------------------------------------------
    # # 3. Load the saved 86-user datasets
    # # ------------------------------------------------------------------
    # print("Loading saved datasets (86 users)...")
    # users_86 = pd.read_csv(os.path.join(ud.cda_data_dir, "selected_users.csv"))
    # lists_86 = pd.read_csv(os.path.join(ud.cda_data_dir, "selected_users_animelist.csv"))

    # # ------------------------------------------------------------------
    # # 4. Clean users (remove unrealistic countries + inactive before 2017)
    # # ------------------------------------------------------------------
    # print("Cleaning users (country filter + last_online >= 2017)...")
    # cleaned = ud.clean_users(users_86)
    # print("Remaining after cleaning:", len(cleaned))

    # # ------------------------------------------------------------------
    # # 5. Select top-20 cleaned users (or fewer if <20 remain)
    # # ------------------------------------------------------------------
    # print("Selecting top-20 cleaned active users...")
    # top20 = ud.select_top20(cleaned, lists_86)
    # print("Top users:", len(top20))

    # # ------------------------------------------------------------------
    # # 6. Save full dataset for these top users
    # # ------------------------------------------------------------------
    # print("Saving top-user dataset...")
    # ud.save_top20_full_info(top20)

    # # ------------------------------------------------------------------
    # # 7. Run analytics on top20
    # # ------------------------------------------------------------------
    # print("Running analytics on cleaned top-20 users...")

    # analytics = UserAnalytics(ud.cda_data_dir)

    # analytics.plot_score_distribution()
    # analytics.plot_activity_by_year()
    # analytics.plot_genres_over_time()
    # analytics.plot_heatmap_activity()
    # analytics.plot_demographics()
    # analytics.plot_score_vs_activity()
    # analytics.plot_user_timelines()
    # analytics.plot_genre_heatmap()


    print("PATH:", r"C:\MariaSamosudova\Projects\UNIVER\ADB\Project\MARS_1.0\data\data_cda\graphs\anime_graph_2006.gexf")
    print("EXISTS:", os.path.exists(r"C:\MariaSamosudova\Projects\UNIVER\ADB\Project\MARS_1.0\data\data_cda\graphs\anime_graph_2006.gexf"))

    # anime_graph_2006 = GraphActionsNX.get_graph(path = r"C:\MariaSamosudova\Projects\UNIVER\ADB\Project\MARS_1.0\data\data_cda\graphs\anime_graph_2006.gexf")
    # anime_graph_2010_47k = GraphActionsNX.get_graph(path = r"C:\MariaSamosudova\Projects\UNIVER\ADB\Project\MARS_1.0\data\data_cda\graphs\anime_graph_2010_47k.gexf")
    # anime_graph_2018 = GraphActionsNX.get_graph(path = r"C:\MariaSamosudova\Projects\UNIVER\ADB\Project\MARS_1.0\data\data_cda\graphs\graph_2018.gpickle")
    # GraphActionsNX.print_graph_metrics(GraphActions.get_metrics_array(anime_graph_2006))

    # GraphActionsNX.print_graph_metrics(
    # anime_graph_2006,
    # GraphActionsNX.get_metrics_dict(anime_graph_2006, graph_name="anime_graph_2006")
    # )
    # anime_graph_2006_path = r"C:\MariaSamosudova\Projects\UNIVER\ADB\Project\MARS_1.0\data\data_cda\graphs\anime_graph_2006.gexf"
   
    # g = GraphActionsIG.load_gexf_as_igraph(anime_graph_2006_path)   
    # metrics = GraphActionsIG.get_full_metrics_dict(g, graph_name="anime_graph_2006")
    # GraphActionsIG.print_graph_metrics(metrics)
    # GraphIO.convert_graph_to_parquet(anime_graph_2006_path)

    # Community detection
    anime_graph_2006_backbone = (
        r"C:\MariaSamosudova\Projects\UNIVER\ADB\Project\MARS_1.0"
        r"\data\data_cda\graphs\Graphs_cleaned_95_percentile_backbone_thr_2"
        r"\anime_graph_2006_backbone.gpickle"
    )

    graphs_dir = r"C:\MariaSamosudova\Projects\UNIVER\ADB\Project\MARS_1.0\data\data_cda\graphs\Graphs_cleaned_95_percentile_backbone_thr_2\communities"

    # Run community detection for all graphs in the folder
    # GraphActionsIG.run_community_detection_for_all(graphs_dir)
    # for fname in os.listdir(graphs_dir):
    #     if fname.endswith(".pkl") or fname.endswith(".gpickle"):
    #         full_path = os.path.join(graphs_dir, fname)
    #         GraphIO.convert_pickle_to_parquet(full_path)

    # print(">>> ENTERED project_cda_run() <<<")

    graphs_dir_communities = r"C:\MariaSamosudova\Projects\UNIVER\ADB\Project\MARS_1.0\data\data_cda\graphs\Graphs_cleaned_95_percentile_backbone_thr_2\communities"

    # 1. Run community detection for all graphs (if needed)
    # run_community_detection_for_all(graphs_dir)

    # 2. Count communities detected by Leiden and Infomap
    GraphActionsIG.count_communities_in_folder(graphs_dir_communities)

if __name__ == "__main__":
            
    project_cda_run()
