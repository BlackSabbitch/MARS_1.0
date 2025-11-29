import sys
import os
import pandas as pd

# Add project root directory to PYTHONPATH
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


from cda_project.user_analytics import UserAnalytics
from cda_project.user_data import UserData
from cda_project.user_distribution import UserDistribution

def cda_project_run():
    print(">>> ENTERED cda_project_run() <<<")
    #=== STARTING CDA PROJECT RUN ===

    print("=== STEP 1 — Initialize UserData ===")
    ud = UserData()

    # ------------------------------------------------------------------
    # 1. Select 86 initial users (2004-2006 join year + rating criteria)
    # ------------------------------------------------------------------
    print("Selecting initial ~86 users...")
    selected_86 = ud.get_user_data()
    print("Found:", len(selected_86), "users")

    # ------------------------------------------------------------------
    # 2. Save complete dataset for these 86 users
    # ------------------------------------------------------------------
    print("Saving full dataset for selected 86 users...")
    ud.extract_full_user_info(selected_86)

    # ------------------------------------------------------------------
    # 3. Load the saved 86-user datasets
    # ------------------------------------------------------------------
    print("Loading saved datasets (86 users)...")
    users_86 = pd.read_csv(os.path.join(ud.cda_data_dir, "selected_users.csv"))
    lists_86 = pd.read_csv(os.path.join(ud.cda_data_dir, "selected_users_animelist.csv"))

    # ------------------------------------------------------------------
    # 4. Clean users (remove unrealistic countries + inactive before 2017)
    # ------------------------------------------------------------------
    print("Cleaning users (country filter + last_online >= 2017)...")
    cleaned = ud.clean_users(users_86)
    print("Remaining after cleaning:", len(cleaned))

    # ------------------------------------------------------------------
    # 5. Select top-20 cleaned users (or fewer if <20 remain)
    # ------------------------------------------------------------------
    print("Selecting top-20 cleaned active users...")
    top20 = ud.select_top20(cleaned, lists_86)
    print("Top users:", len(top20))

    # ------------------------------------------------------------------
    # 6. Save full dataset for these top users
    # ------------------------------------------------------------------
    print("Saving top-user dataset...")
    ud.save_top20_full_info(top20)

    # ------------------------------------------------------------------
    # 7. Run analytics on top20
    # ------------------------------------------------------------------
    print("Running analytics on cleaned top-20 users...")

    analytics = UserAnalytics(ud.cda_data_dir)

    analytics.plot_score_distribution()
    analytics.plot_activity_by_year()
    analytics.plot_genres_over_time()
    analytics.plot_heatmap_activity()
    analytics.plot_demographics()
    analytics.plot_score_vs_activity()
    analytics.plot_user_timelines()
    analytics.plot_genre_heatmap()

    print("=== DONE ===")

if __name__ == "__main__":
            
    cda_project_run()
