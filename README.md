# MARS_1.0
Manga and Anime Research Service (study project)

---

Project for analyzing and working with various datasets (Anime, Cities, etc.).  
Python scripts **expect the datasets to be already downloaded** and placed under the `data/` folder.

---

## Project Structure

MARS_1.0/
├── data/ # All datasets
│ ├── anime_ranks/
│ ├── cities_location/
│ ├── cities_population/
│ └── anime_timestamps/
├── datasets.yaml # Dataset metadata
├── common_tools.py
├── graph_tools.py
├── playground.ipynb
└── README.md


# Setup
pip install -r requirements.txt
kaggle.json → ~/.kaggle/
make data