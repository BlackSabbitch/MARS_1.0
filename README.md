# MARS_1.0
Manga and Anime Research Service (study project)

---

**MARS_1.0** is a comprehensive Python framework designed for the management, preprocessing, and analysis of large-scale anime viewership data.

This project integrates pipelines for data cleaning, user-interaction modeling, and complex graph-based analysis to derive insights
into community evolution. It features dedicated modules for database management (MongoDB, MySQL), machine learning experiments,
and automated reporting.

> **Note:** External datasets must be manually placed in the `data/` directory prior to execution (see Project Structure below).

---

## Project Structure

```
MARS_1.0/
├── data/                                   # Non-code assets (ignored by git)
│   ├── datasets/                           # Original and processed datasets
│   │   ├── anime_azathoth42/               # [Source] Kaggle: myanimelist (2018)
│   │   ├── anime_hernan4444/               # [Source] Kaggle: anime-recommendations-database (2020)
│   │   ├── cities_population.../           # [Source] Kaggle: world-cities-population-cleaned (2022)
│   │   └── myanimelist_countries.../       # [Self-made] SEMRush traffic distribution (2025)
│   ├── experiments/                        # Experiment outputs: metrics, visualizations, partitions
│   │   ├── users/
│   │   └── anime/
│   │       └── ...                         # e.g., community detection results (html, csv)
│   ├── graphs/                             # Serialized graph objects (Pickle/NetworkX)
│   │   ├── anime/
│   │   │   ├── raw/                        # Initial graph snapshots
│   │   │   └── sparse/                     # Sparsified backbones (e.g., Disparity Filter)
│   │   └── helpers/                        # Temporary intermediate files
├── datasets.json                           # Configuration file for dataset paths and metadata
├── project_cda/                            # Code: Community detection analysis module
├── project_db/                             # Code: Database implementations module
├── project_fds/                            # Code: Preprocessing and common ML module
├── reports_latex/                          # LaTeX source code for generated project reports
├── requirements.txt                        # Python dependencies
├── setup.py                                # Package installer script
└── README.md
```

## Setup

### 1. Clone
```
...$ git clone git@github.com:BlackSabbitch/MARS_1.0.git
...$ cd MARS_1.0
```

### 2. virtual environmet installation
```
.../MARS_1.0$ rm -rf .venv  # if there was some before
.../MARS_1.0$ python3 -m venv .name_of_your_new_venv
.../MARS_1.0$ source .name_of_your_new_venv/bin/activate
```

### 3. requirements installation
```
.../MARS_1.0$ pip install --upgrade pip
.../MARS_1.0$ pip install -r requirements.txt
.../MARS_1.0$ pip install -e .
```

### 4. register your jupyter kernel
```
.../MARS_1.0$ python -m ipykernel install --user --name name_of_your_new_kernel --display-name "Name_Of_Your_New_Kernel_To_Visualize"
```