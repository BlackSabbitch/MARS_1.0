# MARS_1.0
Manga and Anime Research Service (study project)

---

Comprehensive Python project for managing, cleaning, and analyzing large datasets related to anime, users, and cities.
Includes pipelines for dataset preprocessing, user-anime interactions, and analytical scripts for generating insights and visualizations.
The project also provides modules for database operations (MongoDB, MySQL), graph-based analysis, and reporting.
All datasets should be placed in the data/ directory before running any scripts or notebooks.

---

## Project Structure

```
MARS_1.0/
├── data/                                           # non-code data
│   ├── datasets/                                   # original and self-made datasets
│   │   ├── anime_azathoth42/                       # (unpacked) https://www.kaggle.com/datasets/azathoth42/myanimelist (2018)
│   │   ├── anime_hernan4444/                       # (unpacked) https://www.kaggle.com/datasets/hernan4444/anime-recommendations-database (2020)
│   │   ├── cities_population_and_location/         # (unpacked) https://www.kaggle.com/datasets/donatoriccio/world-cities-population-cleaned-version (2022)
│   │   └── myanimelist_countries_distribution/     # (self-made)  https://www.semrush.com/analytics/organic/changes/?sortField=traffic&db=us&q=myanimelist.net&searchType=domain (2025)
│   ├── graphs/                                     # graph raw files
│   └── helpers/                                    # JSON, CSV - helpers, temporary files
├── project_cda/                                    # CDA project code
├── project_db/                                     # ADB project code
├── project_fds/                                    # FDS project code
├── setup.py
├── requirements.txt
├── pyproject.toml
├── __init__.py
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