# MARS_1.0
Manga and Anime Research Service (study project)

---

Project for analyzing and working with various datasets (Anime, Cities, etc.).  
Python scripts **expect the datasets to be already downloaded** and placed under the `data/` folder.

---

## Project Structure

```
MARS_1.0/
├── data/
│   ├── anime_ranks/
│   │   ├── anime.csv
│   │   ├── animelist.csv
│   │   ├── anime_with_synopsis.csv
│   │   ├── rating_complete.csv
│   │   └── watching_status.csv
│   ├── anime_timestamps/
│   │   └── anime_timestamps.csv
│   ├── cities_population_and_location/
│   │   └── cities_population_and_location.csv
│   ├── myanimelist_countries_distribution/
│   │   └── myanimelist_countries_distribution.csv
│   └── users/
│       └── profiles.csv
├── db_tools/
│   ├── ...
│   └── ...
├── fds_tools/
│   ├── __init__.py
│   ├── data_cleaner.py
│   ├── fake_user_generator.py (class FakeUsersGenerator)
│   ├── fds_main.py
│   └── project_latex/
│       ├── main.tex
│       ├── chapters/
│       │   ├── introduction.tex
│       │   ├── overview.tex
│       │   ├── fake_user_generation.tex
│       │   ├── dataset_curriculum.tex
│       │   ...
│       │   └── references.bib        
│       └── out/
│           ├── main.pdf
│           └── ...
├── cda_tools/
│   ├── ...
│   └── ...
├── __init__.py
├── .env
├── .gitignore
├── common_tools.py (classes CommonTools, PandasTools, DBTools)
├── datasets.json
├── LICENSE
├── README.md
└── requirements.txt
```

## Setup

### 1. virtual environmet installation
```
.../MARS_1.0$ rm -rf .venv  # if there was some before
.../MARS_1.0$ python3 -m venv name_of_your_new_venv
```
### 2. virtual environment activation
```
.../MARS_1.0$ source name_of_your_new_venv/bin/activate
```
### 2. requirements installation
```
.../MARS_1.0$ pip install -r requirements.txt
```
