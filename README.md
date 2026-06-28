# Spotify Genre Tags, Collaboration Networks, and Shared Labels

This repository contains the code and selected outputs for a term paper on genre formation in the platform era. The project compares three ways of representing artist proximity within a bounded UK-based / UK-affiliated alternative music scene:

1. Spotify genre similarity;
2. track-level collaboration;
3. shared release-label affiliation.

The main aim of the project is to examine whether Spotify genre tags correspond to collaboration-based artist groupings, and whether an additional institutional layer, based on shared-label metadata, helps explain collaboration beyond genre similarity.

The study is exploratory and was developed as a coursework project. It should not be interpreted as a complete map of the UK music industry or as a representative sample of all UK alternative artists.

## Project overview

The analysis is based on a manually validated sample of 96 UK-based / UK-affiliated artists associated with contemporary alternative, electronic, experimental, indie, pop-adjacent, and underground scenes.

For this sample, three artist–artist networks are constructed:

* **Genre-similarity network**: artists are connected when their Spotify artist genre tags overlap.
* **Collaboration network**: artists are connected when they co-appear in Spotify track artist credits.
* **Shared-label network**: artists are connected when their releases are associated with at least one shared cleaned release-label name.

The three layers are compared using descriptive network statistics, visual network inspection, community comparison, edge overlap, and QAP regression.

## Repository structure

```text
.
├── code/
│   ├── 01_spotify_data_collection.py
│   ├── 02_genre_network.py
│   ├── 03_collaboration_network.py
│   ├── 04_label_network.py
│   ├── 05_qap_analysis.R
│   └── 11_artist_descriptive_profile_final_figures.py
│
├── data/
│   └── input/
│       ├── final_artist_sample.csv
│       └── final_artist_country_validation.csv
│
├── outputs/
│   ├── figures/
│   ├── networks/
│   └── tables/
│
├── docs/
│   └── mapping_criteria.txt
│
├── README.md
├── requirements.txt
├── .env.example
└── .gitignore
```

Depending on the exact repository version, some intermediate or auxiliary files may be stored separately. The main analytical outputs used in the paper are located in `outputs/`.

## Data sources

The main data source is the Spotify Web API. Spotify data were used to collect:

* artist identifiers;
* artist genre tags;
* popularity and follower indicators;
* release and track metadata;
* track-level artist credits;
* release-label metadata.

Spotify does not provide a direct and reliable country or scene variable for artists. Therefore, sample construction also involved manual validation and auxiliary checks using external metadata sources such as MusicBrainz, Wikidata, artist pages, label pages, and general web search.

## Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows
.venv\Scripts\activate
```

```bash
# macOS / Linux
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Spotify API credentials

Create a `.env` file in the project root using `.env.example` as a template:

```text
SPOTIPY_CLIENT_ID=your_client_id_here
SPOTIPY_CLIENT_SECRET=your_client_secret_here
SPOTIPY_REDIRECT_URI=your_redirect_uri_here
```

The real `.env` file is not included in the repository and should not be committed to GitHub.

## Running the pipeline

The main scripts should be run in the following order:

```bash
python code/01_spotify_data_collection.py
python code/02_genre_network.py
python code/03_collaboration_network.py
python code/04_label_network.py
Rscript code/05_qap_analysis.R
python code/11_artist_descriptive_profile_final_figures.py
```

### Script description

| Script                                           | Description                                                                                                       |
| ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| `01_spotify_data_collection.py`                  | Collects Spotify artist, release, track, genre, collaboration, and label metadata for the selected artist sample. |
| `02_genre_network.py`                            | Constructs the Spotify genre-similarity network using artist genre tags.                                          |
| `03_collaboration_network.py`                    | Constructs the track-level collaboration network from Spotify track artist credits.                               |
| `04_label_network.py`                            | Cleans release-label metadata and constructs the shared-label network.                                            |
| `05_qap_analysis.R`                              | Runs multilayer comparison and QAP regression models.                                                             |
| `11_artist_descriptive_profile_final_figures.py` | Produces descriptive figures for the sample profile and collaboration centrality.                                 |

Some scripts may require already prepared input files in `data/input/`. If the final input files are available, the full data collection stage does not necessarily need to be repeated.

## Outputs

The repository includes selected final outputs used in the term paper.

### `outputs/tables/`

Contains descriptive tables, network statistics, edge-overlap results, community comparison results, and QAP regression outputs.

### `outputs/figures/`

Contains figures used in the paper, including:

* Spotify popularity distribution;
* follower distribution;
* genre-tag coverage;
* top Spotify genre tags;
* label-count distribution;
* top release-label names;
* collaboration degree distribution;
* top artists by collaboration degree;
* follower count and collaboration degree scatterplot.

### `outputs/networks/`

Contains final network files, including GraphML files for visualization and analysis:

* genre-similarity network;
* collaboration network;
* cleaned shared-label network;
* union network.

The union network combines all artist pairs connected by at least one of the observed relations: genre similarity, track-level collaboration, or cleaned shared-label affiliation. It is used mainly for overview visualization and layout comparison.

## Main empirical results

The final analysis is based on 96 artists.

Main findings:

* Spotify genre similarity is positively associated with collaboration ties, but the association is weak.
* The Spotify genre-similarity network is sparse and fragmented, partly because many artists in the sample have no Spotify genre tags.
* Collaboration ties do not fully reproduce Spotify genre boundaries.
* The cleaned shared-label network is denser than the genre and collaboration networks.
* Shared-label affiliation has the strongest overlap with collaboration ties.
* QAP regression results suggest that shared-label affiliation helps explain collaboration structure beyond Spotify genre similarity.
* The three layers are related, but they do not reproduce the same network structure.

The main conclusion is that the selected UK-based / UK-affiliated alternative scene becomes visible only through a multilayer perspective. Genre tags, collaborations, and shared labels each capture different forms of artist proximity.

## Reproducibility notes

This project depends on Spotify API access. Results may not be perfectly reproducible in the future because Spotify metadata can change over time. Artist genre tags, popularity scores, follower counts, release metadata, and label strings may be updated by the platform.

The repository includes selected outputs used for the coursework submission to make the analysis easier to inspect without rerunning all API requests.

## Limitations

This project has several important limitations:

* The sample is exploratory and not representative of all UK alternative music.
* Spotify metadata are platform-specific and incomplete.
* Some artists have no Spotify genre tags.
* Track-level co-appearance captures only one visible form of collaboration.
* The shared-label layer indicates institutional proximity, not direct collaboration.
* QAP regression models show associations between network layers, not causal effects.

## Security note

Spotify API credentials are not included in this repository. The following files should never be committed:

```text
.env
*.log
*token*
*secret*
*credentials*
```

Use `.env.example` only as a template.

## Suggested citation

If referring to the stable coursework submission version, use the tagged release:

```text
v1.0-coursework-submission
```

## License

This repository is intended for academic coursework and reproducibility purposes. If reused, please cite the repository and respect Spotify API terms of use.
