# Optimal LTE Base Station Placement

This university project explores how computational geometry and clustering can support the placement of LTE base stations among user equipment (UE) locations in a two-dimensional area. The implementation uses K-means clustering to estimate base-station positions and evaluates the resulting network through coverage, distance, load-balancing, interference, and fairness metrics.

## Authors

- [Leonidas Gourvelos](https://github.com/LeonGour)
- [Giorgos Lalas](https://github.com/GEOLAL03)

## Project overview

Given a CSV file containing UE coordinates, the program:

- places a configurable number of base stations with K-means;
- assigns every UE to its nearest base station;
- calculates mean and median UE-to-BS distance and K-means inertia;
- estimates coverage and dead-area percentages for a selected coverage radius;
- evaluates SIR-based interference probability;
- reports base-station load, coefficient of variation, and Jain's fairness index;
- supports multiple random restarts and comparisons across several values of `k`;
- creates scatter, Voronoi, convergence, distribution, and summary visualizations.

The experiments documented in the report illustrate the trade-off between improved coverage from additional base stations and the corresponding increase in interference and infrastructure cost.

## Repository structure

```text
.
|-- data/
|   `-- README.md
|-- docs/
|   |-- presentation.pdf
|   `-- report.pdf
|-- src/
|   `-- lte_base_station_placement.py
|-- .gitattributes
|-- .gitignore
|-- README.md
`-- requirements.txt
```

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

## Input data

The script expects a CSV file with these columns:

```csv
x,y
1.25,3.80
2.10,4.15
```

The original task datasets are not included in this repository. Place a compatible CSV file in `data/` or provide its path directly with `--csv`.

## Usage

Run one placement experiment:

```bash
python src/lte_base_station_placement.py --csv data/ue_coordinates.csv --n_bs 3 --radius 0.7 --sir_db 0
```

Use several random restarts and keep the solution with the lowest inertia:

```bash
python src/lte_base_station_placement.py --csv data/ue_coordinates.csv --n_bs 5 --restarts 10
```

Compare multiple base-station counts:

```bash
python src/lte_base_station_placement.py --csv data/ue_coordinates.csv --k_sweep 3,4,5,6
```

Use `--help` to see every available option:

```bash
python src/lte_base_station_placement.py --help
```

By default, generated figures and CSV summaries are saved in `enhanced_results_2/`.

## Evaluation metrics

- Mean and median UE-to-BS distance
- K-means inertia
- Coverage and dead-area percentage
- SIR-based interference probability
- Load coefficient of variation
- Jain's fairness index

## Documentation

- [Full project report](docs/report.pdf)
- [Project presentation](docs/presentation.pdf)

## Technologies

Python, NumPy, pandas, SciPy, Matplotlib, K-means clustering, and Voronoi diagrams.
