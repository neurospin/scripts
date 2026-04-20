# 2025_NeuroLithe: response to Lithium

## Installation

Install pixi

```
curl -fsSL https://pixi.sh/install.sh | bash
```

Install packages

```
pixi init 2025_NeuroLithe
cd 2025_NeuroLithe
pixi add python
pixi add scikit-learn pandas statsmodels seaborn openpyxl ipykernel
```

Run environment

```
pixi shell
```

## Organisation

Follow: [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/)

### Python files at root of the project

- `config.py`: Configuration file, define `config` dictionnary with all you need
- `ml_utils.py`: Utils for machine learning
- `01_statistics.py`: Univariate statistics
- `10_classif_repeatedcv.py`: Repeated CV
- `11_classif_permutations_scores_featureimportance.py`: Permutations, for classification score and feature importance

### Organisation

```
├── data               <- Input data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
└─── models             <- Trained and serialized models, model predictions, or model summaries
```

### TODO

- Descriptive statistics to add to univariate table
- ROC curve
- Feature importance figure

