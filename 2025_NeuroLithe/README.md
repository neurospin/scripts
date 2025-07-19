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

Run envirement
```
pixi shell
```
## Files

- `config.py`: Configuration file
- `ml_utils.py`: Utils for machine learning
- `01_statistics.py`: Univariate statistics
- `10_classif_repeatedcv.py`: Repeated CV
- `11_classif_permutations_scores_featureimportance.py`: Permuation, for classification score and feature importance


