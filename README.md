# earthquake_prediction_kz

### Description

Strong earthquake events prediction


### Installation

First, you should install poetry. And run this.

```
git clone git@github.com:adetbekov/earthquake_prediction_kz.git

# Enter to virtual env:
poetry shell

# If you want to update dependencies, run:
poetry lock

# Installation
poetry install
```

To reproduce experiment, you should run:

```
poetry run dvc repro
```

To show all experiments:

```
poetry run dvc exp show --sort-by roc_auc --sort-order desc
```

To generate plots of all experiments:

```
poetry run dvc plots diff lightgbm random_forest nn_fc
```
