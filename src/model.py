import os
import json
import shap
import pickle
import dvc.api
import argparse
import pandas as pd
from evaluation import evaluate
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

class Dataset:
    def __init__(self, path, train=True):
        self.df = pd.read_csv(path, sep=";")
        self.drop_cols = ["mag_max", "TARGET"]
        if train:
            self.drop_cols += ["year_", "REGION_"]
        self.X = self.df.drop(
            self.drop_cols, 
            axis = 1
        )
        self.y = self.df["TARGET"]

    
def feature_importance(model, dataset):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(dataset.X)
    
    return shap.summary_plot(shap_values[1], dataset.X, plot_type="dot")

def save_plot(path, func, **kwargs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plot = func(**kwargs)
    plt.savefig(path, format="png", dpi=220)
    plt.clf()

def train_lightgbm(hyperparams, dataset, seed):
    X_train, y_train = dataset.X, dataset.y
    
    clf = lgb.LGBMClassifier(**hyperparams, random_state=seed)
    clf.fit(X_train, y_train)
    
    return clf, X_train.columns

def train_random_forest(hyperparams, dataset, seed):
    X_train, y_train = dataset.X, dataset.y
    
    clf = RandomForestClassifier(**hyperparams, random_state=seed)
    clf.fit(X_train, y_train)
    
    return clf, X_train.columns
    

if __name__ == "__main__":
    params = dvc.api.params_show()
    model_type = params['model_type']
    model_params = params[f"{model_type}"]
    
    train_dataset = Dataset("artifacts/train.csv")
    test_dataset = Dataset("artifacts/test.csv", train=False)
    
    if model_type == "lightgbm":
        train_func = train_lightgbm
    elif model_type == "random_forest":
        train_func = train_random_forest
    
    model, features = train_func(
        hyperparams = model_params["params"],
        dataset = train_dataset, 
        seed = params["seed"]
    )
    
    # Imp
    save_plot(
        f"models/feature_importance.png",
        feature_importance,
        model = model,
        dataset = train_dataset
    )
    
    # Metircs
    metrics_test = evaluate(
        name = model_type,
        model = model, 
        features = features,
        dataset = test_dataset
    )
    
    os.makedirs("models/", exist_ok=True)
    
    pickle.dump(features, open(f"models/features.pkl", 'wb'))
    pickle.dump(model, open(f"models/model.pkl", 'wb'))
    
    with open(f"models/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_test, f, ensure_ascii=False, indent=4)
        