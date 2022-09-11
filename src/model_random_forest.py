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

class Dataset:
    def __init__(self, path):
        self.df = pd.read_csv(path, sep=";")
        self.X = self.df.drop(
            ["mag_max", "TARGET", "year_", "REGION_"], 
            axis = 1
        )
        self.y = self.df["TARGET"]

    
def feature_importance(model, dataset):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(dataset.X)
    
    return shap.summary_plot(shap_values[0], dataset.X, plot_type="violin")

# def dependence_plot(model, dataset):
#     explainer = shap.TreeExplainer(model)
#     shap_values = explainer.shap_values(dataset.X)
    
#     return shap.dependence_plot("rollmax_10y_mag_max", shap_values[1], dataset.X)

def train(hyperparams, dataset, seed):
    X_train, y_train = dataset.X, dataset.y
    
    clf = RandomForestClassifier(**hyperparams, random_state=seed)
    clf.fit(X_train, y_train)
    
    return clf, X_train.columns
    

if __name__ == "__main__":
    model_type = 'rf'
    parser = argparse.ArgumentParser(description='model')
    parser.add_argument('na_strategy', type=str, help='With what to fill na values [mean, min, zero]')
    args = parser.parse_args()
    na_strategy = args.na_strategy
    
    params = dvc.api.params_show()
    model_params = params["random_forest"]
    
    train_dataset = Dataset(f"artifacts/{na_strategy}/train.csv")
    
    model, features = train(
        hyperparams = model_params["params"],
        dataset = train_dataset, 
        seed = params["seed"]
    )
    
    # Imp
    imp_filename = f"artifacts/{na_strategy}/plots/{model_type}_imp.png"
    os.makedirs(os.path.dirname(imp_filename), exist_ok=True)
    imp = feature_importance(model, train_dataset)
    plt.savefig(imp_filename, format="png", dpi=300)
    plt.clf()
    
#     # Explain first
#     first_dep_filename = f"artifacts/{na_strategy}/plots/{model_type}_first_dep.png"
#     os.makedirs(os.path.dirname(imp_filename), exist_ok=True)
#     first_dep = dependence_plot(model, train_dataset)
#     plt.savefig(first_dep_filename, format="png", dpi=300)
#     plt.clf()
    
#     metrics_train = evaluate(
#         name = f"{model_type}_{na_strategy}_train",
#         na_strategy = na_strategy,
#         model = model,
#         features = features,
#         df_path = f"artifacts/{na_strategy}/train.csv"
#     )
    metrics_test = evaluate(
        name = f"{model_type}_{na_strategy}_test",
        na_strategy = na_strategy,
        model = model, 
        features = features,
        df_path = f"artifacts/{na_strategy}/test.csv"
    )
    
    pickle.dump(features, open(f"artifacts/{na_strategy}/{model_type}_features.pkl", 'wb'))
    pickle.dump(model, open(f"artifacts/{na_strategy}/clf_{model_type}.pkl", 'wb'))
    
#     with open(f"artifacts/{na_strategy}/{model_type}_train_metrics.json", "w", encoding="utf-8") as f:
#         json.dump(metrics_train, f, ensure_ascii=False, indent=4)
    with open(f"artifacts/{na_strategy}/{model_type}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_test, f, ensure_ascii=False, indent=4)
        