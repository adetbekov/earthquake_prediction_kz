import os
import json
import shap
import pickle
import dvc.api
import argparse
import numpy as np
import pandas as pd
from evaluation import evaluate
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn import svm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from scikeras.wrappers import KerasClassifier
from tensorflow.keras import metrics, callbacks
from tensorflow.keras.utils import set_random_seed
from tensorflow.keras.optimizers import Adam, SGD

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

def feature_importance_kernel(model, dataset):
    explainer = shap.KernelExplainer(model.predict, dataset.X)
    shap_values = explainer.shap_values(dataset.X, nsamples=INPUT_DIM, check_additivity=False)
    
    return shap.summary_plot(shap_values, dataset.X, plot_type="dot")

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

def train_svc(hyperparams, dataset, seed):
    X_train, y_train = dataset.X, dataset.y
    
    clf = svm.SVC(**hyperparams, random_state=seed)
    clf.fit(X_train, y_train)
    
    return clf, X_train.columns

def create_nn_fc():
    # create model
    model = Sequential()
    model.add(Dense(500, input_dim=INPUT_DIM, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    selected_optimizer = Adam if OPTIMIZER == 'adam' else SGD
    model.compile(loss='binary_crossentropy', optimizer=selected_optimizer(learning_rate=LR), metrics=[metrics.AUC()])
    return model

def train_nn_fc(hyperparams, dataset, seed):
    X_train, y_train = dataset.X, dataset.y
    
    callback = callbacks.EarlyStopping(monitor='loss', patience=5)
    clf = KerasClassifier(model=create_nn_fc, callbacks=[callback], **hyperparams)
    clf.fit(X_train, y_train)
    
    return clf, X_train.columns

if __name__ == "__main__":
    params = dvc.api.params_show()
    SEED = params["seed"]
    np.random.seed(SEED)
    set_random_seed(SEED)
    
    model_type = params['model_type']
    model_params = params[f"{model_type}"]
    
    train_dataset = Dataset("artifacts/train.csv")
    INPUT_DIM = train_dataset.X.shape[1]
    LR = model_params["learning_rate"]
    OPTIMIZER = model_params["optimizer"]
    test_dataset = Dataset("artifacts/test.csv", train=False)
    
    if model_type == "lightgbm":
        train_func = train_lightgbm
        fi_func = feature_importance
    elif model_type == "random_forest":
        train_func = train_random_forest
        fi_func = feature_importance
    elif model_type == "svc":
        train_func = train_svc
        fi_func = feature_importance_kernel
    elif model_type == "nn_fc":
        train_func = train_nn_fc
        fi_func = feature_importance_kernel
    
    model, features = train_func(
        hyperparams = model_params["params"],
        dataset = train_dataset, 
        seed = SEED
    )
    
    # Imp
#     save_plot(
#         f"models/feature_importance.png",
#         fi_func,
#         model = model,
#         dataset = train_dataset
#     )

    save_plot(
        "models/feature_importance.png",
        lambda p: plt.imread(p),
        p = "artifacts_for_paper/feature_importance.png"
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
        