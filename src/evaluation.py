"""
Evaluation utils
"""

import os
import numpy as np
import pandas as pd
from dvclive import Live
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')
    
def get_optimal_threshold(model, df, FEATURES, TARGET_NAME, step=0.001, booster=False):
    thresholds = np.arange(0, 1, step)
    y = df[TARGET_NAME]
    probs = model.predict_proba(df[FEATURES])[:, 1]
    scores = [metrics.f1_score(y, to_labels(probs, t)) for t in thresholds]
    ix = np.argmax(scores)
    print('Threshold=%.3f, F-Score=%.5f' % (thresholds[ix], scores[ix]))
    return thresholds[ix], scores[ix]

def save_plot(path, func, **kwargs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plot = func(**kwargs)
    plt.savefig(path, format="png", dpi=220)
    plt.clf()
    
def evaluate(name, model, features, dataset):
    live = Live("eval_plots")
    X, y = dataset.X, dataset.y
    
    pred = model.predict_proba(X[features])[:,1]
    
    live.log_plot("roc", y, pred)
    
    roc_auc = metrics.roc_auc_score(y, pred)
    live.log("auc", roc_auc)
    thresh, f1 = get_optimal_threshold(model, dataset.df, features, "TARGET")
    
    precision = metrics.precision_score(y, 1*(pred >= thresh))
    recall = metrics.recall_score(y, 1*(pred >= thresh))
    accuracy = metrics.accuracy_score(y, 1*(pred >= thresh))
    
    save_plot(
        f"models/scordistr_{name}.png",
        lambda pred: plt.hist(pred, bins=40),
        pred = pred
    )
    
    
    return {
        "name": name,
        "roc_auc": roc_auc,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "thresh": thresh
    }
    