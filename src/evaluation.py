"""
Evaluation utils
"""

import os
import numpy as np
import pandas as pd
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

def get_roc_auc_curve(preds, y):
    fpr, tpr, threshold = metrics.roc_curve(y, preds)
    roc_auc = metrics.auc(fpr, tpr)
    
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    
    return plt
    
def evaluate(name, model, features, df_path, na_strategy):
    DF = pd.read_csv(df_path, sep=";")
    X, y = DF.drop(["mag_max", "TARGET"], axis=1), DF["TARGET"]
    
    pred = model.predict_proba(X[features])[:,1]
    
    roc_auc = metrics.roc_auc_score(y, pred)
    thresh, f1 = get_optimal_threshold(model, DF, features, "TARGET")
    
    precision = metrics.precision_score(y, 1*(pred >= thresh))
    recall = metrics.recall_score(y, 1*(pred >= thresh))
    accuracy = metrics.accuracy_score(y, 1*(pred >= thresh))
    
    roc_auc_curve = get_roc_auc_curve(pred, y)
    roc_auc_curve_filename = f"artifacts/{na_strategy}/{name}_plots/roc_auc.png"
    os.makedirs(os.path.dirname(roc_auc_curve_filename), exist_ok=True)
    roc_auc_curve.savefig(roc_auc_curve_filename, format="png", dpi=300)
    plt.clf()
    
    distribution_filename = f"artifacts/{na_strategy}/{name}_plots/distribution.png"
    os.makedirs(os.path.dirname(distribution_filename), exist_ok=True)
    plt.hist(pred, bins=40)
    plt.savefig(distribution_filename, format="png", dpi=300)
    plt.clf()
    
    DF["prob"] = pred
    DF["pred_class"] = 1*(DF["prob"] >= thresh)
    
    return {
        "name": f"{name}_{na_strategy}",
        "roc_auc": roc_auc,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "thresh": thresh
    }
    