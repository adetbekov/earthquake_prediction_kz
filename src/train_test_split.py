import dvc.api
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def process(path, test_size=0.2, dropzero_train=False, dropzero_test=False, seed=42):
    df = pd.read_csv(path, sep=";")
    
    X_train, X_test, y_train, y_test = train_test_split(
        df.fillna(0).drop(["TARGET"], axis=1),
        df["TARGET"],
        test_size=test_size,
        random_state=seed
    )
    
    TRAIN_CLEAN = X_train.copy()
    TRAIN_CLEAN["TARGET"] = y_train
    TEST_CLEAN = X_test.copy()
    TEST_CLEAN["TARGET"] = y_test
    
    if dropzero_train:
        TRAIN_CLEAN = TRAIN_CLEAN[TRAIN_CLEAN["mag_max"] != 0]
    if dropzero_test:
        TEST_CLEAN = TEST_CLEAN[TEST_CLEAN["mag_max"] != 0]
    
    return TRAIN_CLEAN, TEST_CLEAN
    

if __name__ == "__main__":
    params = dvc.api.params_show()
    
    TRAIN_CLEAN, TEST_CLEAN = process(
        f"artifacts/data.csv", 
        params["train_test_split"]["test_size"], 
        params["train_test_split"]["dropzero_train"],
        params["train_test_split"]["dropzero_test"],
        params["seed"]
    )
    
    TRAIN_CLEAN.to_csv(f"artifacts/train.csv", sep=";", index=False)
    TEST_CLEAN.to_csv(f"artifacts/test.csv", sep=";", index=False)
    