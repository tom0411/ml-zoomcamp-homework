import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

DATA_FILE = Path(__file__).parent / "heart.csv"
MODEL_FILE = Path(__file__).parent / "model.bin"

NUMERICAL = ["age", "restingbp", "cholesterol", "maxhr", "oldpeak"]
CATEGORICAL = [
    "sex",
    "chestpaintype",
    "fastingbs",
    "restingecg",
    "exerciseangina",
    "st_slope",
]
FEATURES = CATEGORICAL + NUMERICAL


def load_data():
    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    for column in df.select_dtypes(include="object").columns:
        df[column] = df[column].str.lower().str.replace(" ", "_")
    df[["restingbp", "cholesterol"]] = df[["restingbp", "cholesterol"]].replace(0, np.nan)
    return df


def train_model(df):
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
    df_full_train = df_full_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_full_train = df_full_train.heartdisease.values
    y_test = df_test.heartdisease.values
    del df_full_train["heartdisease"]
    del df_test["heartdisease"]

    medians = {}
    for column in ["restingbp", "cholesterol"]:
        medians[column] = float(df_full_train[column].median())
        df_full_train[column] = df_full_train[column].fillna(medians[column])
        df_test[column] = df_test[column].fillna(medians[column])

    dv = DictVectorizer(sparse=False)
    X_full_train = dv.fit_transform(df_full_train[FEATURES].to_dict(orient="records"))
    X_test = dv.transform(df_test[FEATURES].to_dict(orient="records"))

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=1,
        random_state=1,
    )
    model.fit(X_full_train, y_full_train)

    y_pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    return medians, dv, model, auc


def save_model(medians, dv, model):
    with MODEL_FILE.open("wb") as f_out:
        pickle.dump((medians, dv, model), f_out)


if __name__ == "__main__":
    medians, dv, model, auc = train_model(load_data())
    save_model(medians, dv, model)
    print(f"test auc {auc:.3f}")
    print(f"saved {MODEL_FILE.name}")
