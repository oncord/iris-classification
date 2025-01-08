import os
import joblib
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def extract():
    try:
        iris = load_iris()
        df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])
        df.to_csv("iris_raw.csv", index=False)
        print("Data extraction complete.")
    except Exception as e:
        print(f"Error in extraction: {e}")

def transform():
    try:
        df_t = pd.read_csv("iris_raw.csv")
        X = df_t.iloc[:, :-1]
        y = df_t['target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        os.makedirs('/data/processed', exist_ok=True)
        X_train.to_csv('/data/processed/X_train.csv', index=False)
        y_train.to_csv('/data/processed/y_train.csv', index=False)
        X_test.to_csv('/data/processed/X_test.csv', index=False)
        y_test.to_csv('/data/processed/y_test.csv', index=False)

        print("Data transformation complete.")
    except Exception as e:
        print(f"Error in transformation: {e}")

def train():
    try:
        X_train = pd.read_csv('/data/processed/X_train.csv')
        y_train = pd.read_csv('/data/processed/y_train.csv')

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        os.makedirs('models', exist_ok=True)
        joblib.dump(model, 'models/iris_model.pkl')
        print("Model training complete.")
    except Exception as e:
        print(f"Error in training: {e}")

def evaluate():
    try:
        X_test = pd.read_csv('/data/processed/X_test.csv')
        y_test = pd.read_csv('/data/processed/y_test.csv')

        model = joblib.load('models/iris_model.pkl')

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Model accuracy: {accuracy}")
        return accuracy
    except Exception as e:
        print(f"Error in evaluation: {e}")
        return None
