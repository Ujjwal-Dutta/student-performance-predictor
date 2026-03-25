import numpy as np
import pandas as pd
import os

# -----------------------------
# Activation
# -----------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# -----------------------------
# Train supervised FCM
# -----------------------------
def train_fcm(csv_path):

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, csv_path)

    df = pd.read_csv(file_path)

    if "Student_ID" in df.columns:
        df = df.drop(columns=["Student_ID"])

    # Normalize
    df = (df - df.min()) / (df.max() - df.min())

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    W = np.linalg.lstsq(X, y, rcond=None)[0]

    return W


# Train once
W_trained = train_fcm("dataset.csv")


# -----------------------------
# Predict (not FCM loop anymore)
# -----------------------------
def run_fcm(initial_state, W):

    x = initial_state[:-1]   # remove output placeholder

    score = sigmoid(np.dot(x, W))

    return score
