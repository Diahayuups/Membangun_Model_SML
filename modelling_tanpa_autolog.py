import mlflow
import pandas as pd
import joblib
import os
import json

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. Set experiment
mlflow.set_experiment("Breast Cancer - Skilled (No Autolog)")

# 2. Load dataset
data = pd.read_csv("breast_cancer_clean.csv")

X = data.drop("target", axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Train model
model = LogisticRegression(C=1.0, max_iter=1000)

with mlflow.start_run():

    model.fit(X_train, y_train)

    # 4. Predict
    y_pred = model.predict(X_test)

    # 5. Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # MANUAL LOGGING
    mlflow.log_param("C", 1.0)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    # 6. ARTIFACTS (MULAI MIRIP MENTOR)
    os.makedirs("model", exist_ok=True)

    # Artifact 1: model file
    joblib.dump(model, "model/model.pkl")
    mlflow.log_artifact("model/model.pkl")

    # Artifact 2: confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.savefig("model/training_confusion_matrix.png")
    mlflow.log_artifact("model/training_confusion_matrix.png")

    # Artifact 3: metric json
    with open("model/metric_info.json", "w") as f:
        json.dump({"accuracy": acc, "f1_score": f1}, f)
    mlflow.log_artifact("model/metric_info.json")

    print("Accuracy:", acc)
    print("F1 Score:", f1)
