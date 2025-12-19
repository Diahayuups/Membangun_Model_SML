import mlflow
import mlflow.sklearn
import dagshub
import pandas as pd
import joblib
import os
import json
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# 1) CONNECT TO DAGSHUB (ONLINE TRACKING)
dagshub.init(
    repo_owner="Diahayuups",
    repo_name="breast-cancer-mlflow",
    mlflow=True
)

# 2) SET EXPERIMENT (ONLINE)
mlflow.set_experiment("Breast Cancer - Advance (DagsHub)")

# 3) LOAD DATASET (FILE)
data = pd.read_csv("breast_cancer_clean.csv")
X = data.drop("target", axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4) HYPERPARAMETER TUNING
param_grid = {
    "C": [0.01, 0.1, 1, 10]
}

grid = GridSearchCV(
    LogisticRegression(max_iter=1000),
    param_grid,
    scoring="f1",
    cv=3
)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# 5) MANUAL LOGGING + ARTIFACTS
with mlflow.start_run():

    # Params
    mlflow.log_param("C", grid.best_params_["C"])

    # Metrics
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", acc)

    # Prepare artifact dir
    os.makedirs("model", exist_ok=True)

    # Artifact 1: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.savefig("model/confusion_matrix.png")
    mlflow.log_artifact("model/confusion_matrix.png")

    # Artifact 2: Trained Model
    joblib.dump(best_model, "model/model.pkl")
    mlflow.log_artifact("model/model.pkl")

    # Artifact 3: Metrics JSON (extra)
    with open("model/metrics.json", "w") as f:
        json.dump({"accuracy": acc}, f)
    mlflow.log_artifact("model/metrics.json")

    # Register model artifact (mentor-style)
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        input_example=X_train.iloc[:5]
    )

    print("ONLINE Accuracy:", acc)
