import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 1. Set experiment (lokal)
mlflow.set_experiment("Breast Cancer - Basic Modelling")

# 2. Load dataset (FILE LANGSUNG)
data = pd.read_csv("breast_cancer_clean.csv")

X = data.drop("target", axis=1)
y = data["target"]

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train model + MLflow autolog
with mlflow.start_run():
    mlflow.autolog()

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    print("Accuracy:", acc)
