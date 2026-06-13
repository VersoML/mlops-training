import joblib
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_URL = "https://raw.githubusercontent.com/VersoML/mlops-training-data/main/churn.parquet"
MODEL_PATH = "model.joblib"


def train(data_path):
    df = pd.read_parquet(data_path)

    X = df.drop(columns=["churned"])
    y = df["churned"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    pre = make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore"), ["plan"]),
        remainder="passthrough",
    )
    model = make_pipeline(
        pre,
        MLPClassifier(hidden_layer_sizes=(50,), max_iter=100, solver="lbfgs"),
    )
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)
    auc = roc_auc_score(y_test, probs[:, 1])
    loss = log_loss(y_test, probs)
    print(f"auc={auc:.4f} log_loss={loss:.4f}")

    joblib.dump(model, MODEL_PATH)

    return model


if __name__ == "__main__":
    train(DATA_URL)
