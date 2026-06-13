import joblib
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/VersoML/mlops-training-data/main/churn.parquet"
MODEL_PATH = "model.joblib"
OUTPUT_PATH = "predictions.parquet"


def infer(data_path, model_path, output_path):
    df = pd.read_parquet(data_path)
    X = df.drop(columns=["churned"], errors="ignore")

    model = joblib.load(model_path)

    out = df.copy()
    out["prediction"] = model.predict(X)
    out["churn_probability"] = model.predict_proba(X)[:, 1]

    out.to_parquet(output_path)
    print(f"Predictions written to {output_path} ({len(out)} rows)")


if __name__ == "__main__":
    infer(DATA_URL, MODEL_PATH, OUTPUT_PATH)
