import argparse
import glob
import os
import time

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.metrics import DriftedColumnsCount, MeanValue, ValueDrift
from evidently.ui.workspace import RemoteWorkspace

NUMERICAL = ["tenure_days", "nb_logins_30j", "nb_features_used", "support_tickets_90j", "mrr_eur"]
CATEGORICAL = ["plan", "has_integration", "prediction"]
SCHEMA = DataDefinition(numerical_columns=NUMERICAL, categorical_columns=CATEGORICAL)


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df[NUMERICAL + CATEGORICAL].copy()
    df["prediction"] = df["prediction"].astype(int)
    return df


def snapshot_for(reference: Dataset, batch: pd.DataFrame):
    current = Dataset.from_pandas(_prepare(batch), data_definition=SCHEMA)
    return Report(
        [
            DriftedColumnsCount(),
            ValueDrift(column="prediction"),
            MeanValue(column="mrr_eur"),
        ]
    ).run(current, reference)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", default=os.environ.get("CAPTURE_DIR", "/tmp/predictions_log"))
    parser.add_argument("--reference", default="https://raw.githubusercontent.com/VersoML/mlops-training-data/main/reference_scored.parquet")
    parser.add_argument("--workspace-url", default=os.environ.get("EVIDENTLY_WORKSPACE_URL"))
    parser.add_argument("--project-id", default=os.environ.get("EVIDENTLY_PROJECT_ID"))
    parser.add_argument("--secret", default=os.environ.get("EVIDENTLY_SECRET"))
    parser.add_argument("--poll-interval", type=float, default=2.0)
    args = parser.parse_args()

    if not args.workspace_url or not args.project_id:
        raise SystemExit("EVIDENTLY_WORKSPACE_URL and EVIDENTLY_PROJECT_ID must be set")

    reference = Dataset.from_pandas(_prepare(pd.read_parquet(args.reference)), data_definition=SCHEMA)
    ws = RemoteWorkspace(args.workspace_url, secret=args.secret or None)

    seen: set[str] = set()
    print(f"Watching {args.capture_dir} -> project {args.project_id}")
    while True:
        for path in sorted(glob.glob(f"{args.capture_dir}/**/*.parquet", recursive=True)):
            if path in seen:
                continue
            seen.add(path)
            batch = pd.read_parquet(path)
            ws.add_run(args.project_id, snapshot_for(reference, batch), include_data=False, name=os.path.basename(path))
            print(f"pushed snapshot for {path} ({len(batch)} rows)")
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()