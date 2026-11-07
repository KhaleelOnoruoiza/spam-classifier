
import csv
from pathlib import Path
from datetime import datetime


BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = BASE_DIR / "experiment_log.csv"


def log_experiment(
    model,
    feature_extraction,
    train_test_split,
    threshold,
    accuracy,
    precision,
    recall,
    f1_score,
    notes=""
):
    """Save model experiment results to a CSV file."""

    file_exists = LOG_FILE.exists()

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as file:

        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "experiment_id",
                "date",
                "model",
                "feature_extraction",
                "train_test_split",
                "threshold",
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "notes"
            ])

        experiment_id = f"EXP{datetime.now():%Y%m%d%H%M%S}"

        writer.writerow([
            experiment_id,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model,
            feature_extraction,
            train_test_split,
            threshold,
            accuracy,
            precision,
            recall,
            f1_score,
            notes
        ])

    print(f"Experiment {experiment_id} recorded successfully.")


if __name__ == "__main__":

    print("SMS Guard Experiment Tracker")
    print("Import log_experiment() into your training script to record results.")
