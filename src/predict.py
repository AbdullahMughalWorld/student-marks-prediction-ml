"""Predict a sample student's score using the trained model."""
from pathlib import Path
import sys

try:
    import joblib
    import pandas as pd
except ImportError:
    print("Missing dependency. Run: python -m pip install -r requirements.txt")
    raise SystemExit(1)

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = ROOT / "models" / "student_score_model.joblib"


def main() -> int:
    if not MODEL_FILE.exists():
        print("Trained model not found. Run 'python src/train_model.py' first.")
        return 1
    bundle = joblib.load(MODEL_FILE)
    model, features = bundle["model"], bundle["features"]
    try:
        study_hours = float(input("Study hours per day: "))
        attendance = float(input("Attendance percentage (0-100): "))
        previous_score = float(input("Previous score (0-100): "))
    except ValueError:
        print("Please enter numeric values only.")
        return 1
    if study_hours < 0 or not 0 <= attendance <= 100 or not 0 <= previous_score <= 100:
        print("Invalid input: hours must be non-negative and percentages/scores 0-100.")
        return 1
    sample = pd.DataFrame([{
        "study_hours": study_hours,
        "attendance_percent": attendance,
        "previous_score": previous_score
    }], columns=features)
    prediction = float(model.predict(sample)[0])
    print(f"\nEstimated final score: {prediction:.2f}/100")
    print("This is a practice estimate from a small synthetic dataset, not a guaranteed result.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
