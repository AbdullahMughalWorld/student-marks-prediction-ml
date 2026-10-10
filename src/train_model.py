"""Train and evaluate a simple student-score prediction model."""
from pathlib import Path
import sys

try:
    import joblib
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
except ImportError:
    print("Missing dependency. Run: python -m pip install -r requirements.txt")
    raise SystemExit(1)

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "student_scores.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
FEATURES = ["study_hours", "attendance_percent", "previous_score"]
TARGET = "final_score"


def load_dataset(path: Path) -> pd.DataFrame:
    """Read and validate the sample dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    required = FEATURES + [TARGET]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    if df.empty:
        raise ValueError("Dataset is empty.")
    for column in required:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if df[required].isna().any().any():
        raise ValueError("Dataset contains blank or non-numeric values.")
    for column in ["attendance_percent", "previous_score", TARGET]:
        if not df[column].between(0, 100).all():
            raise ValueError(f"{column} values must be between 0 and 100.")
    if (df["study_hours"] < 0).any():
        raise ValueError("Study hours cannot be negative.")
    return df


def main() -> int:
    try:
        df = load_dataset(DATA_FILE)
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        print(f"Error: {exc}")
        return 1

    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "student_score_model.joblib"
    joblib.dump({"model": model, "features": FEATURES}, model_path)

    comparison = X_test.copy()
    comparison["actual_final_score"] = y_test.values
    comparison["predicted_final_score"] = predictions.round(2)
    comparison.to_csv(OUTPUT_DIR / "predictions.csv", index=False)

    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, predictions)
    plt.plot([0, 100], [0, 100], linestyle="--")
    plt.title("Actual vs Predicted Final Scores")
    plt.xlabel("Actual score")
    plt.ylabel("Predicted score")
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close()

    print("\n=== Student Marks Prediction (Linear Regression) ===")
    print(f"Dataset rows: {len(df)}")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(f"Mean Absolute Error (MAE): {mae:.2f} marks")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} marks")
    print(f"R-squared (R²): {r2:.3f}")
    print("\nSample predictions:")
    print(comparison.to_string(index=False))
    print(f"\nSaved model: {model_path.relative_to(ROOT)}")
    print("Saved predictions: outputs/predictions.csv")
    print("Saved chart: outputs/actual_vs_predicted.png")
    print("\nReminder: this is a small synthetic dataset for learning, not a reliable real-world predictor.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
