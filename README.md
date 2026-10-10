# Student Marks Prediction using Machine Learning

A beginner-friendly machine-learning project that trains a **Linear Regression** model to estimate a student's final score from study hours, attendance, and a previous score.

> **Learning disclaimer:** The included dataset is synthetic, small, and fictional. Predictions and metrics are demonstrations only and must not be used to make real educational decisions.

## Features
- Loads and validates CSV data with Pandas
- Splits data into training and test sets
- Trains a Linear Regression model with scikit-learn
- Evaluates with MAE, RMSE, and R²
- Saves a trained model using Joblib
- Writes actual/predicted scores to CSV
- Saves an actual-vs-predicted chart
- Includes an interactive prediction script and unit tests

## Tech stack
Python 3.10+, Pandas, scikit-learn, Matplotlib, Joblib, `unittest`.

## Project structure
```text
StudentMarksPredictionML/
├── data/student_scores.csv
├── docs/data-dictionary.md
├── models/
├── outputs/
├── src/
│   ├── predict.py
│   └── train_model.py
├── tests/test_model.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Setup and run
1. Open a terminal in the project root.
2. Install packages:
   ```bash
   python -m pip install -r requirements.txt
   ```
3. Train and evaluate:
   ```bash
   python src/train_model.py
   ```
4. Try a prediction after training:
   ```bash
   python src/predict.py
   ```
5. Run tests:
   ```bash
   python -m unittest discover -s tests -v
   ```

If `python` is not recognized on Windows, try `py` instead. A virtual environment is recommended.

## Concepts demonstrated
- **Features:** input columns used by the model
- **Target:** the final score being estimated
- **Train/test split:** evaluates the model on records not used for training
- **Linear Regression:** fits a linear relationship between inputs and target
- **MAE:** average absolute prediction error in score points
- **RMSE:** error metric that penalizes larger errors more strongly
- **R²:** compares model predictions with a mean-target baseline; with a small test set it can vary widely and can be negative

## Limitations and responsible use
- Data is synthetic and contains only 20 rows.
- Metrics from this tiny dataset do not indicate real-world accuracy.
- Correlation and prediction do not establish causation.
- Do not use this model to label, rank, or make decisions about real students.
- Serious research requires a larger, representative, ethically collected dataset and stronger evaluation.

## Future improvements
- Compare Linear Regression with Decision Tree and Random Forest
- Use cross-validation and more data
- Add residual plots and feature analysis
- Build a Streamlit dashboard

## Author
**Abdullah Mughal**  
BS Artificial Intelligence student | Learning Python, Data Analysis, and Machine Learning
