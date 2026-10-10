# Dataset Dictionary

The dataset is **synthetic and fictional**, created for practice. It is not a real school's student data.

| Column | Meaning |
|---|---|
| `study_hours` | Example study hours per day |
| `attendance_percent` | Attendance percentage from 0 to 100 |
| `previous_score` | Earlier assessment score from 0 to 100 |
| `final_score` | Final score the model tries to estimate, from 0 to 100 |

The model uses the first three columns as input features and `final_score` as its target. Because the dataset is small and artificially constructed, model metrics are only demonstrations of the workflow.
