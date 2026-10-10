import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from train_model import load_dataset, FEATURES, TARGET


class ModelDataTests(unittest.TestCase):
    def test_dataset_loads(self):
        df = load_dataset(ROOT / "data" / "student_scores.csv")
        self.assertEqual(len(df), 20)
        self.assertTrue(set(FEATURES + [TARGET]).issubset(df.columns))

    def test_scores_are_in_valid_range(self):
        df = load_dataset(ROOT / "data" / "student_scores.csv")
        self.assertTrue(df[TARGET].between(0, 100).all())

    def test_missing_file_raises_error(self):
        with self.assertRaises(FileNotFoundError):
            load_dataset(ROOT / "data" / "not_a_real_file.csv")


if __name__ == "__main__":
    unittest.main()
