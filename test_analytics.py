import unittest
import pandas as pd
from analytics import profile_dataframe, validate_data_dictionary


class AnalyticsTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({"group": ["A", "B", None, "A"], "score": [1, 2, 3, 100]})
        self.dictionary = pd.DataFrame([
            {"column_name": "group", "description": "Group label", "data_type": "categorical", "required": "false", "allowed_values": "A|B"},
            {"column_name": "score", "description": "Score", "data_type": "numeric", "required": "true", "allowed_values": ""},
        ])

    def test_profile_reports_missingness_and_outliers(self):
        profile = profile_dataframe(self.data).set_index("column_name")
        self.assertEqual(profile.loc["group", "missing_count"], 1)
        self.assertEqual(profile.loc["score", "missing_count"], 0)
        self.assertGreaterEqual(profile.loc["score", "IQR_outliers"], 1)

    def test_dictionary_rejects_unlisted_allowed_value(self):
        data = self.data.copy()
        data.loc[1, "group"] = "C"
        result = validate_data_dictionary(data, self.dictionary)
        self.assertFalse(result["ok"])
        self.assertIn("outside allowed_values", result["message"])

    def test_dictionary_accepts_matching_values(self):
        data = self.data.copy()
        data["group"] = ["A", "B", "A", "A"]
        data["score"] = [1, 2, 3, 4]
        result = validate_data_dictionary(data, self.dictionary)
        self.assertTrue(result["ok"])

    def test_dictionary_requires_all_dataset_columns(self):
        result = validate_data_dictionary(self.data, self.dictionary.iloc[:1])
        self.assertFalse(result["ok"])
        self.assertIn("score", result["message"])


if __name__ == "__main__":
    unittest.main()
