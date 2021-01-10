import os
import sys
import copy
import itertools
import unittest

import numpy as np
import pandas as pd

from ColEnum import ColEnum
from NotCSVError import NotCSVError
from ColumnMapError import ColumnMapError
from DataManager import DataManager
from RFClassifier import RFClassifier


class TestRFClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls._test_classifier = RFClassifier()

        return

    def test_normal_setup(self):

        cls = self.__class__

        self.assertEqual(None, cls._test_classifier.get_classifier())

        return

    def test_prime_classifier(self):

        cls = self.__class__

        data_path = os.path.join(
            os.getcwd(), "SparseClassifier", "test_data_files")
        expected_file_path = os.path.join(
            data_path, "usa_Kansas_train_feature10_iter1_unit.csv")

        # Construct the expected dataframe
        expected_df = pd.read_csv(expected_file_path)
        raw_df = copy.deepcopy(expected_df)

        # Drop unnecessary columns
        expected_df = expected_df.drop(columns=["category", "details_name"])

        # Rename the other column names
        renaming_scheme = {
            "x2gr_sin_t1": "Feature0",
            "x3rd_sin_t1": "Feature1",
            "x6re_cos_t1": "Feature2",
            "x6re_cos_t2": "Feature3",
            "x4ni_cos_t1": "Feature4",
            "x4ni_cos_t2": "Feature5",
            "x5sw_sin_t1": "Feature6",
            "x5sw_cos_t2": "Feature7",
            "gcvi_sin_t1": "Feature8",
            "gcvi_cos_t2": "Feature9",
            "details_id": "FieldID",
            "value": "Class",
        }

        expected_df = expected_df.rename(columns=renaming_scheme)

        y_values = expected_df["Class"].values
        X_values = expected_df.drop(columns=["FieldID", "Class"])

        cls._test_classifier.prime_classifier(X_values, y_values)
        self.assertNotEqual(None, cls._test_classifier.get_classifier())

        return

    def test_train_classifier(self):

        cls = self.__class__

        data_path = os.path.join(
            os.getcwd(), "SparseClassifier", "test_data_files")
        expected_file_path = os.path.join(
            data_path, "usa_Kansas_train_feature10_iter1_unit.csv")

        # Construct the expected dataframe
        expected_df = pd.read_csv(expected_file_path)
        raw_df = copy.deepcopy(expected_df)

        # Drop unnecessary columns
        expected_df = expected_df.drop(columns=["category", "details_name"])

        # Rename the other column names
        renaming_scheme = {
            "x2gr_sin_t1": "Feature0",
            "x3rd_sin_t1": "Feature1",
            "x6re_cos_t1": "Feature2",
            "x6re_cos_t2": "Feature3",
            "x4ni_cos_t1": "Feature4",
            "x4ni_cos_t2": "Feature5",
            "x5sw_sin_t1": "Feature6",
            "x5sw_cos_t2": "Feature7",
            "gcvi_sin_t1": "Feature8",
            "gcvi_cos_t2": "Feature9",
            "details_id": "FieldID",
            "value": "Class",
        }

        expected_df = expected_df.rename(columns=renaming_scheme)

        y_values = expected_df["Class"].values
        X_values = expected_df.drop(columns=["FieldID", "Class"])

        cls._test_classifier.train_classifier(X_values, y_values)
        self.assertNotEqual(None, cls._test_classifier.get_classifier())

    def test_pickle_classifier(self):

        cls = self.__class__

        # Get the file and path to save the pickled model
        file_path = os.path.join(os.getcwd(), "test_pickle.pickle")

        cls._test_classifier.pickle_classifier(filepath=file_path)

        # Check that the pickled file exists
        self.assertTrue(os.path.exists(
            os.path.join(os.getcwd(), "test_pickle.pickle")))

        # Delete the file afterwards
        os.remove(os.path.join(
            os.path.join(os.getcwd(), "test_pickle.pickle")))

    def test_get_prediction(self):

        test_classifier = RFClassifier()

        data_path = os.path.join(
            os.getcwd(), "SparseClassifier", "test_data_files")
        expected_file_path = os.path.join(
            data_path, "usa_Kansas_train_feature10_iter1_unit.csv")

        # Construct the expected dataframe
        expected_df = pd.read_csv(expected_file_path)
        raw_df = copy.deepcopy(expected_df)

        # Drop unnecessary columns
        expected_df = expected_df.drop(columns=["category", "details_name"])

        # Rename the other column names
        renaming_scheme = {
            "x2gr_sin_t1": "Feature0",
            "x3rd_sin_t1": "Feature1",
            "x6re_cos_t1": "Feature2",
            "x6re_cos_t2": "Feature3",
            "x4ni_cos_t1": "Feature4",
            "x4ni_cos_t2": "Feature5",
            "x5sw_sin_t1": "Feature6",
            "x5sw_cos_t2": "Feature7",
            "gcvi_sin_t1": "Feature8",
            "gcvi_cos_t2": "Feature9",
            "details_id": "FieldID",
            "value": "Class",
        }

        expected_df = expected_df.rename(columns=renaming_scheme)

        y_values = expected_df["Class"].values
        X_values = expected_df.drop(columns=["FieldID", "Class"])

        test_classifier.prime_classifier(X_values, y_values)

        # Create some dummy data to predict
        test_pred = np.array(
            [[0.007323115, 0.021445044, 0.020201009, 0.026767563, 0.036590024,
                0.016756232, 0.017781036, 0.01937046, -1.504149843, -0.020641898],
             [0.009363148, 0.029038046, 0.02254572, 0.003919989, 0.051379101,
                0.007203569, 0.026204014, -0.003842525, -1.681117267, 0.143642406]]
        )

        self.assertNotEqual(None, test_classifier.get_classifier())
        prediction = test_classifier.get_prediction(test_pred)
        self.assertEqual((2,), prediction.shape)


if __name__ == '__main__':
    unittest.main()
