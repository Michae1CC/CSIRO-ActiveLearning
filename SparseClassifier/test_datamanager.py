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


class TestDataManager(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        return

    @staticmethod
    def unit_column_map(col_name):

        if ("sin" in col_name) or ("cos" in col_name):
            return ColEnum.FEATURE

        if col_name == "details_name":
            return ColEnum.OMIT

        if col_name == "category":
            return ColEnum.OMIT

        if col_name == "details_id":
            return ColEnum.FIELD

        if col_name == "value":
            return ColEnum.CLASS

    def test_incorrect_data_path(self):

        bad_dir = r"C:\Users\badUser"

        try:
            test_dm = DataManager(["Feat1", "Feat2"], bad_dir, 1, 10,
                                  "Feat1", TestDataManager.unit_column_map, pixel_as_field=True)
        except FileNotFoundError:
            pass
        except Exception:
            self.fail(
                "Did not throw a FileNotFound error for bad data file path.")
        else:
            self.fail(
                "Did not throw a FileNotFound error for bad data file path.")

    def test_not_csv(self):

        not_csv_path = os.path.join(
            os.getcwd(), "SparseClassifier", "test_not_csv")

        try:
            test_dm = DataManager(["Feat1", "Feat2"], not_csv_path, 1, 10,
                                  "Feat1", TestDataManager.unit_column_map, pixel_as_field=True)
        except NotCSVError:
            pass
        except Exception:
            self.fail(
                "Did not throw a FileNotFound error for bad data file path.")
        else:
            self.fail(
                "Did not throw a FileNotFound error for bad data file path.")

    def test_normal_setup(self):

        data_path = os.path.join(
            os.getcwd(), "SparseClassifier", "test_data_files")

        features = "Kansas,Nebraska".split(sep=",")

        test_dm = DataManager(features, data_path, 1, 10,
                              "Kansas", TestDataManager.unit_column_map, pixel_as_field=True)

        self.assertEqual("Kansas", test_dm.get_target_class())
        self.assertEqual(sorted("Kansas,Nebraska".split(sep=",")),
                         sorted(test_dm.get_source_classes()))
        self.assertEqual(1, test_dm.get_iteration())
        self.assertEqual(10, test_dm.get_num_features())

    def test_feature_generator(self):

        # Test the generator when no stop values is given
        for index, feature_name in enumerate(DataManager.feature_name_gen()):
            self.assertEqual(f"Feature{index}", feature_name)

            if index > 50:
                break

        # Now test with a stopping value
        feature_gen = DataManager.feature_name_gen(stop=10)
        for index, feature_name in enumerate(feature_gen):
            self.assertEqual(f"Feature{index}", feature_name)

        self.assertEqual(index, 10)

    def test_file_matching_regex(self):

        data_file_path = os.listdir(os.path.join(
            os.getcwd(), "SparseClassifier", "test_data_files"))

        test_patterns = []
        test_patterns.append("Kansas")
        test_patterns.append(r"sparse")
        test_patterns.append(r"feature[s]?" + "10")
        test_patterns.append(
            r"iter[atione]*[^a-zA-Z0-9]*" + "1")

        matched_patterns = DataManager.file_matching_regex(
            data_file_path, test_patterns)

        self.assertEqual(
            ["usa_Kansas_sparse_feature10_iter1_unit.csv"], matched_patterns)

    def test_frame_standardize(self):

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
        expected_df["PixelID"] = expected_df["FieldID"]

        features = "Kansas,Nebraska".split(sep=",")
        test_dm = DataManager(features, data_path, 1, 10,
                              "Kansas", TestDataManager.unit_column_map, pixel_as_field=True)

        try:
            pd.testing.assert_frame_equal(
                expected_df, test_dm.standardize_dataframe(raw_df),
                check_less_precise=True, check_names=True)
        except AssertionError:
            self.fail(
                "Expected and actual standardized dataframes do not have the."
                "same structure.")

    def test_training_setup(self):

        data_path = os.path.join(
            os.getcwd(), "SparseClassifier", "test_data_files")
        data_path_list = os.listdir(data_path)
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
        expected_df["PixelID"] = expected_df["FieldID"]

        features = "Kansas,Nebraska".split(sep=",")
        test_dm = DataManager(features, data_path, 1, 10,
                              "Kansas", TestDataManager.unit_column_map, pixel_as_field=True)
        actual_df = test_dm.create_training_dataframe()
        self.assertIsNotNone(getattr(test_dm, "_training_dataframe"))

        try:
            pd.testing.assert_frame_equal(
                expected_df, actual_df,
                check_less_precise=True, check_names=True)
        except AssertionError:
            self.fail(
                "Expected and actual training dataframes do not have the."
                "same structure.")

    def test_training_setup_from_file(self):

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
        expected_df["PixelID"] = expected_df["FieldID"]

        features = "Kansas,Nebraska".split(sep=",")
        test_dm = DataManager(features, data_path, 1, 10,
                              "Kansas", TestDataManager.unit_column_map,
                              training_files=[expected_file_path], pixel_as_field=True)
        actual_df = test_dm.create_training_dataframe()
        self.assertIsNotNone(getattr(test_dm, "_training_dataframe"))

        try:
            pd.testing.assert_frame_equal(
                expected_df, actual_df,
                check_less_precise=True, check_names=True)
        except AssertionError:
            self.fail(
                "Expected and actual training dataframes do not have the."
                "same structure.")

    def test_testing_setup(self):

        data_path = os.path.join(
            os.getcwd(), "SparseClassifier", "test_data_files")
        data_path_list = os.listdir(data_path)
        expected_file_path = os.path.join(
            data_path, "usa_Kansas_test_feature10_iter1_unit.csv")

        # Construct the expected dataframe
        expected_df = pd.read_csv(expected_file_path)
        raw_df = copy.deepcopy(expected_df)

        # Drop unnecessary columns
        expected_df = expected_df.drop(
            columns=["category", "details_name"])

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
        expected_df["PixelID"] = expected_df["FieldID"]

        features = "Kansas,Nebraska".split(sep=",")
        test_dm = DataManager(features, data_path, 1, 10,
                              "Kansas", TestDataManager.unit_column_map, pixel_as_field=True)
        actual_df = test_dm.create_testing_dataframe()
        self.assertIsNotNone(getattr(test_dm, "_testing_dataframe"))

        try:
            pd.testing.assert_frame_equal(
                expected_df, actual_df,
                check_less_precise=True, check_names=True)
        except AssertionError:
            self.fail(
                "Expected and actual testing dataframes do not have the."
                "same structure.")

    def test_population_setup(self):

        data_path = os.path.join(
            os.getcwd(), "SparseClassifier", "test_data_files")
        data_path_list = os.listdir(data_path)
        expected_file_path = os.path.join(
            data_path, "usa_Kansas_population_feature10_iter1_unit.csv")

        # Construct the expected dataframe
        expected_df = pd.read_csv(expected_file_path)
        raw_df = copy.deepcopy(expected_df)

        # Drop unnecessary columns
        expected_df = expected_df.drop(
            columns=["category", "details_name"])

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
        expected_df["PixelID"] = expected_df["FieldID"]

        features = "Kansas,Nebraska".split(sep=",")
        test_dm = DataManager(features, data_path, 1, 10,
                              "Kansas", TestDataManager.unit_column_map, pixel_as_field=True)
        actual_df = test_dm.create_population_dataframe()
        self.assertIsNotNone(getattr(test_dm, "_population_dataframe"))

        try:
            pd.testing.assert_frame_equal(
                expected_df, actual_df,
                check_less_precise=True, check_names=True)
        except AssertionError:
            self.fail(
                "Expected and actual testing dataframes do not have the."
                "same structure.")

    def test_pooled_setup(self):

        data_path = os.path.join(
            os.getcwd(), "SparseClassifier", "test_data_files")
        data_path_list = os.listdir(data_path)
        expected_kansas_path = os.path.join(
            data_path, "usa_Kansas_train_feature10_iter1_unit.csv")
        expected_nebraska_path = os.path.join(
            data_path, "usa_Nebraska_train_feature10_iter1_unit.csv")

        # Construct the expected dataframe
        expected_kansas_df = pd.read_csv(expected_kansas_path)
        expected_nebraska_df = pd.read_csv(expected_nebraska_path)
        raw_df = copy.deepcopy(expected_kansas_df)
        raw_df = copy.deepcopy(expected_nebraska_df)

        # Drop unnecessary columns
        expected_kansas_df = expected_kansas_df.drop(
            columns=["category", "details_name"])
        expected_nebraska_df = expected_nebraska_df.drop(
            columns=["category", "details_name"])

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
            "value": "Class"
        }

        expected_kansas_df = expected_kansas_df.rename(columns=renaming_scheme)
        expected_nebraska_df = expected_nebraska_df.rename(
            columns=renaming_scheme)
        expected_kansas_df["PixelID"] = expected_kansas_df["FieldID"]
        expected_nebraska_df["PixelID"] = expected_nebraska_df["FieldID"]

        expected_df = pd.concat(
            [expected_kansas_df, expected_nebraska_df], ignore_index=True)

        features = "Kansas,Nebraska".split(sep=",")
        test_dm = DataManager(features, data_path, 1, 10,
                              "Kansas", TestDataManager.unit_column_map, pixel_as_field=True)
        actual_df = test_dm.create_pooled_dataframe()
        self.assertIsNotNone(getattr(test_dm, "_pooled_dataframe"))

        try:
            pd.testing.assert_frame_equal(
                expected_df, actual_df,
                check_less_precise=True, check_names=True)
        except AssertionError:
            self.fail(
                "Expected and actual pooled dataframes do not have the."
                "same structure.")

    def test_pool_to_training(self):
        data_path = os.path.join(
            os.getcwd(), "SparseClassifier", "test_data_files")
        data_path_list = os.listdir(data_path)
        expected_kansas_path = os.path.join(
            data_path, "usa_Kansas_train_feature10_iter1_unit.csv")
        expected_nebraska_path = os.path.join(
            data_path, "usa_Nebraska_train_feature10_iter1_unit.csv")

        features = "Kansas,Nebraska".split(sep=",")
        test_dm = DataManager(features, data_path, 1, 10,
                              "Kansas", TestDataManager.unit_column_map,
                              training_files=expected_kansas_path,
                              pooled_files=expected_nebraska_path,
                              pixel_as_field=True)

        # Get the current field ids for the training and pooled dataframes
        training_ids = test_dm._training_dataframe['FieldID'].values
        pooled_ids = test_dm._pooled_dataframe['FieldID'].values

        test_dm.pool_to_training(20)

        training_ids = np.concatenate((training_ids, np.array([20])))
        pooled_ids = np.delete(pooled_ids, np.where(pooled_ids == 20))

        np.testing.assert_almost_equal(
            np.sort(test_dm._training_dataframe['FieldID'].values),
            np.sort(training_ids),
            decimal=5)

        np.testing.assert_almost_equal(
            np.sort(test_dm._pooled_dataframe['FieldID'].values),
            np.sort(pooled_ids),
            decimal=5)

        test_dm.pool_to_training(np.array([24, 44]))

        training_ids = np.concatenate((training_ids, np.array([24, 44])))
        pooled_ids = np.delete(pooled_ids, np.where(pooled_ids == 24))
        pooled_ids = np.delete(pooled_ids, np.where(pooled_ids == 44))

        np.testing.assert_almost_equal(
            np.sort(test_dm._training_dataframe['FieldID'].values),
            np.sort(training_ids),
            decimal=5)

        np.testing.assert_almost_equal(
            np.sort(test_dm._pooled_dataframe['FieldID'].values),
            np.sort(pooled_ids),
            decimal=5)


if __name__ == '__main__':
    unittest.main()
