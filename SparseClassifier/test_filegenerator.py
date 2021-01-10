import os
import sys
import copy
import shutil
import unittest

import numpy as np
import pandas as pd

from FileGenerator import FileGenerator
from ColEnum import ColEnum
from NotCSVError import NotCSVError
from ColumnMapError import ColumnMapError
from DataManager import DataManager


class TestDataManager(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls._output_folder_path = os.path.join(os.getcwd(), "SparseClassifier", "test_output")
        
        # Create a new folder to send all the generated files
        os.mkdir(cls._output_folder_path)

        source_path = os.path.join(os.getcwd(), "SparseClassifier", "test_data_files", "mock_origin.csv")

        cls._file_gen = FileGenerator("Nebraska", source_path=source_path, output_path=cls._output_folder_path,
            file_ratio = {"test": 10, "train": 5, "pseudo_population": 20})

        return

    def test_source_setup(self):

        cls = self.__class__

        # Create the source file
        cls._file_gen.create_source_to_target()

        # Retrieve the source file
        self.assertTrue(os.path.exists(os.path.join(cls._output_folder_path, "source_to_target_oracle.csv")),
            msg="Source to Target oracle file does not exist.")
        actual_source_df = pd.read_csv(os.path.join(cls._output_folder_path, "source_to_target_oracle.csv"))

        # Retrieve the expected file
        expected_source_df = pd.read_csv(os.path.join(
            os.getcwd(), "SparseClassifier", "test_data_files", "expected_source.csv"))
        expected_source_df.drop(columns=["category", "cropland", "coi"], inplace=True)

        pd.testing.assert_frame_equal(
                expected_source_df, actual_source_df, check_less_precise=True, check_names=True)

    def test_target_files(self):

        cls = self.__class__

        cls._file_gen.create_target_files(iterations=2)

        self.assertTrue(os.path.exists(os.path.join(cls._output_folder_path, "test_file_0.csv")))
        self.assertTrue(os.path.exists(os.path.join(cls._output_folder_path, "test_file_1.csv")))
        # There should only be 2 test files
        self.assertFalse(os.path.exists(os.path.join(cls._output_folder_path, "test_file_2.csv")))

        self.assertTrue(os.path.exists(os.path.join(cls._output_folder_path, "train_file_0.csv")))
        self.assertTrue(os.path.exists(os.path.join(cls._output_folder_path, "train_file_1.csv")))
        self.assertFalse(os.path.exists(os.path.join(cls._output_folder_path, "train_file_2.csv")))

        self.assertTrue(os.path.exists(os.path.join(cls._output_folder_path, "target_to_target_oracle_file_0.csv")))
        self.assertTrue(os.path.exists(os.path.join(cls._output_folder_path, "target_to_target_oracle_file_1.csv")))
        self.assertFalse(os.path.exists(os.path.join(cls._output_folder_path, "target_to_target_oracle_file_2.csv")))

        self.assertTrue(os.path.exists(os.path.join(cls._output_folder_path, "pseudo_population_file_0.csv")))
        self.assertTrue(os.path.exists(os.path.join(cls._output_folder_path, "pseudo_population_file_1.csv")))
        self.assertFalse(os.path.exists(os.path.join(cls._output_folder_path, "pseudo_population_file_2.csv")))

        # Check that the files contain the correct number of rows
        test_0_actual_df = pd.read_csv(os.path.join(cls._output_folder_path, "test_file_0.csv"))
        test_0_rows, _ = test_0_actual_df.values.shape

        # We expect 8 rows due to the ratios given in the instantiation
        self.assertEqual(10, test_0_rows)
        del test_0_actual_df

        # Check that the training file has the correct number of rows
        train_0_actual_df = pd.read_csv(os.path.join(cls._output_folder_path, "train_file_0.csv"))
        train_0_rows, _ = train_0_actual_df.values.shape

        self.assertEqual(5, train_0_rows)
        del train_0_actual_df

        # Check that the oracle file has the correct number of rows
        t2t_oracle_0_actual_df = pd.read_csv(os.path.join(cls._output_folder_path, "target_to_target_oracle_file_0.csv"))
        t2t_oracle_0_rows, _ = t2t_oracle_0_actual_df.values.shape

        self.assertEqual(35, t2t_oracle_0_rows)
        del t2t_oracle_0_actual_df

        # Finally check that the pesudo population contains the correct
        # of rows
        pop_0_actual_df = pd.read_csv(os.path.join(cls._output_folder_path, "pseudo_population_file_0.csv"))
        pop_0_rows, _ = pop_0_actual_df.values.shape

        self.assertEqual(20, pop_0_rows)
        del pop_0_actual_df

    @classmethod
    def tearDownClass(cls):

        if os.path.exists(cls._output_folder_path):
            shutil.rmtree(cls._output_folder_path)
        return

if __name__ == '__main__':
    unittest.main()