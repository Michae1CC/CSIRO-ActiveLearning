import os
import platform
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, make_scorer

from AbstractClassifier import AbstractClassifier
from RSCVClassifier import RSCVClassifier
from NaiveLearner import NaiveLearner

if not sys.warnoptions:
    warnings.simplefilter("ignore")

if sys.platform.startswith('win32'):
    ROOT_DIR = r"F:"
elif sys.platform.startswith('linux'):
    ROOT_DIR = r"/OSM/CBR/AF_DIGI_SAT/work"


class FileGenerator(object):
    """
    This class is used for creating the source, test, training and
    pseudo population files to simulate a sparse data scenario.
    """

    def __init__(self, target_domain: str, source_path: tuple =
                 (ROOT_DIR, "crop_mapping_sparse", "data", "usa", "s2_images",
                  "US_2018_training_hls_harmonic_regression.csv"),
                 wanted_crops=["Soybeans", "Corn", "Alfalfa"],
                 output_path: str = (os.getcwd(), "file_gen_out"), redundant_columns: list = None, file_ratio: dict = None,
                 seed: int = None, value_fields=["xbl_intercept", "xbl_cos_t1", "xbl_sin_t1", "xbl_cos_t2",
                                                 "xbl_sin_t2", "x2gr_intercept", "x2gr_cos_t1", "x2gr_sin_t1",
                                                 "x2gr_cos_t2", "x2gr_sin_t2", "x3rd_intercept", "x3rd_cos_t1",
                                                 "x3rd_sin_t1", "x3rd_cos_t2", "x3rd_sin_t2", "x6re_intercept",
                                                 "x6re_cos_t1", "x6re_sin_t1", "x6re_cos_t2", "x6re_sin_t2",
                                                 "x4ni_intercept", "x4ni_cos_t1", "x4ni_sin_t1", "x4ni_cos_t2",
                                                 "x4ni_sin_t2", "x5sw_intercept", "x5sw_cos_t1", "x5sw_sin_t1",
                                                 "x5sw_cos_t2", "x5sw_sin_t2", "gcvi_intercept", "gcvi_cos_t1",
                                                 "gcvi_sin_t1", "gcvi_cos_t2"]):
        """
        Initializes a file generator class.

        Parameters:
            target_domain:
                The name (as a string) of our target domain. For example:
                'Nebraska'.

            source_path:
                A list of folders that form a file path to the file were
                all of the data is kept.

            output_path:
                A file path to save all of the file produced from the file
                generator.

            redundant_columns:
                A list of redundant columns to remove within the data file.

            file_ratio:
                A dictionary specifying what how many rows each file should
                receive.

            source_to_target:
                If True the source file will be constructed from data outside 
                of the target domain. Otherwise the source file will only contain
                data from the target domain.

            seed:
                A seed that can be set to control the randomness of file
                generation.
        """

        # Save our target domain
        self._target_domain = target_domain

        self._value_fields = value_fields

        # Create the source path as a string
        self._source_path = os.path.join(
            *source_path) if isinstance(source_path, tuple) else source_path

        # Create the output path as a string
        self._output_path = os.path.join(
            *output_path) if isinstance(output_path, tuple) else output_path

        # Check the input and output paths exist
        if not os.path.exists(self._source_path):
            raise OSError(
                "Source path does not exist:\n" + self._source_path)

        if not os.path.exists(self._output_path):
            raise OSError(
                "Output path does not exist:\n" + self._source_path)

        # Load the source file as a data frame
        self._source_df = pd.read_csv(self._source_path)

        # Save the numpy seed
        self._seed = seed

        # Load the source file as a data frame
        self._source_df = self._source_df[self._source_df["category"].isin(
            wanted_crops)]

        # Set a default list for the redundant_columns
        if redundant_columns is None:
            redundant_columns = ["category", "cropland", "coi"]

        # Use a default file ratio if none is specified
        self._file_ratio = {"test": 5000, "train": 20,
                            "pseudo_population": 10000} if file_ratio is None else file_ratio

        # Drop the redundant columns of the csv file
        self._source_df.drop(columns=redundant_columns, inplace=True)

        # Remove all the null values from the source dataframe
        self.remove_not_null()

    def remove_not_null(self):
        """
        Removes all the null values from the original source file.
        """

        for value_field in self._value_fields:
            self._source_df = self._source_df[pd.notnull(
                self._source_df[value_field])]
            self._source_df = self._source_df[pd.notna(
                self._source_df[value_field])]

        field_value_types = {
            field_val: "float32" for field_val in self._value_fields}
        self._source_df.astype(field_value_types)

    def create_files(self, iterations=30):
        """
        Creates all the files required for the project
        """

        self.create_target_files(iterations=iterations)
        self.create_source_to_target()

        return

    def create_source_to_target(self):
        """
        Creates a single oracle file which contains all the information from the
        original file except for points that are part of the target domain.
        """

        print("Creating source_to_target_oracle", flush=True)

        source_df = self._source_df[self._source_df["details_name"]
                                    != self._target_domain]

        source_file_path = os.path.join(
            self._output_path, "source_to_target_oracle.csv")

        source_df.to_csv(path_or_buf=source_file_path, index=False)

        del source_df

        return

    def create_target_files(self, iterations=30):
        """
        Creates all the files that contain target domain information,
        including the test, train and pseudo population files.
        """

        # Extract all the pixels that belong to the target domain
        target_domain_df = self._source_df[self._source_df["details_name"]
                                           == self._target_domain]

        # Make sure that there are enough rows in the dataframe that we can
        # safely index using the provided file ratio. First find the total
        # amount of indices that the dictionary assumes.
        dict_rows_required = sum(self._file_ratio.values())

        if dict_rows_required > target_domain_df.values.shape[0]:
            raise IndexError(f"The ratio dictionary assumes the existence of {dict_rows_required} "
                             f"rows but only {target_domain_df.values.shape[0]} rows "
                             "were present from the target domain.")

        # Extract the details id column (this uniquely identifies one row) within
        # the dataframe
        detail_ids = target_domain_df["details_id"].values

        for index in range(iterations):
            print(f"Creating files for iter {index}", flush=True)
            self.generate_target_suit(index, detail_ids, target_domain_df)

        return

    def create_source_to_target_state(self, state_name):

        print(f"Creating source_to_target_{state_name}_oracle", flush=True)

        source_df = self._source_df[self._source_df["details_name"] == state_name]

        source_file_path = os.path.join(
            self._output_path, f"source_to_target_{state_name}_oracle.csv".lower().replace(" ", "_"))

        source_df.to_csv(path_or_buf=source_file_path, index=False)

        del source_df

        return

    def generate_target_suit(self, iteration_num, detail_ids, target_domain_df):

        # Set the seed
        np.random.seed(seed=self._seed)

        # Shuffle the details_ids
        np.random.shuffle(detail_ids)

        # Select the correct number of rows to put in each file using the
        # details id as a key
        test_ids = detail_ids[:self._file_ratio["test"]]
        train_ids = detail_ids[self._file_ratio["test"]:self._file_ratio["test"] + self._file_ratio["train"]]
        t2t_oracle_ids = detail_ids[self._file_ratio["test"] +
                                    self._file_ratio["train"]:]
        # The population ids are a random sample of the t2t oracle ids
        pop_ids = np.random.choice(
            t2t_oracle_ids, size=self._file_ratio["pseudo_population"], replace=False)

        # Extract the detail ids for the test file and save the dataframe
        # to the output directory
        test_df = self._source_df[self._source_df["details_id"].isin(test_ids)]
        test_file_path = os.path.join(
            self._output_path, f"test_file_{iteration_num}.csv")
        test_df.to_csv(path_or_buf=test_file_path, index=False)
        del test_df

        train_df = self._source_df[self._source_df["details_id"].isin(
            train_ids)]
        train_file_path = os.path.join(
            self._output_path, f"train_file_{iteration_num}.csv")
        train_df.to_csv(path_or_buf=train_file_path, index=False)
        del train_df

        t2t_oracle_df = self._source_df[self._source_df["details_id"].isin(
            t2t_oracle_ids)]
        t2t_oracle_file_path = os.path.join(
            self._output_path, f"target_to_target_oracle_file_{iteration_num}.csv")
        t2t_oracle_df.to_csv(path_or_buf=t2t_oracle_file_path, index=False)
        del t2t_oracle_df

        pop_df = self._source_df[self._source_df["details_id"].isin(pop_ids)]
        pop_file_path = os.path.join(
            self._output_path, f"pseudo_population_file_{iteration_num}.csv")
        pop_df.to_csv(path_or_buf=pop_file_path, index=False)
        del pop_df

        return


if __name__ == '__main__':

    # North Dakota, Kansas, South Dakota, Nebraska

    output_path = os.path.join(
        ROOT_DIR, "crop_mapping_sparse", "data", "usa", "classes_3")

    nebraska_dict = {"test": 3500, "train": 20, "pseudo_population": 6000}
    kansas_dict = {"test": 1850, "train": 20, "pseudo_population": 3000}
    north_dakota_dict = {"test": 1850, "train": 20, "pseudo_population": 3000}
    south_dakota_dict = {"test": 2500, "train": 20, "pseudo_population": 4500}
    # NOTE: change back ratio!
    file_gen = FileGenerator("South Dakota", output_path=output_path,
                             file_ratio=south_dakota_dict)

    # file_gen.create_files(iterations=30)
    file_gen.create_source_to_target_state("Nebraska")
    file_gen.create_source_to_target_state("North Dakota")
    file_gen.create_source_to_target_state("South Dakota")
    file_gen.create_source_to_target_state("Kansas")
