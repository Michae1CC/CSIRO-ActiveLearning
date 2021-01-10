import glob
import itertools
import os
import platform
import sys
import re

import numpy as np
import pandas as pd

from ColEnum import ColEnum
from NotCSVError import NotCSVError
from ColumnMapError import ColumnMapError


class DataManager(object):
    """
    Reads in csv data in from a specified folder and formats it into a
    standardized DataFrame.
    """

    def __init__(self, data_classes: list, data_folder_path: str,
                 iteration: int,
                 num_features: int, target_class: str, column_map,
                 source_classes: list = None, training_files: list = None, pooled_files: list = None,
                 population_files: str = None, testing_files: str = None, pixel_as_field: bool = False):
        """
        Initialises an instance of the data management class.

        Parameters:
            data_classes:
                A list of names of all the classes found in the data.

            data_folder_path:
                A path to the folder where all the datafiles are kept.

            iteration:
                The iteration number that is to be loaded into the DataManager.

            num_features:
                The number of features to be used in the classification.

            target_class:
                The class that the model will train on to identify.

            pooled_files:
                A list of all the files that will be used to train the model on.

            population_files:
                A file that holds the population data.

            test_file:
                A file that holds the testing data.

            pixel_as_field:
                If a pixel id column does not exist in the input files, the
                DataManager will create it using the field id column.
        """

        # Check that the specified folder exists
        if not os.path.exists(data_folder_path):
            raise FileNotFoundError("Could not locate the specified folder: {folder_name}".format(
                folder_name=data_folder_path))

        self._source_classes = None

        # If the source classes have not been specified then use all the
        # available different classes
        if (source_classes is None) or (source_classes == []):
            self._source_classes = data_classes
        else:
            self._source_classes = source_classes

        # Keep a reference to our target class
        self._target_class = target_class

        self._column_map = column_map
        self._pixel_as_field = pixel_as_field
        self._data_folder_path = data_folder_path
        self._iteration = iteration
        self._num_features = num_features

        # Keep track of which files will be need to construct each dataframe
        self._training_files = training_files
        self._testing_files = testing_files
        self._population_files = population_files
        self._pooled_files = pooled_files

        # Keep a dictionary that maps between the unstandardized and standardized
        self._feature_map = None

        # A dataframe to hold all the field ids that have been added to
        # the training data set
        self._added_ids = np.array([])

        # List all the directories in the data folder
        self._data_files = os.listdir(data_folder_path)

        if not all(map((lambda filename: filename[-4:] == ".csv"), self._data_files)):
            # Get all the files with the incorrect file format
            bad_files = itertools.filterfalse((lambda filename: filename[-4:] == ".csv"),
                                              self._data_files)
            bad_files_str = ",\n".join(bad_files)
            raise NotCSVError("The following data files do not have the correct" +
                              " file extension (.csv):\n{bad_files_str}".format(
                                  bad_files_str=bad_files_str))

        # Set up each of the dataframes
        self.create_training_dataframe()
        self.create_testing_dataframe()
        self.create_population_dataframe()
        self.create_pooled_dataframe()

    def get_target_class(self):
        """
        Returns the target class.
        """
        return self._target_class

    def get_source_classes(self):
        """
        Returns the classes used for the source data.
        """
        return self._source_classes

    def get_iteration(self):
        """
        Returns which iteration is being used for the data in this DataManager.
        """
        return self._iteration

    def get_num_features(self):
        """
        Returns how many features are being used in the data in this DataManager.
        """
        return self._num_features

    @staticmethod
    def feature_name_gen(stop: int = None):
        """
        A generator that creates standardized feature names for the features
        in a new data set.

        Parameter:
            stop:
                Used to specify a upper bound for the features to be produced.

        Yields:
            Features names in ascending order.
        """

        counter = itertools.count()
        index = 0

        while (stop is None) or (index < stop):
            index = next(counter)
            yield f"Feature{index}"

    @staticmethod
    def file_matching_regex(data_folders: list, patterns: list):
        """
        Finds all the data files which match all the given regular expressions.

        Parameters:
            data_folders:
                A list of all the data file names (as strings).

            patterns:
                A list of all the regular expressions that the file names must
                contain.

        Return:
            A list of all the file names that match every regular expression.
        """

        matching_all = []

        for filename in data_folders:
            # Map each of the regular expressions to the filename
            # If the filename contains all of the searched expressions
            # append it to the matching_all list.
            if all(map(lambda pattern: re.findall(pattern, filename, re.IGNORECASE), patterns)):
                matching_all.append(filename)

        return matching_all

    def standardize_dataframe(self, raw_df: pd.DataFrame):
        """
        Standardizes a the given dataframe so that the raw dataframe uses a
        common set of column names.

        Parameters:
            raw_df:
                The raw DataFrame that needs to be standardized.
        """

        # Find all the columns that be omitted
        omitted_cols = list(filter((lambda col_name: self._column_map(
            col_name) == ColEnum.OMIT), raw_df.columns))

        raw_df = raw_df.drop(columns=omitted_cols)

        # Rename the Pixel id
        field_id = list(filter((lambda col_name: self._column_map(
            col_name) == ColEnum.FIELD), raw_df.columns))

        if len(field_id) != 1:
            raise ColumnMapError("Column map mapped incorrect number of"
                                 " field ids.")
        else:
            old_field_name = field_id[0]
            raw_df = raw_df.rename(columns={old_field_name: "FieldID"})

        # Rename the class column
        class_cols = list(filter((lambda col_name: self._column_map(
            col_name) == ColEnum.CLASS), raw_df.columns))

        if len(class_cols) != 1:
            raise ColumnMapError("Column map mapped incorrect number of"
                                 "class columns.")
        else:
            old_class_name = class_cols[0]
            raw_df = raw_df.rename(columns={old_class_name: "Class"})

        # Rename of add the pixel id
        pixel_id = list(filter((lambda col_name: self._column_map(
            col_name) == ColEnum.CLASS), raw_df.columns))

        if (len(pixel_id) == 0) and self._pixel_as_field:
            # We need to create a new pixel field that using the field ID
            raw_df["PixelID"] = raw_df["FieldID"]

        elif len(field_id) == 1:
            # We only need to rename the current pixel class
            old_pixel_name = pixel_id[0]
            raw_df = raw_df.rename(columns={old_pixel_name: "PixelID"})

        else:
            raise ColumnMapError("Column map mapped incorrect number of "
                                 "pixel columns.")

        # Standardize the feature names, first get all the current feature names
        current_feature_names = filter((lambda col_name: self._column_map(
            col_name) == ColEnum.FEATURE), raw_df.columns)

        if self._feature_map is None:
            self._feature_map = {
                old_feat_name: new_feat_name for old_feat_name, new_feat_name in
                zip(current_feature_names, DataManager.feature_name_gen())
            }

        # Rename the feature names using a standardized naming convention
        raw_df = raw_df.rename(columns=self._feature_map)

        return raw_df

    def construct_dataframe(self, dataframe_type: str, additional_patterns: list, can_be_none: bool = False):
        """
        Constructs a new Dataframe using data from the input data folder.

        Parameters:
            dataframe_type:
                The 'type' of data (in the context of active learning) 
                that will be stored in the new dataframe. The current
                dataframe_types that are supported are:
                    'training', 'testing', 'population', 'pooled'

            additional_patterns:
                If no filenames are given in the init method to construct
                one of these dataframes, regular expression will be used to
                search for appropriately named files. A list of additional
                patterns (on top of some default pattern) may to given to 
                the file search for better search refinement.

            can_be_none:
                If no files are given AND no appropriate files are found then
                the dataframe AND 'can_be_none' is set to True then the
                dataframe is set to None. Otherwise an error is thrown.
        """

        # Get the attribute which should be a single file containing all the
        # data for the dataframe or a list of files containing the data for
        # the dataframe

        dataframe_files = getattr(self, f"_{dataframe_type}_files")

        # Make sure that the dataframe_files is a list
        if isinstance(dataframe_files, str):
            dataframe_files = [dataframe_files]

        # Create a name for the dataframe attribute
        dataframe_name = f"_{dataframe_type}_dataframe"

        # Initialise the dataframe to None
        self.__dict__[dataframe_name] = None

        # If the dataframe files have not been set yet, using regular expressions
        # to try and locate the files within the data files
        if (dataframe_files is None) or (dataframe_files == []):
            file_patterns = []
            file_patterns.append(r"feature[s]?" + str(self._num_features))
            file_patterns.append(
                r"iter[atione]*[^a-zA-Z0-9]*" + str(self._iteration))
            file_patterns.extend(additional_patterns)

            dataframe_files = DataManager.file_matching_regex(
                self._data_files, file_patterns)
        else:
            # Check that the provided files exist
            for filename in dataframe_files:

                full_file_path = os.path.join(self._data_folder_path, filename)

                if not os.path.exists(full_file_path):
                    raise FileNotFoundError("Could not locate the specified "
                                            "folder: {file_path}".format(
                                                file_path=full_file_path))

        # If the dataframe cannot be None check that we have found at least
        # found one file to construct the dataframe
        if (len(dataframe_files) == 0) and can_be_none:
            return None
        elif len(dataframe_files) == 0:
            raise FileNotFoundError(
                f"No suitable sparse data file could be found to create {dataframe_name}.")

        # Get the full file paths for each of the dataframe files
        target_file_paths = list(map(lambda filename: os.path.join(
            self._data_folder_path, filename), dataframe_files))

        # Merge all the data together
        if len(target_file_paths) > 1:
            new_dataframe = pd.concat(
                map(pd.read_csv, target_file_paths), ignore_index=True)
        else:
            new_dataframe = pd.read_csv(target_file_paths[0])

        # Standardize the column names of the new dataframe
        new_dataframe = self.standardize_dataframe(new_dataframe)
        setattr(self, dataframe_name, new_dataframe)

        return new_dataframe

    def create_training_dataframe(self):
        """
        Creates a dataframe for the training data.
        """

        additional_patterns = []
        additional_patterns.append(f"{self._target_class}")
        additional_patterns.append(r"sparse")

        new_dataframe = self.construct_dataframe(
            "training", additional_patterns, can_be_none=False)

        return new_dataframe

    def create_testing_dataframe(self):
        """
        Creates a dataframe for the testing data.
        """

        additional_patterns = []
        additional_patterns.append(
            "(" + "|".join(map(lambda single_cls: single_cls, self._target_class)) + ")")
        additional_patterns.append(r"test[ings]{0,4}")

        new_dataframe = self.construct_dataframe(
            "testing", additional_patterns)

        return new_dataframe

    def create_population_dataframe(self):
        """
        Creates a dataframe for the population data.
        """

        additional_patterns = []
        additional_patterns.append(
            "(" + "|".join(map(lambda single_cls: single_cls, self._source_classes)) + ")")
        additional_patterns.append(r"pop[ulation]{0,7}")

        # TODO: Can population be none?
        new_dataframe = self.construct_dataframe(
            "population", additional_patterns, can_be_none=True)

        return new_dataframe

    def create_pooled_dataframe(self):
        """
        Creates a dataframe for the pooled data.
        """

        additional_patterns = []
        additional_patterns.append(
            "(" + "|".join(map(lambda single_cls: single_cls, self._target_class)) + ")")
        # TODO: Should pooled come from population?
        additional_patterns.append(r"train[ing]{0,3}")

        new_dataframe = self.construct_dataframe(
            "pooled", additional_patterns)

        return new_dataframe

    def pool_to_training(self, field_id):
        """
        Moves entries the input field id/s from the pooled data to the training
        data.

        Parameters:
            field_id:
                Entries that have a field id matching the input field id will
                be transferred from the pooled data to the training data.
        """

        if isinstance(field_id, int):
            field_id = np.array([field_id])

        # Extract all the training data from the pooled dataframe
        removed_values = self._pooled_dataframe[self._pooled_dataframe['FieldID'].isin(
            field_id)]

        # Remove the pooled
        self._pooled_dataframe.drop(removed_values.index, inplace=True)

        self._added_ids = np.concatenate(
            (self._added_ids, removed_values.index))

        # Add the new values to the training set
        self._training_dataframe = pd.concat(
            [self._training_dataframe, removed_values], ignore_index=True)

        return
