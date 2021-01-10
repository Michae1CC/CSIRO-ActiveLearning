import os
import platform
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, make_scorer

from AbstractClassifier import AbstractClassifier
from RSCVClassifier import RSCVClassifier

if not sys.warnoptions:
    warnings.simplefilter("ignore")

if sys.platform.startswith('win32'):
    ROOT_DIR = r"F:"
elif sys.platform.startswith('linux'):
    ROOT_DIR = r"/OSM/CBR/AF_DIGI_SAT/work"


class NaiveLearner(object):
    """
    Trains a classifier using data from an oracle by randomly taking points
    from the oracle and using them to train the classifier.
    """

    def __init__(self, source_path, scoring_methods: dict, iteration: int,
                 cls_constructor: type = RSCVClassifier,
                 oracle_pts_used: int = 2000, num_cores=4, oracle_source: str = "t2t",
                 data_output_path: str = os.getcwd()):
        """
        Initializes an environment for a naive learner.

        Parameters:
            source_path:
                A path leading to the data files used for testing.

            scoring_methods:
                A dictionary whose keys are a human given/chosen name for the
                scoring method and whose values are a tuple where the first
                entry is the scoring method and the second entry is a dictionary
                of key word arguments. Example:
                {
                    'f1_macro': (f1_score, {"average": "macro"}),
                    'accuracy_score': (accuracy_score, dict())
                }

            iteration:
                The iteration number to be completed.

            cls_constructor:
                A constructor used to build a classifier wrapper that inherits
                from AbstractClassifier.

            oracle_pts_used:
                The total number of points that will be taken from the oracle
                during the training process.

            num_cores:
                The number of cores used to train the classifier.

            oracle_source:
                A string specifying the origin for the data in the oracle file.
        """

        # Hold a reference to the classifier constructor
        self._cls_constructor = cls_constructor

        # Save the source path
        self._source_path = source_path if isinstance(
            source_path, str) else os.path.join(*source_path)

        self._output_path = data_output_path

        # Check that the output path exist
        if not os.path.exists(self._output_path):
            raise OSError(
                "Output folder path does not exist:\n" + self._output_path)

        # Hold a reference to the scoring methods
        self._scoring_methods = scoring_methods

        # Hold a reference to the scoring conditionals and the number of
        # point that will be taken from the oracle when training and the
        # the number of iterations
        self._oracle_pts_used = oracle_pts_used
        self._iteration = iteration
        self._num_cores = num_cores

        self._oracle_source = None
        self._oracle_state = None

        if oracle_source.startswith("s2t"):
            self._oracle_source, self._oracle_state = oracle_source.split(
                "_", maxsplit=1)
        else:
            self._oracle_source = oracle_source

    def commence_work(self):

        test_df = pd.read_csv(os.path.join(
            self._source_path, f"test_file_{self._iteration}.csv"))
        train_df = pd.read_csv(os.path.join(
            self._source_path, f"train_file_{self._iteration}.csv"))

        oracle_df = None

        if self._oracle_source in ["t2t", "target_to_target", "target_2_target"]:

            oracle_df = pd.read_csv(os.path.join(
                self._source_path, f"target_to_target_oracle_file_{self._iteration}.csv"))

        # A source to target oracle is being requested instead
        else:

            # Check which state the data is being sourced from
            if self._oracle_state == "all":
                oracle_df = pd.read_csv(os.path.join(
                    self._source_path, "source_to_target_oracle.csv"))

            else:
                oracle_df = pd.read_csv(os.path.join(
                    self._source_path, f"source_to_target_{self._oracle_state}_oracle.csv"))

            # Shuffle around the data of the oracle file, otherwise we are going
            # to sample the same data across different iterations
            oracle_df = oracle_df.sample(frac=1)

        if oracle_df.shape[0] < self._oracle_pts_used:
            print("WARN: Oracle contains less points than requested.")
            self._oracle_pts_used = oracle_df.shape[0]

        oracle_source_name = None

        if self._oracle_state is None:
            oracle_source_name = self._oracle_source
        else:
            oracle_source_name = f"{self._oracle_source}_{self._oracle_state}"

        self.complete_iteration(oracle_source_name,
                                self._iteration, test_df, train_df, oracle_df)

        return

    def complete_iteration(self, source_type: str, iteration: int,
                           test_df: pd.DataFrame, train_df: pd.DataFrame, oracle_df: pd.DataFrame):
        """
        Completes all the scoring for a single iteration.

        Parameters:
            score_datastructure:
                The dictionary datastructure in which all the scores are kept.

            iteration:
                The current file iteration in use.

            test_df:
                A dataframe holding data to create score values.

            train_df:
                A dataframe that will store the initial sparse training set.

            oracle_df:
                A dataframe that will hold data will slowly added to the 
                training set.
        """

        print(
            f"Filename convention: scorename_{source_type}_iter_{iteration}.csv", flush=True)

        # Create a dictionary that will store all the scores for each of the
        # iterations
        score_dict = dict()

        for scoring_name in self._scoring_methods.keys():
            score_dict[scoring_name] = np.zeros(self._oracle_pts_used,
                                                dtype=float)

        test_df_y = test_df["value"].values
        test_df_X = test_df.drop(
            columns=["details_id", "details_name", "value"]).values

        train_df_y = train_df["value"].values
        train_df_X = train_df.drop(
            columns=["details_id", "details_name", "value"]).values

        oracle_df_y = oracle_df["value"].values
        oracle_df_X = oracle_df.drop(
            columns=["details_id", "details_name", "value"]).values

        print("Iterations completed:\n")

        for data_used in range(self._oracle_pts_used):

            print(f"{data_used},", end="", flush=True)

            # Take a the first row from the oracle
            oracle_inst_row_X = oracle_df_X[0]
            oracle_inst_row_y = oracle_df_y[0]

            # Remove these points from the oracle
            oracle_df_X = np.delete(oracle_df_X, 0, axis=0)
            oracle_df_y = oracle_df_y[1:]

            # Add this new point to the training set
            train_df_X = np.vstack((train_df_X, oracle_inst_row_X))
            train_df_y = np.append(train_df_y, [oracle_inst_row_y], axis=0)

            # Create a new classifier
            iter_cls = self._cls_constructor()

            # Prime the classifier on the sparse data
            iter_cls.prime_classifier(
                train_df_X, train_df_y, n_jobs=self._num_cores)

            y_pred = iter_cls.get_prediction(test_df_X)

            # Score the classifier once trained on the new data
            for scoring_name, score_tuple in self._scoring_methods.items():

                score_method, score_kwargs = score_tuple
                score_val = score_method(test_df_y, y_pred, **score_kwargs)

                score_dict[scoring_name][data_used] = score_val

        # Save the resulting scores
        for scoring_name in self._scoring_methods.keys():

            filename = os.path.join(
                self._output_path, f"{scoring_name}_{source_type}_iter_{iteration}.csv")
            np.savetxt(filename, score_dict[scoring_name], delimiter=",")

            print(f"\nSaved {scoring_name} to " + filename, flush=True)

        return


def test_learner():

    data_path = os.path.join(os.getcwd(), "SparseClassifier", "sample_output")
    mock_output_path = os.path.join(
        os.getcwd(), "SparseClassifier", "mock_out")

    scoring_methods = {'f1_macro': (f1_score, {"average": "macro"}),
                       'accuracy_score': (accuracy_score, dict())}

    new_nv = NaiveLearner(data_path, scoring_methods, 0, oracle_pts_used=10,
                          num_cores=1, data_output_path=mock_output_path)
    new_nv.commence_work()


def drive_learner(iteration, oracle_source, data_folder):

    # The data folder will take the form "michael_naive_statename"
    state_used = data_folder.split("_", maxsplit=2)[-1]

    data_path = os.path.join(
        ROOT_DIR, "crop_mapping_sparse", "data", "usa", "classes_3", data_folder)
    output_path = os.path.join(os.getcwd(), "naive_learner_output", state_used)

    scoring_methods = {'f1_macro': (f1_score, {"average": "macro"}),
                       'accuracy_score': (accuracy_score, dict())}
    new_nv = NaiveLearner(data_path, scoring_methods, iteration,
                          oracle_pts_used=2000, num_cores=4, oracle_source=oracle_source,
                          data_output_path=output_path)

    print(
        f"Commencing oracle_source:{oracle_source}, iteration:{iteration}, data_folder:{data_folder}", flush=True)

    new_nv.commence_work()


def main():

    if (len(sys.argv) < 3) or (not sys.argv[1].isdigit()):
        USAGE = """USAGE:\n\tnaive_learner_driver.py [iter] [oracle source] [optional: data folder]"""
        print(USAGE)
        return

    # Have a default data file
    data_folder = "michael_naive_nebraska"

    if len(sys.argv) == 4:
        data_folder = sys.argv[3]

    iteration = int(sys.argv[1])
    oracle_source = sys.argv[2]

    drive_learner(iteration, oracle_source, data_folder)


if __name__ == '__main__':
    main()
