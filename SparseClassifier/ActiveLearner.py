#!/usr/bin/env python3

__author__ = 'Michael Ciccotosto-Camp'
__version__ = ''

import os
import platform
import sys
import random
import warnings
import argparse

import numpy as np
import pandas as pd
from copy import deepcopy
from random import sample
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest

from AbstractClassifier import AbstractClassifier
from RSCVClassifier import RSCVClassifier

if sys.platform.startswith('win32'):
    ROOT_DIR = r"F:"
elif sys.platform.startswith('linux'):
    ROOT_DIR = r"/OSM/CBR/AF_DIGI_SAT/work"

if not sys.warnoptions:
    warnings.simplefilter("ignore")

STANDARD_PIXELS = 750

class ActiveLearner(object):
    """
    Trains a classifier using data from an oracle by randomly taking source_vectors
    from the oracle and using them to train the classifier.
    """

    def __init__(self, source_path, scoring_methods: dict, iteration: int,
                 cls_constructor: type = RSCVClassifier,
                 distance_measure=None, uncertainty_measure=None, k_best: int = None,
                 oracle_pts_used: int = 2000, num_cores=4, oracle_source: str = "t2t",
                 pick_top_n: int = 20, pick_cv_rounds: int = 20, seed_val: int = None,
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

            distance_measure:
                A function used to measure the distance between vectors in the
                feature space. By default the L2-norm is used.

            uncertainty_measure:
                A function used to measure the uncertainty of a prediction. By
                default the erp uncertainty is used.

            k_best:
                A parameter to specify that only the k most important features
                in the feature space are to be used when computing distances
                between vectors in the feature space.

            oracle_pts_used:
                The total number of source_vectors that will be taken from the oracle
                during the training process.

            num_cores:
                The number of cores used to train the classifier.

            oracle_source:
                A string specifying the origin for the data in the oracle file.

            pick_top_n:
                Used to sample from the n most uncertain points in the target
                domain which will be used to select a point from the source
                domain.

            pick_cv_rounds:
                Used to specify how many rounds of cross validation
                will be used in the out-of-bag scoring.
        """

        # Hold a reference to the classifier constructor
        self._cls_constructor = cls_constructor

        # Save the source path
        self._source_path = source_path if isinstance(
            source_path, str) else os.path.join(*source_path)

        self._output_path = data_output_path

        # Check that the output path exists
        if not os.path.exists(self._output_path):
            raise OSError(
                f"Output folder path does not exist.\n{self._output_path}")

        # Check that the source path exists
        if not os.path.exists(self._source_path):
            raise OSError(f"Source path does not exist.\n{self._source_path}")

        # Hold a reference to the scoring methods
        self._scoring_methods = scoring_methods

        self._oracle_source = None
        self._oracle_state = None

        if oracle_source.startswith("s2t"):
            self._oracle_source, self._oracle_state = oracle_source.split(
                "_", maxsplit=1)
        else:
            self._oracle_source = oracle_source

        self._distance_measure = ActiveLearner.L2_distance if distance_measure is None \
            else distance_measure
        self._uncertainty_measure = ActiveLearner.erp_confidence if uncertainty_measure is None \
            else uncertainty_measure
        self._k_best = k_best

        # Hold a reference to the scoring conditionals and the number of
        # point that will be taken from the oracle when training and the
        # the number of iterations
        self._oracle_pts_used = oracle_pts_used
        self._iteration = iteration
        self._num_cores = num_cores
        self._pick_top_n = pick_top_n
        self._pick_cv_rounds = pick_cv_rounds

        # Lists to record the max uncertainty field and uncertainty value
        self._pop_uncertainty_ids = [-1]
        self._pop_uncertainty_vals = [-1]

        self._oracle_chosen_ids = [-1]
        self._oracle_chosen_class = [-1]
        self._oracle_chosen_states = ["NA"]

        self._oob_splits = []
        self._oob_scores = [-1]

        random.seed(seed_val)
        np.random.seed(seed_val)

    @staticmethod
    def edi_value(pred_vals: np.ndarray) -> np.array:
        """
        Gets the edi value values using classification prediction
        probabilites. See Equation 12 of of:
            An information-based criterion to measure pixel-level thematic
            uncertainty in land cover classifications. (DOI 10.1007/s00477-016-1310-y)

        Return:
            Returns a vector of edi values for each prediction.
        """

        max_prob = np.max(pred_vals, axis=1)
        max_prob_index = np.argmax(pred_vals, axis=1)

        rows, cols = pred_vals.shape
        max_prob_index = np.arange(rows) * cols + max_prob_index
        deleted_max = np.delete(
            pred_vals.ravel(), max_prob_index, axis=0).reshape(rows, -1)

        # Set all the 0 values to 1's to prevent nan on log. This would have
        # been set to 0 regardless after multiplying by original matrix
        deleted_max_ones = deepcopy(deleted_max)
        deleted_max_ones[deleted_max_ones == 0] = 1

        intmd_prod = deleted_max * np.log(deleted_max_ones)
        edi = np.log(max_prob) - (1 / (1 - max_prob)) * \
            np.nansum(intmd_prod, axis=1)

        return edi

    @staticmethod
    def erp_confidence(pred_prob: np.ndarray) -> np.array:
        """
        Returns the erp confidence values for a given prediction vector.
        See Equation 22 of of:
            An information-based criterion to measure pixel-level thematic
            confidence in land cover classifications. (DOI 10.1007/s00477-016-1310-y)
        """

        exp_edi = np.exp(ActiveLearner.edi_value(pred_prob))
        erp = exp_edi / (exp_edi + (pred_prob.shape[1] - 1))
        return erp

    @staticmethod
    def erp_uncertainty(pred_prob: np.ndarray) -> np.array:
        """
        Returns an uncertainty measure based on the erp confidence.
        """

        return 1 - ActiveLearner.erp_confidence(pred_prob)

    @staticmethod
    def entropy_uncertainty(pred_prob: np.ndarray) -> np.array:
        """
        Return the information entropy for a given prediction vector.
        See Equation 3 of:
            An information-based criterion to measure pixel-level thematic
            confidence in land cover classifications. (DOI 10.1007/s00477-016-1310-y)
        """

        log_values = np.log(pred_prob)
        entropy = - np.sum(pred_prob * log_values, axis=1)

        return entropy

    @staticmethod
    def vector_dim_reduction(target_vector, source_vectors, rows_used):
        """
        Reduces the dimensionality of the target and source vectors based
        on the rows_used vector.

        Parameters:
            target_vector:
                A 1D array being the most 'uncertain' vector from the pseudo
                population.

            source_vectors:
                A 2D vector containing each of the source vectors as rows.

            rows_used:
                An list/array of indices that will be extracted from the
                target and source vectors.
        """

        remove_rows = [x for x in range(
            len(target_vector)) if x not in rows_used]

        target_vector = np.delete(target_vector, remove_rows, axis=0)
        source_vectors = np.delete(source_vectors, remove_rows, axis=1)

        return target_vector, source_vectors

    @staticmethod
    def cos_distance(target_vector, source_vectors, rows_used=None):
        """
        Finds the vector with the cosine distance in the source
        domain using the most uncertain vector from the target domain.

        Parameters:
            target_vector:
                A 1D array being the most 'uncertain' vector from the pseudo
                population.

            source_vectors:
                A 2D vector containing each of the source vectors as rows.

            rows_used:
                An list/array of indices that will be used to calculate the
                distance measure.
        """

        if rows_used is not None:
            target_vector, source_vectors = ActiveLearner.vector_dim_reduction(
                target_vector, source_vectors, rows_used)

        # Reshape the target vector so that its dimensions are
        # compatible with the cosine_similarity function
        target_vector = target_vector.reshape(1, -1)

        dist = np.abs(cosine_similarity(source_vectors, target_vector))

        return dist.argmax()

    @staticmethod
    def L2_distance(target_vector, source_vectors, rows_used=None):
        """
        Finds the vector with the smallest Euclidean distance in the source
        domain using the most uncertain vector from the target domain.

        Parameters:
            target_vector:
                A 1D array being the most 'uncertain' vector from the pseudo
                population.

            source_vectors:
                A 2D vector containing each of the source vectors as rows.

            rows_used:
                An list/array of indices that will be used to calculate the
                distance measure.
        """

        if rows_used is not None:
            target_vector, source_vectors = ActiveLearner.vector_dim_reduction(
                target_vector, source_vectors, rows_used)

        dist = np.linalg.norm(target_vector - source_vectors, axis=1)
        query_index = dist.argmin()

        return query_index

    @staticmethod
    def manhattan_distance(target_vector, source_vectors, rows_used=None):
        """
        Finds the vector with the smallest Euclidean distance in the source
        domain using the most uncertain vector from the target domain.

        Parameters:
            target_vector:
                A 1D array being the most 'uncertain' vector from the pseudo
                population.

            source_vectors:
                A 2D vector containing each of the source vectors as rows.

            rows_used:
                An list/array of indices that will be used to calculate the
                distance measure.
        """

        if rows_used is not None:
            target_vector, source_vectors = ActiveLearner.vector_dim_reduction(
                target_vector, source_vectors, rows_used)

        manhattan_dist = np.sum(np.abs(target_vector - source_vectors), axis=1)

        return manhattan_dist.argmin()

    def commence_work(self):

        test_df = pd.read_csv(os.path.join(
            self._source_path, f"test_file_{self._iteration}.csv"))
        train_df = pd.read_csv(os.path.join(
            self._source_path, f"train_file_{self._iteration}.csv"))
        pseudo_pop_df = pd.read_csv(os.path.join(
            self._source_path, f"pseudo_population_file_{self._iteration}.csv"))

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
                                self._iteration, test_df, train_df, oracle_df, pseudo_pop_df)

        return

    def set_k_top(self, X, y):
        """
        Finds the k most useful parameters.


        """

        n_features = X.shape[1]

        if self._k_best is not None and self._k_best > n_features:
            raise ValueError("k best features cannot exceed number of features in feature space\n"
                             f"self._k_best: {self._k_best}, n_features {n_features}")
        elif self._k_best is None:
            # Use all the features
            return

        self._k_best = SelectKBest(k=self._k_best).fit(
            X, y).get_support(indices=True)

        return

    def init_split_gen(self, sparse_rows: int):
        """
        Generates partitioning for the initial training data for the out of bag
        cross validation.

        Parameters:
            sparse_rows:
                The number of rows in the sparse training set.

        Yields:
            A tuple of vectors detailing how the data should be split for the
            cross validation.
        """

        split_list = []

        for _ in range(self._pick_cv_rounds):

            # We need 80% of the sparse rows to be included in the testing set
            sparse_train_pts = int(0.2 * sparse_rows)
            # Sample 80% of the row indexes to put into the training set
            test_vec = np.random.choice(
                sparse_rows, sparse_train_pts, replace=False)

            train_vec = np.setdiff1d(np.arange(sparse_rows), test_vec,
                                     assume_unique=True)

            split_list.append((train_vec, test_vec))

        self._oob_splits = split_list

    def oob_split_gen(self, sparse_rows: int, total_rows: int):
        """
        Generates partitioning for the training data for the out of bag
        cross validation.

        Yields:
            A tuple of vectors detailing how the data should be split for the
            cross validation.
        """

        if sparse_rows != total_rows:

            for train_vec, test_vec in self._oob_splits:

                filler_test_array = np.arange(sparse_rows, total_rows)
                train_vec = np.append(train_vec, filler_test_array, axis=0)

                yield train_vec, test_vec

        else:

            # We don't need to append additional arrays of ones to the
            # split vectors.
            for train_vec, test_vec in self._oob_splits:

                yield train_vec, test_vec

    def complete_iteration(self, source_type: str, iteration: int,
                           test_df: pd.DataFrame, train_df: pd.DataFrame,
                           oracle_df: pd.DataFrame, pseudo_pop_df: pd.DataFrame):
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

        # Create a dictionary that will store all the scores for each of the
        # iterations
        score_dict = dict()

        for scoring_name in self._scoring_methods.keys():
            # +1 for the offset
            score_dict[scoring_name] = np.zeros(self._oracle_pts_used + 1,
                                                dtype=float)

        test_df_y = test_df["value"].values
        test_df_X = test_df.drop(
            columns=["details_id", "details_name", "value"]).values

        pseudo_pop_ids = pseudo_pop_df["details_id"].values
        pseudo_pop_df_X = pseudo_pop_df.drop(
            columns=["details_id", "details_name", "value"]).values

        train_df_y = train_df["value"].values
        train_df_X = train_df.drop(
            columns=["details_id", "details_name", "value"]).values
        # Count the number of rows in the sparse data
        sparse_rows = train_df_X.shape[0]

        oracle_df_y = oracle_df["value"].values
        oracle_ids = oracle_df["details_id"].values
        oracle_state_names = oracle_df["details_name"].values
        oracle_df_X = oracle_df.drop(
            columns=["details_id", "details_name", "value"]).values

        self.set_k_top(train_df_X, train_df_y)

        # Initialise the split generator
        self.init_split_gen(sparse_rows)

        # Create a new classifier
        iter_cls = self._cls_constructor()

        # Prime the classifier on the sparse data
        iter_cls.prime_classifier(
            train_df_X, train_df_y, n_jobs=self._num_cores)

        prev_prediction = iter_cls.get_classifier().predict_proba(pseudo_pop_df_X)

        y_pred = iter_cls.get_prediction(test_df_X)

        # Get the initial scores
        for scoring_name, score_tuple in self._scoring_methods.items():

                score_method, score_kwargs = score_tuple
                score_val = score_method(test_df_y, y_pred, **score_kwargs)

                print(f"\nInitial scoring\n{scoring_name}-{score_val}\n", end="", flush=True)

                # +1 for the offset
                score_dict[scoring_name][0] = score_val

        print("\nIterations completed:\n")

        for data_used in range(self._oracle_pts_used):

            print(f"{data_used},", end="", flush=True)

            # Get the probability predictions for each of the values in the
            # in the pseudo population
            prob_predict = prev_prediction

            # Get the erp uncertainty values for the probability predictions
            # on the test data
            uncertainty = self._uncertainty_measure(prob_predict)

            # Set any nan uncertainty of nan values to -0.1
            uncertainty[np.isnan(uncertainty)] = -0.1

            # Sample an id from the top n most uncertain values
            top_uncertainty = uncertainty.argsort()[-self._pick_top_n:]

            max_sample_index = 0

            if self._pick_top_n > 1:
                max_sample_index = top_uncertainty[random.randint(
                    0, self._pick_top_n - 1)]

            # Get the field id of the most uncertain point
            uncertain_id = pseudo_pop_ids[max_sample_index]
            self._pop_uncertainty_ids.append(uncertain_id)
            # Save the uncertainty value
            max_uncertainty = uncertainty[max_sample_index]
            self._pop_uncertainty_vals.append(max_uncertainty)

            # Get the input vector corresponding to the most uncertain id
            X_uncertain = pseudo_pop_df_X[max_sample_index]

            # Find vectors from oracle that most closely match the vector
            # from the pseudo population
            query_index = self._distance_measure(
                X_uncertain, oracle_df_X, rows_used=self._k_best)

            oracle_id = oracle_ids[query_index]
            self._oracle_chosen_ids.append(oracle_id)
            self._oracle_chosen_class.append(oracle_df_y[query_index])
            self._oracle_chosen_states.append(oracle_state_names[query_index])

            # Get the min X and y vector from the oracle
            y_min_oracle = oracle_df_y[query_index]
            X_min_oracle = oracle_df_X[query_index]

            # Add the new source_vectors to the training data
            train_df_X = np.vstack((train_df_X, X_min_oracle))
            train_df_y = np.append(train_df_y, [y_min_oracle], axis=0)

            # Remove this same point from the oracle dataframe
            oracle_df_X = np.delete(oracle_df_X, query_index, axis=0)
            oracle_df_y = np.delete(oracle_df_y, query_index)
            oracle_ids = np.delete(oracle_ids, query_index)

            # Create a new classifier
            iter_cls = self._cls_constructor()

            # Prime the classifier on the sparse data
            iter_cls.prime_classifier(
                train_df_X, train_df_y, n_jobs=self._num_cores)

            # Create a prediction for next iteration
            prev_prediction = iter_cls.get_classifier().predict_proba(pseudo_pop_df_X)

            y_pred = iter_cls.get_prediction(test_df_X)

            # Score the classifier once trained on the new data
            for scoring_name, score_tuple in self._scoring_methods.items():

                score_method, score_kwargs = score_tuple
                score_val = score_method(test_df_y, y_pred, **score_kwargs)

                print(f"\n{scoring_name}-{score_val}\n", end="", flush=True)

                # +1 for the offset
                score_dict[scoring_name][data_used + 1] = score_val

            # Create a custom cross validation split
            cross_val_split = self.oob_split_gen(
                sparse_rows, train_df_X.shape[0])
            oob_score = cross_val_score(
                iter_cls.get_classifier(), train_df_X, train_df_y, cv=cross_val_split,
                n_jobs=self._num_cores)

            self._oob_scores.append(np.mean(oob_score))

        output_file_data = {
            "max_pseudo_uncertainty": self._pop_uncertainty_vals,
            "pseudo_id": self._pop_uncertainty_ids,
            "source_id": self._oracle_chosen_ids,
            "source_classes": self._oracle_chosen_class,
            "source_states": self._oracle_chosen_states,
            "oob_scores": self._oob_scores
        }

        # Add the scores onto the output data
        output_file_data.update(score_dict)

        output_dataframe = pd.DataFrame(output_file_data)

        filename = os.path.join(
            self._output_path, f"{source_type}-{iteration}-{self._pick_top_n}.csv")
        output_dataframe.to_csv(path_or_buf=filename, index=False)

        print("\nSaved output dataframe to " + filename, flush=True)


def test_learner(args):

    # Test learner: python SparseClassifier\ActiveLearner.py -t kansas -s nebraska
    # -u erp -m l2_norm -b 3 -n 5 -i 0 -p 3

    data_path = os.path.join(os.getcwd(), "SparseClassifier", "sample_output")

    scoring_methods = {
        'f1_macro': (f1_score, {"average": "macro"}),
        'accuracy_score': (accuracy_score, dict())
    }

    iteration = args.iter

    if args.metric == 'l2_norm':
        distance_measure = ActiveLearner.L2_distance
    elif args.metric == 'cosine':
        distance_measure = ActiveLearner.cos_distance

    if args.uncertainty == 'erp':
        uncertainty_measure = ActiveLearner.erp_uncertainty
    elif args.uncertainty == 'entropy':
        uncertainty_measure = ActiveLearner.entropy_uncertainty

    if args.k_best == 0:
        k_best = None
    else:
        k_best = args.k_best

    oracle_pts_used = args.pixels_used

    if args.target == args.source:
        oracle_source = "t2t"
    else:
        oracle_source = f"s2t_{args.source}"

    pick_top_n = args.top_n

    output_path = os.path.join(
        os.getcwd(), "SparseClassifier", "mock_out")

    new_al = ActiveLearner(data_path, scoring_methods, iteration,
                           distance_measure=distance_measure,
                           uncertainty_measure=uncertainty_measure, k_best=k_best,
                           oracle_pts_used=oracle_pts_used, num_cores=1, oracle_source=oracle_source,
                           pick_top_n=pick_top_n,
                           data_output_path=output_path)

    print(
        f"Commencing ActiveLearner on:\noracle_source:{oracle_source}, iteration:{iteration}, data_folder:mock_out",
        flush=True)

    new_al.commence_work()


def drive_learner(args):

    global STANDARD_PIXELS

    data_folder = f"michael_naive_{args.target}"

    data_path = os.path.join(
        ROOT_DIR, "crop_mapping_sparse", "data", "usa", "classes_3", data_folder)

    scoring_methods = {
        'f1_macro': (f1_score, {"average": "macro"}),
        'accuracy_score': (accuracy_score, dict())
    }

    iteration = args.iter

    if args.metric.lower() == 'l2_norm':
        distance_measure = ActiveLearner.L2_distance
    elif args.metric.lower() == 'cosine':
        distance_measure = ActiveLearner.cos_distance
    elif args.metric.lower() == 'manhattan':
        distance_measure = ActiveLearner.manhattan_distance

    if args.uncertainty.lower() == 'erp':
        uncertainty_measure = ActiveLearner.erp_uncertainty
    elif args.uncertainty.lower() == 'entropy':
        uncertainty_measure = ActiveLearner.entropy_uncertainty

    if args.k_best == 0:
        k_best = None
    else:
        k_best = args.k_best

    oracle_pts_used = args.pixels_used

    if args.target == args.source:
        oracle_source = "t2t"
    else:
        oracle_source = f"s2t_{args.source}"

    top_n = args.top_n

    output_folder_name = f"{args.target.lower()}_{args.uncertainty.lower()}_{args.metric.lower()}_kbest_{'all' if k_best is None else k_best}_topn_{top_n}"

    if args.pixels_used == STANDARD_PIXELS:
        batch_folder_name = f"{args.target.lower()}_{args.uncertainty.lower()}_{args.metric.lower()}_kbest_{'all' if k_best is None else k_best}_topn_{top_n}"
    else:
        batch_folder_name = f"{args.target.lower()}_{args.uncertainty.lower()}_{args.metric.lower()}_kbest_{'all' if k_best is None else k_best}_topn_{top_n}_pixels_{args.pixels_used}"

    output_path = os.path.join(
        os.getcwd(), "active_learner_output", output_folder_name)

    # If the output path does not exist then create it
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    new_al = ActiveLearner(data_path, scoring_methods, iteration,
                           distance_measure=distance_measure,
                           uncertainty_measure=uncertainty_measure, k_best=k_best,
                           oracle_pts_used=oracle_pts_used, num_cores=16, oracle_source=oracle_source,
                           pick_top_n=top_n,
                           data_output_path=output_path)

    print(
        f"Commencing ActiveLearner on:\noracle_source:{oracle_source}, iteration:{iteration}, data_folder:{data_folder}",
        flush=True)

    new_al.commence_work()


def main():

    parser = argparse.ArgumentParser(description="Trains a classifier using "
                                     "active learning.")

    parser.add_argument('-t', '--target', type=str, default='kansas',
                        help='The target domain which the classifier will be trained to categorize.')
    parser.add_argument('-s', '--source', type=str, default='nebraska',
                        help='The domain from which data will be sourced from. If no source domain '
                        'is specified the the source domain will be the same as the target domain.')
    parser.add_argument('-u', '--uncertainty', type=str, default='entropy',
                        help='A function used to measure the uncertainty of probability prediction.')
    parser.add_argument('-m', '--metric', type=str, default='l2_norm',
                        help='The name of a function used to measure the distance between '
                        'vectors in the feature space.')
    parser.add_argument('-b', '--k_best', type=int, default=10,
                        help='If specified, uses the k most important features only when computing '
                        'the distance between vectors in the feature space.')
    parser.add_argument('-n', '--top_n', type=int, default=10,
                        help='Samples data from the top n most uncertain values '
                        'from the pseudo population.')
    parser.add_argument('-i', '--iter', type=int, default=0,
                        help='The iteration to be computed.')
    parser.add_argument('-p', '--pixels_used', type=int, default=5,
                        help='The total number of pixels that will be selected '
                        'domain during that active learning process.')

    args = parser.parse_args()

    drive_learner(args)


if __name__ == '__main__':
    main()
