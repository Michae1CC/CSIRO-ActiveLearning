import os
import sys
import copy
import shutil
import unittest

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

from ActiveLearner import ActiveLearner


class TestActiveLearner(unittest.TestCase):

    def test_vector_dim_reduction(self):

        target_vector = np.array([34, 53, 23, 12, 86, 24])
        source_vectors = np.asarray([
            [73, 15, 27, 12, 93, 37],
            [23, 53, 78, 53, 14, 76],
            [42, 25, 54, 58, 70, 12]
        ])

        target_out, source_out = ActiveLearner.vector_dim_reduction(target_vector,
                                                                    source_vectors, [0, 2, 5])

        expected_target = np.array([34, 23, 24])
        expected_source = np.asarray([
            [73, 27, 37],
            [23, 78, 76],
            [42, 54, 12]
        ])

        np.testing.assert_almost_equal(
            expected_target, target_out
        )

        np.testing.assert_almost_equal(
            expected_source, source_out
        )

    def test_L2_distance(self):

        # Test for R3
        vec_1 = np.array([1, 0, 0])
        vec_2 = np.asarray([[0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 1, 1]])

        query_index = ActiveLearner.L2_distance(
            vec_1, vec_2)

        self.assertEqual(query_index, 1, "Actual min index does not"
                         "match expected value.")

        # Test for R5
        vec_1 = np.array([1, 5, 2, 4.5, 3])
        vec_2 = np.asarray([
            [-1, 3.5, 4, 7, 0],
            [0, 9, 8, 5.5, 3.5],
            [1.5, 7, -1.1, 4, 3.2],
            [0.5, 4.1, 5, 4, 3.8],
            [0.9, 4.3, 2.5, 4.1, 2.7],
            [3, 1, 7, 4.5, 7]])

        query_index = ActiveLearner.L2_distance(
            vec_1, vec_2)

        self.assertEqual(query_index, 4, "Actual min index does not"
                         "match expected value.")

        # Test dimensionality reduction
        vec_1 = np.array([0, 1, 5, 1, 10, 0, 8])
        vec_2 = np.asarray([
            [10, 92, 0, 32, 2, 10, 1],
            [0, 1, 29, 1, -3, 1, 10],
            [-11, 1, 5, 1, 10, 0, 8]
        ])

        query_index = ActiveLearner.L2_distance(
            vec_1, vec_2, rows_used=[0, 1, 3, 5])

    def test_cos_distance(self):

        # Test for R3, positive case
        vec_1 = np.array([1, 1, 1])
        vec_2 = np.asarray([[1, 0, 0], [0, 1, 1], [1, 1, 1], [2, 1, 1]])

        query_index = ActiveLearner.cos_distance(
            vec_1, vec_2)

        self.assertEqual(query_index, 2, "Actual min index does not"
                         "match expected value.")

        # Test for R3, negative case
        vec_1 = np.array([1, 1, 1])
        vec_2 = np.asarray([[1, 0, 0], [0, 1, 1], [-1, -1, -1], [2, 1, 1]])

        query_index = ActiveLearner.cos_distance(
            vec_1, vec_2)

        self.assertEqual(query_index, 2, "Actual min index does not"
                         "match expected value.")

        # Test for R2, negative case
        vec_1 = np.array([1, 0])
        vec_2 = np.asarray([[0, 1], [1, -1]])

        query_index = ActiveLearner.cos_distance(
            vec_1, vec_2)

        self.assertEqual(query_index, 1, "Actual min index does not"
                         "match expected value.")

    def test_manhattan_distance(self):

        # Test for R3, positive case
        vec_1 = np.array([1, -1, 2])
        vec_2 = np.asarray([[1, -1, 2], [-3.5, 1, 1], [0, 0, 0], [-6, 2, -3]])

        query_index = ActiveLearner.manhattan_distance(
            vec_1, vec_2)

        self.assertEqual(query_index, 0, "Actual min index does not"
                         "match expected value.")

    def test_edi_value(self):

        # Testing array 1
        test_vec_1 = np.asarray([
            [0.5, 0.5, 0, 0],
            [0.5, 0.3, 0.1, 0.1],
            [0.5, 0.2, 0.2, 0.1]
        ])

        expected_vec = np.array([0, 0.9503, 1.0549])
        actual_vec = ActiveLearner.edi_value(test_vec_1)

        np.testing.assert_almost_equal(
            expected_vec, actual_vec, decimal=3
        )

        # Testing array 2
        test_vec_2 = np.asarray([
            [0.7, 0.3, 0, 0],
            [0.7, 0.2, 0.1, 0],
            [0.7, 0.1, 0.1, 0.1]
        ])

        expected_vec = np.array([0.8473, 1.4838, 1.9459])
        actual_vec = ActiveLearner.edi_value(test_vec_2)

        np.testing.assert_almost_equal(
            expected_vec, actual_vec, decimal=3
        )

        # Testing array 3
        test_vec_3 = np.asarray([
            [0.5, 0.5, 0, 0],
            [0.5, 0.25, 0.25, 0],
            [0.5, 1/6, 1/6, 1/6]
        ])

        expected_vec = np.array([0, 0.6931, 1.0986])
        actual_vec = ActiveLearner.edi_value(test_vec_3)

        np.testing.assert_almost_equal(
            expected_vec, actual_vec, decimal=3
        )

        # Testing array 4
        test_vec_4 = np.asarray([
            [0.7, 0.3, 0, 0],
            [0.7, 0.1, 0.1, 0.1],
            [0.7, 0.15, 0.15, 0],
            [0.8, 0.2, 0, 0],
            [0.8, 0.1, 0.1, 0]
        ])

        expected_vec = np.array([0.85, 1.95, 1.54, 1.39, 2.08])
        actual_vec = ActiveLearner.edi_value(test_vec_4)

        np.testing.assert_almost_equal(
            expected_vec, actual_vec, decimal=2
        )

    def test_erp_value(self):

        # Testing array 1
        test_vec_1 = np.asarray([
            [0.7, 0.1, 0.1, 0.1],  # close to 0.7
            [0.51, 0.48, 0.05, 0.05],  # close to 0.05
            [0.5, 0.2, 0.2, 0.1]
        ])

        expected_vec = np.array([0.7, 0.39136153, 0.48907871])
        actual_vec = ActiveLearner.erp_confidence(test_vec_1)

        np.testing.assert_almost_equal(
            expected_vec, actual_vec, decimal=3
        )

    def test_entropy_uncertainty(self):

        test_vec = np.asarray(
            [
                [0.7, 0.1, 0.1, 0.1],
                [0.51, 0.48, 0.05, 0.05],
                [0.5, 0.2, 0.2, 0.1],
                [0.25, 0.25, 0.25, 0.25]
            ]
        )

        actual_output = ActiveLearner.entropy_uncertainty(test_vec)

        expected_output = np.array(
            [0.9404479886553263, 0.9952841535584157, 1.2206072645530173, 1.3862943611198906])

        np.testing.assert_almost_equal(
            expected_output, actual_output, decimal=5
        )

    def test_oob_split_gen(self):

        data_path = os.path.join(
            os.getcwd(), "SparseClassifier", "sample_output")

        new_nv = ActiveLearner(data_path, None, 0, oracle_pts_used=10,
                               num_cores=1, pick_top_n=20, pick_cv_rounds=5)

        sparse_rows = 40
        total_rows = 60

        expected_test_len = int(0.2 * sparse_rows)

        new_nv.init_split_gen(sparse_rows)

        test_split = new_nv.oob_split_gen(sparse_rows, total_rows)

        for train_vec, test_vec in test_split:

            self.assertEqual(expected_test_len, len(test_vec))
            self.assertEqual(total_rows - expected_test_len, len(train_vec))
            self.assertEqual(len(test_vec) + len(train_vec), total_rows)

            self.assertEqual(len(np.intersect1d(train_vec, test_vec)), 0)

            # Append the train and test vectors together
            append_vec = np.append(train_vec, test_vec)
            append_vec = np.sort(append_vec)

            np.testing.assert_equal(
                np.arange(total_rows), append_vec
            )

        test_split_1 = new_nv.oob_split_gen(sparse_rows, total_rows)
        test_split_2 = new_nv.oob_split_gen(sparse_rows, total_rows)

        for (train_vec_1, test_vec_1), (train_vec_2, test_vec_2) in zip(test_split_1, test_split_2):

            np.testing.assert_equal(
                train_vec_1, train_vec_2
            )

            np.testing.assert_equal(
                test_vec_1, test_vec_2
            )




if __name__ == '__main__':
    unittest.main()
