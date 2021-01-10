import numpy as np
from copy import deepcopy
from random import sample
from sklearn import datasets

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity


def erp_test():
    digits = datasets.load_digits()
    rfc = RandomForestClassifier(n_estimators=100)
    rfc.fit(digits.data[:-50], digits.target[:-50])
    pred_prob = rfc.predict_proba(digits.data[-50:])
    max_prob = np.max(pred_prob, axis=1)
    max_prob_index = np.argmax(pred_prob, axis=1)

    m, n = pred_prob.shape
    # Calculate deletion index for the matrix instead of just a single row.
    max_prob_index = np.arange(m) * n + max_prob_index
    deleted_max = np.delete(
        pred_prob.ravel(), max_prob_index, axis=0).reshape(m, -1)

    deleted_max_ones = deepcopy(deleted_max)
    deleted_max_ones[deleted_max_ones == 0] = 1

    print(deleted_max)
    print(deleted_max_ones)

    # TODO: What to do with log(0) values (classes with a prob predicition of 0)?
    intmd_prod = deleted_max * np.log(deleted_max_ones)
    edi = np.log(max_prob) - (1 / (1 - max_prob)) * \
        np.nansum(deleted_max, axis=1)

    exp_edi = np.exp(edi)
    erp = exp_edi / (exp_edi + pred_prob.shape[1] - 1)
    print(erp)


def L2_distance(point1, points):
    dist = np.linalg.norm(point1 - points, axis=1)
    min_dist = dist[dist.argmin()]
    query_index = dist.argmin()
    query_instance = points[dist.argmin()]
    return min_dist, query_index, query_instance


def cos_distance():
    vec_1 = np.random.rand(10).reshape(1, -1)
    vec_2 = np.random.rand(10, 10) * 40

    vec_1 = np.array([1, 1, 1]).reshape(1, -1)
    vec_2 = np.asarray([[1, 0, 0], [0, 1, 1], [1, 1, 1], [2, 1, 1]])

    dist = cosine_similarity(vec_1, vec_2)
    print(dist)
    print(dist[0])

    print(cosine_similarity(np.array([1, 0, 0]).reshape(
        1, -1), np.array([0, 1, 0]).reshape(1, -1)))


deleted_max = np.array([0.1, 0, 0.7, 0.2, 0, 0.1])
