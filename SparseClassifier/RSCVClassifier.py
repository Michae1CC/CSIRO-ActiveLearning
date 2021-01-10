import abc
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from AbstractClassifier import AbstractClassifier


class RSCVClassifier(AbstractClassifier):

    def __init__(self):
        super().__init__()

    def prime_classifier(self, train, test, n_jobs=-1):

        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [50]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Number of trees in random forest
        n_estimators = [100, 250, 500]
        random_grid = {'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'n_estimators': n_estimators}

        # search across different combinations, and use all available cores
        self._classifier = RandomizedSearchCV(estimator=RandomForestClassifier(warm_start=True),
                                              param_distributions=random_grid, n_iter=30, n_jobs=n_jobs)

        self._classifier.fit(train, test)

        return
