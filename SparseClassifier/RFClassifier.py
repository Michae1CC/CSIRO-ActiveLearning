import abc
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from AbstractClassifier import AbstractClassifier


class RFClassifier(AbstractClassifier):

    def __init__(self):
        super().__init__()

    def prime_classifier(self, train, test, n_jobs=1):

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
        n_estimators = [10, 20, 50]
        random_grid = {'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'n_estimators': n_estimators}

        # search across different combinations, and use all available cores
        cls_search = RandomizedSearchCV(estimator=RandomForestClassifier(warm_start=True),
                                        param_distributions=random_grid, n_iter=30)
        cls_search.fit(train, test)

        # Attain the optimal parameters from the gridsearch
        cls_params = {
            "min_samples_split": cls_search.best_params_.get('min_samples_split'),
            "min_samples_leaf": cls_search.best_params_.get('min_samples_leaf'),
            "max_features": cls_search.best_params_.get('max_features'),
            "max_depth": cls_search.best_params_.get('max_depth'),
            "n_estimators": cls_search.best_params_.get('n_estimators')
        }

        # Construct a random forest classifier using the optimal parameters
        self._classifier = RandomForestClassifier(**cls_params)
        self._classifier.fit(train, test)

        return
