import abc
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd


class AbstractClassifier(abc.ABC):

    def __init__(self):
        """
        Initializes an Abstract classifier.
        """

        self._classifier = None

    def get_classifier(self):
        """
        Returns the underlying classifier being used to make predictions.
        """

        return self._classifier

    @abc.abstractmethod
    def prime_classifier(self, train, test, n_jobs=1):
        """
        Primes the classifier for fitting data and making predictions.

        Parameters:
            train:
                Labelled input data to aid priming.

            test:
                Labels for the input data.
        """

    def train_classifier(self, X_data: np.array, y_data: np.array):
        """
        Trains the classifier labelled data.
        """

        self._classifier.fit(X_data, y_data)

        return

    def pickle_classifier(self, filepath: str = None):
        """
        Pickles the classifier for later use.

        Parameters:
            filepath:
                The path to the file to save the pickled model.
        """

        if filepath is None:
            # The filepath has not been specified, produce a default
            # filename (MODELTYPE_DATE) and path (CWD)
            classifier_model = type(self._classifier).__name__
            datetime_str = datetime.now().strftime(r"%Y%m%d_%I%M%S")

            filepath = f"{classifier_model}_{datetime_str}.pickle"
            filepath = os.path.join(os.getcwd(), filepath)

        with open(filepath, 'wb') as pickle_file:
            pickle.dump(self._classifier, pickle_file)

    def get_prediction(self, X_data: np.array):
        """
        Gets the classifier to make a prediction of unlabelled data
        """

        return self._classifier.predict(X_data)

    def get_prob_prediction(self, X_data):
        """
        Gets the probabilities for an input vector belonging to
        each of the different categories.
        """

        pred_vals = np.ones(shape=(X_data.shape[0],))

        try:
            pred_vals = self._classifier.predict(X_data)
        except:
            pass

        return pred_vals
