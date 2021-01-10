# Sparse Data Programs Manual

## Overview

This document provides a comprehensive guide to using program files that are part of the active learning on sparse data project.

## The `AbstractClassifier` class

The `AbstractClassifier` class is an abstract base class used to act as a wrapper class for `sklearn`'s classifiers and to provide some of the base functionality for the different learning processes. One important note for using this class is that the `prime_classifier` method **must** initialize and set the the value of `self._classifier` to the value of the new classifier. Some methods that might be of interest when using this class are listed below:

* The `get_classifier` method which returns the underlying classifier instance
* The `train_classifier` method which trains the underlying classifier with feature vectors `X_data` and their corresponding labels `y_data`
* The `pickle_classifier` method which pickles the classifier in its current state. A file path may be specified to save the model, otherwise the model is saved in the current working directory with a filename generated based on the date and time.
* The `get_prediction` method which provides a probability prediction for a given set of feature vectors.

## The `RSCVClassifier` class

The `RSCVClassifier` is a child class of `AbstractClassifier`. This class primes a `RandomizedSearchCV` and sets it as the `self._classifier` value.

## The `NaiveLearner` class

The `NaiveLearner` class (featured in `NaiveLearner.py`) naively constructs a training set by randomly selecting points from the source domain. Most of the low level detail is automatically handled so all that is required from the user is proper initiation. Running `NaiveLearner.py` with the correct command line arguments will automatically create an instance of the `NaiveLearner` class that will begin work. The `NaiveLearner.py` expects three additional command line arguments, these being the *iteration number* that the `NaiveLearner` will complete, *domain type* (eg `t2t` or `s2t_nebraska`) and a data folder within `F:\crop_mapping_sparse\data\usa\classes_3` to be used. An example execution is given below
```
python3.7 -W ignore ActiveLearner.py 0 s2t_all michael_naive_kansas
```
Once the script has finished running the results will be saved to the current working directory inside the folder "naive_learner_output/{target_state}". The data source folder and output folder may be change using the `data_path` and `output_path` parameters within the `drive_learner` function, respectively.

## The `ActiveLearner` class

The `ActiveLearner` class (featured in `ActiveLearner.py`) naively constructs a training set by intelligently selecting points from the source domain. Much like `NaiveLearner.py` most of the low level detail is automatically handled so all that is required from the user is proper initiation. Running `ActiveLearner.py` with the correct command line arguments will automatically create an instance of the `ActiveLearner` class that will begin work. `ActiveLearner.py` expects several different command line arguments, listed below:
* `-t` or `--target`: The target domain which the classifier will be trained to categorize. Example `kansas`.
* `-s` or `source`: The domain from which data will be sourced from. If no source domain is specified the the source domain will be the same as the target domain. Example: `north_dakota`.
* `-u` or `--uncertainty`: The function (name) used to measure the uncertainty of probability prediction. Example `erp`.
* `-m` or `--metric`: The name of a function used to measure the distance between vectors in the feature space. Example `cosine`.
* `-b` or `--k_best`: If specified, uses the k most important features only when computing the distance between vectors in the feature space. Example `10`.
* `-n` or `--top_n`: Samples data from the top n most uncertain values from the pseudo population. Example `20`.
* `-i` or `--iter`: The iteration number to be computed.
* `-p` or `--pixels_used`: The total number of pixels that will be selected domain during that active learning process.

An example run of the `ActiveLearner` is given below
```
python3.7 -W ignore ActiveLearner.py --target kansas --source north_dakota --uncertainty erp --metric cosine --k_best 10 --top_n 1 --iter 0 --pixels_used 100
```

Once the script has finished running the results will be saved to the current working directory inside the folder with the following naming convention "active_learner_output/{target_state}_{uncertainty_name}_{meteric_name}_kbest_{k_best_value}_topn_{top_n_value}". The data source folder and output folder may be change using the `data_path` and `output_path` parameters within the `drive_learner` function, respectively.