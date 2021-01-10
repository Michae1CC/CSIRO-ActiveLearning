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

def drive_learner(iteration, oracle_source, data_folder):
    
    data_path = os.path.join(ROOT_DIR, "crop_mapping_sparse", "data", "usa", data_folder)
    output_path = os.path.join(os.getcwd(), "naive_learner_output")

    scoring_methods = {'f1_macro': (f1_score, {"average": "macro"}),
                       'accuracy_score': (accuracy_score, dict())}
    new_nv = NaiveLearner(data_path, output_path, iteration,
        oracle_pts_used=2000, num_cores=4, oracle_source=oracle_source,
        output_path=output_path)

    print(f"Commencing oracle_source:{oracle_source}, iteration:{iteration}, data_folder:{data_folder}", flush=True)

    new_nv.commence_work()

def main():

    if (len(sys.argv) != 3) or (not sys.argv[1].isdigit()):
        USAGE = """USAGE:\n\tnaive_learner_driver.py [iter] [oracle source] [optional: data folder]"""
        return

    # Have a default data file
    data_folder = "michael_naive_nebraska"

    if len(sys.argv) == 4:
        data_file = sys.argv[3]
    
    iteration = int(sys.argv[1])
    oracle_source = sys.argv[2]

    drive_learner(iteration, oracle_source, data_folder)

if __name__ == '__main__':
    main()