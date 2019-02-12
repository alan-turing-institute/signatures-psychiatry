#!/usr/bin/env python
"""Code from the paper "A signature-based machine
learning model for bipolar disorder and borderline
personality disorder".

Classifies participants according to the clinical
group they were linked with at the beginning
of the study.
"""

from __future__ import print_function
import argparse
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score, accuracy_score

from esig import tosig

import psychiatry
from logger import Logger


__author__ = "Imanol Perez Arribas"
__credits__ = ["Imanol Perez Arribas", "Guy M. Goodwin", "John R. Geddes",
                "Terry Lyons", "Kate E. A. Saunders"]
__version__ = "1.0.1"
__maintainer__ = "Imanol Perez Arribas"
__email__ = "imanol.perez@maths.ox.ac.uk"


def _findMin(p, A):
    """Given a point p and a list of points A, returns the point
    in A closest to p.

    Parameters
    ----------
    p : array
        Point on the plane.
    A : list
        List of points on the plane.

    Returns
    -------
    tuple
        Point in A closest to p in the Euclidean metric.

    """

    m=(-1, (0,0))
    for p0 in A:
            dist = np.linalg.norm(p0-np.array(p))
            if m[0]==-1 or m[0]>dist:
                    m = (dist, p0)
    
    return tuple(m[1])

def test(collection, reg, threshold, order=2, is_sig=False):
    """Tests the model against an out-of-sample set.

    Parameters
    ----------
    collection : list
        The out-of-sample set.
    reg : RandomForestRegressor
        Trained random forest.
    threshold : array
        List of 3 points on the plane.
    order : int, optional
        Order of the signature.
        Default is 2.
    is_sig : bool, optional
        Indicates whether the data in the training set has already been converted to the signature.

    Returns
    -------
    float
        Accuracy of the predictions.

    """

    # x will contain the input of the model, while
    # y will contain the output.
    x=[]
    y=[]

    for X in collection:
        if is_sig:
            x.append(X.data)
        else:
            x.append(list(tosig.stream2sig(np.array(X.data), order)))

        y.append(threshold[X.diagnosis])

    predicted_raw = reg.predict(x)
    predicted = np.array([_findMin(prediction, threshold) for prediction in predicted_raw])
    
    # Convert predictions and true values to labels
    predicted_labels = [threshold.tolist().index(val.tolist()) for val in predicted]
    y_labels = [threshold.tolist().index(val.tolist()) for val in y]

    acc = accuracy_score(y_labels, predicted_labels)
    roc = roc_auc_score(y_labels, predicted_labels)

    return acc, roc


def fit(collection, threshold, order=2, is_sig=False):
    """Fits the model using the training set.

    Parameters
    ----------
    collection : list
        Training set.
    threshold : array
        List of 3 points on the plane.
    order : int, optional
        Order of the signature.
        Default is 2.
    is_sig : bool, optional
        Indicates whether the data in the training set has already been converted to the signature.
        If True, order is unused.
        Default is False.

    Returns
    -------
    RandomForestRegressor
        Trained model.

    """

    # x will contain the input of the model, while
    # y will contain the output.
    x=[]
    y=[]

    for participant in collection:
        # The input will be the signature of the stream of
        # the participant.
        if is_sig:
            x.append(participant.data)
        else:
            x.append(tosig.stream2sig(np.array(participant.data), order))

        # The output, on the other hand, will be the point
        # on the plane corresponding to the clinical group
        # of the participant.
        y.append(threshold[participant.diagnosis])

    # We train the model using Random Forest.
    reg = RandomForestRegressor(n_estimators=100, oob_score=True)
    reg.fit(x, y)

    return reg

if __name__ == "__main__":
    # Each clinical group is associated with a point on the
    # plane. These points were found using cross-validation.

    # Set up command line argument parsers
    # --seed (optional): Sets the random seed; default is the original value used in this script.
    # --synth (optional): If not specified at all, mood score data is loaded.
    #                     If --synth is given without a value, cohort 772192 (synthetic signatures) is loaded.
    #                     If a value for --synth is given, the specified cohort of synthetic signatures is loaded.
    parser = argparse.ArgumentParser(
        description="Perform classification on pairs of diagnoses (healthy, borderline and bipolar)")
    parser.add_argument("--seed", type=int, default=83042,
        help="seed for the random number generators (int, default=83042)")
    parser.add_argument("--synth", nargs="?", type=int, const=772192,
        help="ID of cohort of synthetic mood score signatures, if they are to be used (int, default=772192 if --synth \
              alone is provided, or None (i.e. load original mood score data) if not)")
    args = parser.parse_args()

    logger = Logger("pairwise_group_classification")

    random.seed(args.seed)
    np.random.seed(args.seed)
    logger.log("Random seed has been set to {}".format(args.seed))

    if args.synth is None:
        logger.log("Preparing to load mood score data")
        use_synth_sig = False
    else:
        logger.log("Preparing to load synthetic signatures from cohort {}\n".format(args.synth))
        use_synth_sig = True

    threshold=np.array([[1, 0],
                        [0, 1],
                        [-1/np.sqrt(2), -1/np.sqrt(2)]])

    diagnosis = ("healthy", "bipolar", "borderline")
    accuracy_results = pd.DataFrame(index=diagnosis, columns=diagnosis)
    auc_results = pd.DataFrame(index=diagnosis, columns=diagnosis)

    for i, group1 in enumerate(diagnosis):
        for group2 in diagnosis[i + 1:]:
            
            groups = (group1, group2)
            
            # The training and out-of-sample sets are built
            logger.log("Loading {} and {}...".format(group1, group2))

            if use_synth_sig:
                ts, os = psychiatry.buildSyntheticSigData("synthetic_signatures", cohort=args.synth, training=0.7,
                                                          groups=groups)
            else:
                ts, os = psychiatry.buildData(20, "../data", training=0.7,
                                              groups=groups)

            logger.log("Done.")

            # We fit data
            logger.log("Training the model...")
            reg = fit(ts, order=2, threshold=threshold, is_sig=use_synth_sig)
            logger.log("Done.")

            # We check the performance of the algorithm with out of sample data
            logger.log("Testing the model...")
            accuracy, auc = test(os, reg, order=2, threshold=threshold, is_sig=use_synth_sig)
            logger.log("Done.\n")

            # We save the accuracy in the results table.
            accuracy_results.loc[group1][group2] = accuracy
            auc_results.loc[group1][group2] = auc


    logger.log("###########")
    logger.log("  Results  ")
    logger.log("###########")

    logger.log("\nAccuracy:")
    logger.log(accuracy_results.to_string())
    logger.log("\nAUC:")
    logger.log(auc_results.to_string())
