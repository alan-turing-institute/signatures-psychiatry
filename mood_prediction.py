#!/usr/bin/env python
"""Code from the paper "A signature-based machine
learning model for bipolar disorder and borderline
personality disorder".

The code predicts, using a stream of data of a
specific length of a participant, the mood
of the participant the following recorded observation.
"""

from __future__ import print_function
import random
import numpy as np
from sklearn.ensemble import RandomForestRegressor

import pandas as pd
from esig import tosig
import matplotlib.pyplot as plt

import psychiatry
from logger import Logger


__author__ = "Imanol Perez Arribas"
__credits__ = ["Imanol Perez Arribas", "Guy M. Goodwin", "John R. Geddes",
                    "Terry Lyons", "Kate E. A. Saunders"]
__version__ = "1.0.1"
__maintainer__ = "Imanol Perez Arribas"
__email__ = "imanol.perez@maths.ox.ac.uk"


def test(collection, reg, order=2):
    """Test the model against out-of-sample data.

    Parameters
    ----------
    collection : list
        Out-of-sample set.
    reg : RandomForestRegressor
        Trained random forest.
    order : int, optional
        Order of the signature, optional.
        Default is 2.
    
    Returns
    -------
    list
        Accuracy.
    list
        MAE.

    """

    # x will contain the input of the model, while
    # y will contain the output that will be used
    # to check the accuracy of the predictions.
    x=[]
    y=[]

    for X in collection:
            # The input will be the signature of the stream of
            # the participant.
            x.append(tosig.stream2sig(np.array(X.data), order))

            # The output, on the other hand, will be the mood
            # of the participant the following observation.
            y.append(X.nextDay[1:len(X.nextDay)])

    # We make predictions based on x.
    predicted = reg.predict(x)

    # We calculate the percentage of correct predictions and
    # the Mean Absolute Error.
    guesses = np.zeros(np.shape(y)[1])
    mae = np.zeros(np.shape(y)[1])

    for i in range(len(x)):
            for j in range(len(y[i])):
                    mae[j] += abs(y[i][j]-predicted[i][j])
                    if abs(y[i][j]-round(predicted[i][j]))<=1:
                            guesses[j] += 1

    guesses /= float(len(x))
    mae /= float(len(x))

    return guesses, mae




def fit(collection, order=2):
    """Trains a random forest.

    Parameters
    ----------
    collection : list
        Training set.
    order : int, optional
        Order of the signature.
        Default is 2.
    
    Returns
    -------
    RandomForestRegressor
        Trained model.

    """

    # x will contain the inputs of the model, and y
    # will contain the outputs.
    x=[]
    y=[]

    for X in collection:
        # The input will be the signature of the stream of
        # the participant.
        x.append(tosig.stream2sig(np.array(X.data), order))

        # The output, on the other hand, will be the mood of
        # the participant the following observation.
        y.append(X.nextDay[1:len(X.nextDay)])

    # We train the model using Random Forests.
    reg = RandomForestRegressor(n_estimators=100)
    reg.fit(x, y)

    return reg



if __name__ == "__main__":
    logger = Logger("mood_prediction")

    clinical_groups = {
                        "Borderline":   0,
                        "Healthy":      1,
                        "Bipolar":      2
                      }
    results = {}

    for group in clinical_groups:
            logger.log(group)
            # We build the training and out of sample sets for
            # the clinical group.
            logger.log("\tBuilding the data...")
            ts, os = psychiatry.buildData(20, path="../data", group=clinical_groups[group])
            logger.log("\tDone.\n")

            # We fit data
            logger.log("\tFitting the model...")
            reg = fit(ts, order=2)
            logger.log("\tDone.\n")

            # We check the performance of the algorithm with out of sample data
            logger.log("\tChecking performance with out of sample data...")
            acc, mae = test(os, reg, order=2)
            logger.log("\tDone.\n")

            percentages = ["{}%".format(np.round(100 * val)) for val in acc]
            mae = np.round(mae, 2)
            
            results[group] = (percentages, mae)

    logger.log("###########")
    logger.log("  Results  ")
    logger.log("###########")

    MOODS = ["Anxious", "Elated", "Sad", "Angry", "Irritable", "Energetic"]

    for group in results:
            logger.log("\n\n{}".format(group))
            logger.log("-"*len(group))
            df_results = pd.DataFrame(np.array(results[group]).T, index=MOODS,
                                      columns=["Accuracy", "MAE"])
            logger.log(df_results.to_string())
