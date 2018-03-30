#!/usr/bin/env python
"""Code from the paper "A signature-based machine
learning model for bipolar disorder and borderline
personality disorder".

The code predicts, using a stream of data of a
specific length of a participant, the mood
of the participant the following recorded observation.
"""

from __future__ import print_function
from sklearn.metrics import roc_curve, auc
import random
import numpy as np
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor

import pandas as pd
from esig import tosig
import matplotlib.pyplot as plt
import time
import matplotlib.dates as mdates
import datetime

from Psychiatry import *
import csv


__author__ = "Imanol Perez Arribas"
__credits__ = ["Imanol Perez Arribas", "Guy M. Goodwin", "John R. Geddes",
                    "Terry Lyons", "Kate E. A. Saunders"]
__version__ = "1.0.1"
__maintainer__ = "Imanol Perez"
__email__ = "imanol.perez@maths.ox.ac.uk"

def check(collection, reg, order=2):
        '''Calculates performance of the model
        against out of sample data.

        Arguments:
                collection (list): The out-of-sample set.

                reg (RandomForestRegressor): The trained random forest.

                order (int): Order of the signature.

        Returns:
                list, list: Two vectors of size equal to the number of
                mood categories - six - and where the first vector
                indicates the percentage of correct predictions of the
                model, where what a "correct prediction" is defined
                as in the paper, and the second vector contains the
                Mean Absolute Errors.

        '''

        # x will contain the input of the model, while
        # y will contain the output that will be used
        # to check the accuracy of the predictions.
        x=[]
        y=[]

        for X in collection:
                # The input will be the signature of the stream of
                # the participant.
                x.append(list(tosig.stream2sig(np.array(X.data), order)))

                # The output, on the other hand, will be the mood
                # of the participant the following observation.
                y.append(X.nextDay[1:len(X.nextDay)])

        # We make predictions based on x.
        predicted=reg.predict(x)

        # We calculate the percentage of correct predictions and
        # the Mean Absolute Error.
        guesses=[0 for i in range(len(y[0]))]
        mae=[0 for i in range(len(y[0]))]

        total=0
        for i in range(len(x)):
                for j in range(len(y[i])):
                        mae[j]+=abs(y[i][j]-predicted[i][j])
                        if abs(y[i][j]-round(predicted[i][j]))<=1:
                                guesses[j]+=1
                total+=1
        guesses=[g/float(total) for g in guesses]
        mae=[m/float(total) for m in mae]

        return guesses, mae




def fit(collection, order=2):
        '''Calculates performance of the model
        against out of sample data.

        Arguments:
                collection (list): The out-of-sample set.

                order (int): Order of the signature.

        Returns:
                RandomForestRegressor: Trained model.

        '''

        # x will contain the inputs of the model, and y
        # will contain the outputs.
        x=[]
        y=[]

        for X in collection:
                # The input will be the signature of the stream of
                # the participant.
                x.append(list(tosig.stream2sig(np.array(X.data), order)))

                # The output, on the other hand, will be the mood of
                # the participant the following observation.
                y.append(X.nextDay[1:len(X.nextDay)])

        # We train the model using Random Forests.
        reg = RandomForestRegressor(n_estimators=100)
        reg.fit(x, y)

        return reg



if __name__ == "__main__":
        clinical_groups = {
                                "Borderline": -1,
                                "Healthy": 0,
                                "Bipolar": 1
                           }
        results = {}

        for group in clinical_groups:
                print(group)
                # We build the training and out of sample sets for
                # the clinical group.
                print("\tBuilding the data...")
                ts, os=buildData(20, group=clinical_groups[group])
                print("\tDone.\n")

                # We fit data
                print("\tFitting the model...")
                reg=fit(ts, order=2)
                print("\tDone.\n")

                # We check the performance of the algorithm with out of sample data
                print("\tChecking performance with out of sample data...")
                guesses, mae = check(os, reg, order=2)
                print("\tDone.\n")

                percentage = ["%s%%"%np.round(100*p) for p in guesses]
                mae = np.round(mae, 2)
                results[group] = (percentage, mae)

        print("===========")
        print("  Results  ")
        print("===========")

        MOODS = ["Anxious", "Elated", "Sad", "Angry", "Irritable", "Energetic"]

        for group in results:
                print("\n\n%s"%group)
                print("%s"%"-"*len(group))
                df_results = pd.DataFrame(np.array(results[group]).T, index=MOODS, columns=["Accuracy", "MAE"])
                print(df_results)
