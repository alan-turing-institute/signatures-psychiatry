#!/usr/bin/env python
"""Code from the paper "A signature-based machine
learning model for bipolar disorder and borderline
personality disorder".

Classifies participants according to the clinical
group they were linked with at the beginning
of the study.
"""

from __future__ import print_function
import random
import numpy as np
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor

from esig import tosig
import matplotlib.pyplot as plt
import time
import matplotlib.dates as mdates
import datetime
import random

from Psychiatry import *
import csv
from logger import Logger

__author__ = "Imanol Perez Arribas"
__credits__ = ["Imanol Perez Arribas", "Guy M. Goodwin", "John R. Geddes",
                    "Terry Lyons", "Kate E. A. Saunders"]
__version__ = "1.0.1"
__maintainer__ = "Imanol Perez"
__email__ = "imanol.perez@maths.ox.ac.uk"

def f(diagnosis, threshold):
        """Maps a clinical group to the corresponding point on the
        plane.

        Args:
            bipolar (int): Clinical group. Borderline if 0, healthy if 1
            and bipolar if 2.

            threshold (list): List of 3 points on the plane.

        Returns:
            2-d tuple: Point on the plane corresponding to the clinical group.

        """

        return threshold[diagnosis]


def findMin(p, A):
        """Given a point p and a list of points A, returns the point
        in A closest to p.

        Args:
            p (2-d tuple): Point on the plane.
            A (list): List of points on the plane.

        Returns:
            2-d tuple: Point in A closest to p.

        """

        m=(-1, (0,0))
        for p0 in A:
                dist=np.linalg.norm(p0-np.array(p))
                if m[0]==-1 or m[0]>dist:
                        m=(dist, p0)
        return m[1]

def check(collection, reg, threshold, order=2):
        """Checks the performance of the model against an out
        of sample set.

        Args:
            collection (list): The out-of-sample set.

            reg (RandomForestRegressor): The trained random forest.

            threshold (list): List of 3 points on the plane.

            order (int): Order of the signature.

        Returns:
            float: Percentage of correct guesses of the predictions.

        """

        # x will contain the input of the model, while
        # y will contain the output.
        x=[]
        y=[]

        for X in collection:
                x.append(list(tosig.stream2sig(np.array(X.data), order)))

                y.append(f(X.diagnosis, threshold=threshold))

        predicted=reg.predict(x)

        guesses=0
        total=0
        for i in range(len(x)):
                if set(findMin(predicted[i], threshold))==set(y[i]):
                        guesses+=1
                total+=1

        return guesses/float(total)


def fit(collection, threshold, order=2):
        """Fits the model using the training set.

        Args:
            collection (list): The training set.

            threshold (list): List of 3 points on the plane.

            order (int): Order of the signature.

        Returns:
            RandomForestRegressor: Trained model.

        """

        # x will contain the input of the model, while
        # y will contain the output.
        x=[]
        y=[]

        for participant in collection:
                # The input will be the signature of the stream of
                # the participant.
                x.append(list(tosig.stream2sig(np.array(participant.data), order)))

                # The output, on the other hand, will be the point
                # on the plane corresponding to the clinical group
                # of the participant.
                y.append(f(participant.diagnosis, threshold=threshold))

        # We train the model using Random Forests.
        reg = RandomForestRegressor(n_estimators=100, oob_score=True)
        reg.fit(x, y)

        return reg
if __name__ == "__main__":
    # Each clinical group is associated with a point on the
    # plane. These points were found using cross-valiation.
    random.seed(1)
    np.random.seed(1)

    threshold=np.array([[1, 0],
                        [0, 1],
                        [-1/np.sqrt(2), -1/np.sqrt(2)]])


    logger = Logger("group_classification")

    # The training and out-of-sample sets are built
    logger.log("Building training and out-of-sample sets...")
    ts, os=buildData(20, training=0.7)
    logger.log("Done.\n")

    # We fit data
    logger.log("Training the model...")
    reg=fit(ts, order=2, threshold=threshold)
    logger.log("Done.\n")

    # We check the performance of the algorithm with out of sample data
    logger.log("Testing the model...")
    accuracy = check(os, reg, order=2, threshold=threshold)
    logger.log("Accuracy of predictions: " + str(round(100*accuracy, 2)) + "%")
