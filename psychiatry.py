#!/usr/bin/env python
"""Code from the paper "A signature-based machine
learning model for bipolar disorder and borderline
personality disorder".

This is the core file, that handles objects of type
Participant, loads the cohort, etc.
"""

import random
import os
import numpy as np
import matplotlib.dates as mdates
import datetime
import time
import csv

__author__ = "Imanol Perez Arribas"
__credits__ = ["Imanol Perez Arribas", "Guy M. Goodwin", "John R. Geddes",
                "Terry Lyons", "Kate E. A. Saunders"]
__version__ = "1.0.1"
__maintainer__ = "Imanol Perez Arribas"
__email__ = "imanol.perez@maths.ox.ac.uk"

class Participant:
    """Class that handles individual participants.

    Parameters
    ----------
    data : list
        Score data of the participant.
    id_n : int
        ID number of the participant.
    diagnosis : int
        Whether the participant is borderline (0), healthy (1)
        or bipolar (2).
    nextDay : list
        Next day's list of scores.

    """
    def __init__(self, data, id_n, diagnosis, nextDay):

            self.idNumber = id_n
            self.data = data
            self.diagnosis = diagnosis
            self.nextDay = nextDay


def string2datenum(s, f):
	"""Converts a string date in format f to a float.

	Parameters
    ----------
    s : str
        Date.
    f : str
        Format.

    Returns
    -------
    float
        `s` converted to float.
	"""
	return mdates.date2num(datetime.datetime.fromtimestamp(time.mktime(time.strptime(s, f))))


def loadCSV(file, path):
    """Loads data from CSV file.

    Parameters
    ----------
    file : str
        Path of the file.

    """

    try:
        data = list(csv.reader(open(file)))
    except:
        return False
    
    data = [(string2datenum(l[0].split(".")[0], "%Y-%m-%d %H:%M:%S"),
            int(l[1]), int(l[2]), int(l[3]), int(l[4]), int(l[5]),
            int(l[6])) for l in data]

    n = int(file.split("/")[-1].split("-")[0])

    # patients.csv contains the clinical group associated
    # with each participant.
    participants=list(csv.reader(open(os.path.join(path, "patients.csv"))))
    for l in participants:
        if int(l[0])==n:
            if not l[1].isdigit():
                return False

            bp0 = int(l[1])
            break
    bp = {1: 2, 2: 0, 3: 1}[bp0]

    participant=Participant(data, n, bp, [])

    return participant

def loadParticipants(path):
    """Loads the participant cohort.
    
    Parameters
    ----------
    path : str
        Location of the directory with the data.

    Returns
    -------
    list of Participant

    """

    participants=[]
    for filename in sorted(os.listdir(os.path.join(path, "cohort_data"))):
        participant=loadCSV(os.path.join(path, "cohort_data/%s"%filename), path)
        if not participant: continue
        participants.append(participant)

    return participants



def normalise(participant, time=True):
    """Normalises the mood path of a given patient.

    Parameters
    ----------
    participant : Participant
        Participant whose data will be normalised.
    time : bool, optional
        If True, time is also normalised.
        Default is True.

    Returns
    -------
    Participant

    """

    t0=participant.data[0][0]
    t1=participant.data[-1][0]
    scoreMAX=7
    scoreMIN=1
    cum=np.zeros(np.shape(participant.data)[1])

    for i in range(len(participant.data)):
        participant.data[i] = list(participant.data[i])
        for j in range(len(participant.data[i])):
            if j==0:
                if time:
                    # Normalise time
                    participant.data[i][j] -= t0
                    participant.data[i][j] /= t1-t0
            else:
                # Normalise scores
                participant.data[i][j] -= scoreMIN
                participant.data[i][j] *= 2/float(scoreMAX-scoreMIN)
                participant.data[i][j] -= scoreMIN
                cum[j] += participant.data[i][j]
                participant.data[i][j] = cum[j]
        participant.data[i] = tuple(participant.data[i])

    return participant

def buildData(size, path, training=0.7, groups=None):
    """Builds the training and out-of-sample sets.

    Parameters
    ----------
    size : int
        Size of each bucket (number of observations).
    path : str
        Directory where the data is located.
    training : float, optional
        Percentage of the data that will be used to
        train the model.
        Default is 0.7.
    groups : list or None, optional
        If not None, only the corresponding diagnosis
        group will be used to build the dataset.
        Deafult is None.
    
    Returns
    -------
    list
        Training set.
    list
        Out-of-sample set.
    
    """

    diagnosis_map = {
                        "borderline":   0,
                        "healthy":      1,
                        "bipolar":      2
                    }
    if groups is not None:
        groups = [diagnosis_map[group] for group in groups]
        
    patients=loadParticipants(path)
    data=[]

    for patient in patients:
        if groups is not None and patient.diagnosis not in groups:
            continue

        for i in range(0, len(patient.data)-size, size):
            p = Participant(patient.data[i:i+size], patient.idNumber,
                            patient.diagnosis, patient.data[i+size])
            data.append(normalise(p))

    random.shuffle(data)
    training_set = data[0:int(training*len(data))]

    out_of_sample = data[int(training*len(data)):len(data)]
    return training_set, out_of_sample

def buildSyntheticSigData(path, cohort=239673, training=0.7, groups=None):
    """ Builds training and out of sample sets of synthetic signatures.

    Parameters
    ----------
    path : str
        Directory where the data is located.
    cohort : int
        ID of the synthetic cohort.
        Default is 239673.
    training : float, optional
        Proportion of the data that will be used to train the model.
        Default is 0.7.
    groups : list or None, optional
        If not None, only the corresponding diagnosis groups will be used to build the dataset.
        Default is None.

    Returns
    -------
    list
        Training set.
    list
        Out-of-sample set.

    """

    diagnosis_map = {
        "borderline": -1,
        "healthy": 0,
        "bipolar": 1
    }
    if groups is not None:
        groups = [diagnosis_map[group] for group in groups]

    signatures = np.genfromtxt(os.path.join(path, "cohort_" + str(cohort) + "_sigs.pickle"), delimiter=',')
    diagnoses = np.genfromtxt(os.path.join(path, "cohort_" + str(cohort) + "_diagnosis.pickle"), delimiter=',')
    data = []

    for i, (diagnosis, signature) in enumerate(zip(diagnoses, signatures)):
        if groups is not None and diagnosis not in groups:
            continue

        # Relabel the diagnoses at this point to match the convention in the psychiatry package
        p = Participant(signature, i, int(diagnosis+1), None)
        data.append(p)

    random.shuffle(data)

    training_set = data[0:int(training * len(data))]
    out_of_sample = data[int(training * len(data)):len(data)]

    return training_set, out_of_sample
