#!/usr/bin/env python
"""Code from the paper "A signature-based machine
learning model for bipolar disorder and borderline
personality disorder".

This is the core file, that handles objects of type
Participant, loads the cohort, etc.
"""

import random
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor

from esig import tosig
import matplotlib.pyplot as plt
import time
import matplotlib.dates as mdates
import datetime
import csv

__author__ = "Imanol Perez Arribas"
__credits__ = ["Imanol Perez Arribas", "Guy M. Goodwin", "John R. Geddes",
                    "Terry Lyons", "Kate E. A. Saunders"]
__version__ = "1.0.1"
__maintainer__ = "Imanol Perez"
__email__ = "imanol.perez@maths.ox.ac.uk"

class Participant:
        idNumber=0
        data=[]
        diagnosis=1
        nextDay=[]
        signature=[]
        def __init__(self, data, id_n, diagnosis, nextDay):
                """Class that handles individual participants.

                Args:
                    data (list): Score data of the participant.

                    id_n (int): ID number of the participant.

                    diagnosis (int): Whether the participant is borderline (0),
                    healthy (1) or bipolar (2).

                    nextDay (list): Next day's scores.


                """

                self.idNumber=id_n
                self.data=data
                self.diagnosis=diagnosis
                self.nextDay=nextDay

def string2datenum(s, f):
	'''Converts a string date in format f to a number

	Arguments:
		s (string): Date that has to be converted to int
		f (string): Format of s
	'''
	return mdates.date2num(datetime.datetime.fromtimestamp(time.mktime(time.strptime(s, f))))


def loadCSV(file):
        '''Loads data from CSV file.

        Arguments:
                file (string): Path of the file.

        '''

        try:
                data=list(csv.reader(open(file)))
        except:
                return False
        data=[(string2datenum(l[0].split(".")[0], "%Y-%m-%d %H:%M:%S"), int(l[1]), int(l[2]), int(l[3]), int(l[4]), int(l[5]), int(l[6])) for l in data]
        n=int(file.split("/")[1].split("-")[0])

        # patients.csv contains the clinical group associated
        # with each participant.
        participants=list(csv.reader(open("patients.csv")))
        for l in participants:
                if int(l[0])==n:
                        if not l[1].isdigit():
                                return False
                        bp0=int(l[1])
                        break
        bp = {1: 2, 2: 0, 3: 1}[bp0]

        participant=Participant(data, n, bp, [])
        return participant

def loadParticipants():
        '''Loads the participants cohort.

        '''

        participants=[]
        for filename in sorted(os.listdir("cohort_data")):
                participant=loadCSV("cohort_data/%s"%filename)
                if not participant: continue
                participants.append(participant)
        return participants



def normalise(participant, time=True):
        '''Constructs the normalised path for a given stream of data, as
        described on the original paper.

        Arguments:
                participant (Participant): Participant whose data will
                be normalised.

                time (bool): If True, time is also normalised.
        '''

        t0=participant.data[0][0]
        t1=participant.data[-1][0]
        scoreMAX=7
        scoreMIN=1
        cum=[0 for i in range(len(participant.data[0]))]
        for i in range(len(participant.data)):
                participant.data[i]=list(participant.data[i])
                for j in range(len(participant.data[i])):
                        if j==0:
                                if time:
                                        # Normalise time
                                        participant.data[i][j]-=t0
                                        participant.data[i][j]/=(t1-t0)
                        else:
                                # Normalise scores
                                participant.data[i][j]-=scoreMIN
                                participant.data[i][j]*=2/float(scoreMAX-scoreMIN)
                                participant.data[i][j]-=scoreMIN
                                cum[j]+=participant.data[i][j]
                                participant.data[i][j]=cum[j]
                participant.data[i]=tuple(participant.data[i])
        return participant

def buildData(size, training=0.7, group = None):
        '''Builds the training and out of sample sets

        Arguments:
                size (int): Size of each bucket (number of
                obserevations)

                training (float): Percentage of the data that
                will be used to train the model.
        '''

        patients=loadParticipants()
        data=[]
        for patient in patients:
                if group is not None and patient.diagnosis != group:
                        continue

                for i in range(0, len(patient.data)-size, size):
                        p=Participant(patient.data[i:i+size], patient.idNumber, patient.diagnosis, patient.data[i+size])
                        data.append(normalise(p))
        random.shuffle(data)
        training_set=data[0:int(training*len(data))]

        out_of_sample=data[int(training*len(data)):len(data)]
        return training_set, out_of_sample
