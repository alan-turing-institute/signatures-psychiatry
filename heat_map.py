#!/usr/bin/env python
"""Code from the paper "A signature-based machine
learning model for bipolar disorder and borderline
personality disorder".

Given a participant of the study, trains the model
using all other participants from the cohort
in order to test the model with this participant
then. This provides three numbers (p_1, p_2, p_3)
with p_1 + p_2 + p_3 = 1, where p_i indicates
the number of 20-observations buckets that was
classified as belonging to the clinical group i.

This is done for all participants of the cohort,
in order to plot a heat map on a triangle then.
"""

import numpy as np
import scipy
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from esig import tosig
import matplotlib.pyplot as plt
import csv
import math
import os
import pickle
import random
from tqdm import tqdm
from Psychiatry import *

__author__ = "Imanol Perez Arribas"
__credits__ = ["Imanol Perez Arribas", "Guy M. Goodwin", "John R. Geddes",
                    "Terry Lyons", "Kate E. A. Saunders"]
__version__ = "1.0.1"
__maintainer__ = "Imanol Perez"
__email__ = "imanol.perez@maths.ox.ac.uk"


class Model:
        def __init__(self, reg):
                """The signature-based machine learning
                model introduced in the original paper.

                Args:
                    reg (RandomForestRegressor): The trained random forest
                    regressor.

                """

                self.reg=reg

        def test(self, path, order=2):
                """Tests the model against a particular participant.

                Args:
                    path (str): Path of the pickle file containing the streams
                    of data from the participant.

                    order (int): Signature order.

                Returns:
                    list: 3-dimensional vector indicating how often the participant
                    has buckets that were classified in each clinical group.

                """

                # We load the pickle file of the participant
                file = open(path,'rb')
                collection = pickle.load(file)
                file.close()

                # Each clinical group is assigned a point
                # on the plane, which was found using crossvalidation.

                threshold=np.array([[1, 0],    # Point for borderline participants
                                    [0, 1],   # Point for healthy participants
                                    [-1/np.sqrt(2), -1/np.sqrt(2)]]) # Point for bipolar participants



                # We construct the inputs and outputs to test the model
                x=[]
                y=[]

                for X in collection:
                        # The input is the signature of the normalised path
                        x.append(list(tosig.stream2sig(np.array(X.data), order)))

                        # The function f returns the point for the corresponding
                        # clinical group
                        y.append(f(X.diagnosis, threshold=threshold))

                # We find the predictions corresponding to the computed inputs
                predicted=self.reg.predict(x)

                # We find which group the predictions belong to, and
                # store how often the participant belongs to each group
                vector=np.zeros([3])
                for i in range(len(x)):
                        threshold2=[tuple(l) for l in threshold.tolist()]
                        #prediction = np.argmax(predicted[i])
                        #vector[prediction] += 1
                        vector[threshold2.index(tuple(findMin(predicted[i], threshold)))]+=1
                vector/=float(len(x))

                return vector




def f(bipolar, threshold):
        """Maps a clinical group to the corresponding point on the
        plane.

        Args:
            bipolar (int): Clinical group. Borderline if 0, healthy if 1
            and bipolar if 2.

            threshold (list): List of 3 points on the plane.

        Returns:
            2-d tuple: Point on the plane corresponding to the clinical group.

        """

        return threshold[bipolar]


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

def getCategory(id):
        """Finds the clinical group a given participant belongs to.

        Args:
            id (int): ID of the participant.

        Returns:
            str: Clinical group that the participant with the
            given ID belongs to.

        """


        file = open("data/"+str(id)+"/os.obj",'rb')
        collection = pickle.load(file)
        file.close()

        categories=["borderline", "healthy", "bipolar"]

        return categories[collection[0].diagnosis]


def train(path, order=2):
        """Trains the model, as specified in the original paper.

        Args:
            path (str): Path of the training set.

        Returns:
            Model: Trained model.

        """


        file = open(path,'rb')
        collection = pickle.load(file)
        file.close()


        # Each clinical group is associated with a point on the
        # plane. These points were found using cross-valiation.
        threshold=np.array([[1, 0],    # Point for borderline participants
                            [0, 1],   # Point for healthy participants
                            [-1/np.sqrt(2), -1/np.sqrt(2)]]) # Point for bipolar participants




        # x will contain the inputs of the model, while y
        # will contain the outputs.
        x=[]
        y=[]

        for participant in collection:
                # The input will be the signature of the stream of
                # the participant.
                x.append(list(tosig.stream2sig(np.array(participant.data), order)))


                # The output, on the other hand, will be te point
                # on the plane corresponding to the clinical group
                # of the participant.
                y.append(f(participant.diagnosis, threshold=threshold))

        # We train the model using Random Forests.
        reg = RandomForestRegressor(n_estimators=100)
        reg.fit(x, y)

        # Return the trained model.
        return Model(reg)



def plotDensityMap(scores):
        """Plots, given a set of scores, the density map
        on a triangle.

        Args:
            scores (list): List of scores, where each score
            is a 3-dimensional list.

        """


        TRIANGLE=np.array([[math.cos(math.pi*0.5), math.sin(math.pi*0.5)],
                           [math.cos(math.pi*1.166), math.sin(math.pi*1.166)],
                           [math.cos(math.pi*1.833), math.sin(math.pi*1.833)]])


        pointsX=[score.dot(TRIANGLE)[0] for score in scores]
        pointsY=[score.dot(TRIANGLE)[1] for score in scores]

        vertices=[]
        vertices.append(np.array([1,0,0]).dot(TRIANGLE))
        vertices.append(np.array([0,1,0]).dot(TRIANGLE))
        vertices.append(np.array([0,0,1]).dot(TRIANGLE))
        for i in range(3):
                p1=vertices[i]
                if i==2:
                        p2=vertices[0]
                else:
                        p2=vertices[i+1]
                c=0.5*(p1+p2)
                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='k', linestyle='-', linewidth=2)
                plt.plot([0, c[0]], [0, c[1]], color='k', linestyle='-', linewidth=1)



        ax=plt.gca()
        ax.set_xlim([-1.2, 1.32])
        ax.set_ylim([-0.7,1.3])

        ax.text(0.8, -0.6, 'Bipolar')
        ax.text(-1.1, -0.6, 'Healthy')
        ax.text(-0.15, 1.05, 'Borderline')


        data=[[pointsX[i], pointsY[i]] for i in range(len(pointsX))]

        H, xedges, yedges=np.histogram2d(pointsX,pointsY,bins=40,normed=True)
        norm=H.sum()
        contour1=0.75
        target1=norm*contour1
        def objective(limit, target):
                w=np.where(H>limit)
                count=H[w]
                return count.sum()-target

        level1=scipy.optimize.bisect(objective, H.min(), H.max(), args=(target1,))
        levels=[level1]

        data=np.array(data)
        #plt.scatter(np.array(pointsX), np.array(pointsY))
        sns.kdeplot(np.array(pointsX), np.array(pointsY), shade=True, ax=ax)
        sns.kdeplot(np.array(pointsX), np.array(pointsY), n_levels=3, ax=ax, cmap="Reds")
        plt.show()


def export(l, i):
        """Saves as a pickle file the training or testing sets.

        Args:
            l (list): List of participants that should be exported.
            If the length of the list is 1, the set is the out of
            sample set. Otherwise, it is the training set.

            i (int): A random ID that will be used to export the file.

        """

        size=20 # Number of observations of each stream of data

        if not os.path.exists("data/"+str(i)):
                os.makedirs("data/"+str(i))

        if len(l)==1:
                # We want to export a single participant for
                # testing.
                setType="os"
        else:
                # We want to export the rest of the cohort
                # for training.
                setType="ts"

        dataset=[]

        # For each participant and each bucket of appropriate size,
        # add the stream of data to dataset.
        for participant in l:
                for v in range(0, len(participant.data)-size, size):
                        p=Participant(participant.data[v:v+size], participant.idNumber, participant.diagnosis, participant.data[v+size])
                        dataset.append(normalise(p))

        # Export the dataset.
        filehandler = open("data/"+str(i)+"/"+setType+".obj","wb")
        pickle.dump(dataset,filehandler)
        filehandler.close()

def get_folders(a_dir):
    """Finds all folders in a directory.

    Args:
        a_dir (str): Directory where folder should be searched.

    Returns:
        list: List of folders in the directory.

    """

    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

if __name__ == "__main__":
        '''

        Plots a triangle and the density map of the proportion
        of periods of time participants with a specific diagnosis
        spend in each clinical category.

        Arguments:
                group: string. Can only take three values: "borderline",
                "healthy" or "bipolar".
        '''

        random.seed(1)
        np.random.seed(1)


        '''
        Step 1

        Load the cohort. Then, take each participant
        and create a testing set (using the participant)
        and a training set (using the rest of the
        cohort) and save it as a file.

        '''

        # We load all participants in the study
        print("Loading cohort...")
        cohort=loadParticipants()

        # Number of observations of each stream of data
        size=20

        # Only consider participants that provided at least 7 buckets of data
        valid_participants=[participant for participant in cohort if len(participant.data)>5*size]

        print("Exporting participants...")
        for ref_participant in tqdm(valid_participants):
                #continue
                # Use participant for testing
                test_participant=[ref_participant]

                # Use the remaining participants for training.
                train_participants=[participant for participant in cohort if participant!=ref_participant]

                # Check that ref_participant is not in train_participants
                assert ref_participant not in train_participants

                # Save the testing and training sets as a file
                random_id=random.randint(0, 1e8)
                export(test_participant, random_id)
                export(train_participants, random_id)


        '''
        Step 2

        For each participant in the clinical group we are interested in,
        test the model with data from this participant. The model is
        trained using the remaining participants in the cohort.

        '''




        folders=get_folders("data/")
        scores=[]
        print("Calculating points...")
        for folder in tqdm(folders):
                #continue
                # Train the model
                model=train("data/"+folder+"/ts.obj")

                # Test the model
                score=model.test("data/"+folder+"/os.obj")

                # Save the score
                scores.append((folder, score))



        '''
        Step 3

        Asign each score to the corresponding clinical group

        '''

        trianglePoints={
                            "bipolar": [],
                            "healthy": [],
                            "borderline": []
                        }

        for id, score in scores:
            category=getCategory(id)
            trianglePoints[category].append(score)



        '''
        Step 4

        Plot the triangle and the density map.

        '''


        a = np.random.random(size=(7, 3))
        a = np.diag(1/a.sum(axis=1)).dot(a)
        plotDensityMap(trianglePoints["bipolar"])
        plotDensityMap(trianglePoints["healthy"])
        plotDensityMap(trianglePoints["borderline"])
