#!/usr/bin/env python
"""Code from the paper "A signature-based machine
learning model for bipolar disorder and borderline
personality disorder".

Given a participant of the study, trains the model
using all other participants from the cohort
in order to test the model with this participant
then. This provides three non-negative numbers
(p_1, p_2, p_3) with p_1 + p_2 + p_3 = 1, where p_i
indicates the number of 20-observations buckets that
was classified as belonging to the clinical group i.

This is done for all participants of the cohort,
in order to plot a heat map on a triangle then.
"""

import argparse
import datetime
import numpy as np
import scipy
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from esig import tosig
import matplotlib.pyplot as plt
import math
import os
import pickle
import random
import shutil
from tqdm import tqdm
import psychiatry
from logger import Logger


__author__ = "Imanol Perez Arribas"
__credits__ = ["Imanol Perez Arribas", "Guy M. Goodwin", "John R. Geddes",
                "Terry Lyons", "Kate E. A. Saunders"]
__version__ = "1.0.1"
__maintainer__ = "Imanol Perez Arribas"
__email__ = "imanol.perez@maths.ox.ac.uk"


class Model:
    def __init__(self, reg):
        """The signature-based machine learning model introduced
        in the original paper.

        Parameters
        ----------
            reg : RandomForestRegressor
                The trained random forest regressor.

        """

        self.reg=reg

    def test(self, path, order=2, is_sig=False):
        """Tests the model against a particular participant.

        Parameters
        ----------
        path : str
            Path of the pickle file containing the streams
            of data from the participant.
        order : int, optional
            Order of the signature.
            Default is 2.
        is_sig : bool, optional
            Whether the test set files contain signatures.
            Default is false, in which case conversion to signatures will be carried out here.
            
        Returns
        -------
        list
            3-dimensional vector indicating how often the participant
            has buckets that were classified in each clinical group.

        """

        # We load the pickle file of the participant
        file = open(path,'rb')
        collection = pickle.load(file)
        file.close()

        # Each clinical group is assigned a point
        # on the plane, which was found using cross-validation.

        threshold = np.array([[1, 0],                          # Borderline participants
                             [0, 1],                           # Healthy participants
                             [-1/np.sqrt(2), -1/np.sqrt(2)]])  # Bipolar participants



        # We construct the inputs and outputs to test the model
        x=[]
        y=[]

        for X in collection:
                # The input is the signature of the normalised path
                if is_sig:
                    # If using synthetic data, the input is already a signature
                    x.append(X.data)
                else:
                    # If using the original data, we convert the normalised path into the signature here
                    x.append(tosig.stream2sig(np.array(X.data), order))

                # The function f returns the point for the corresponding
                # clinical group
                y.append(threshold[X.diagnosis])

        # We find the predictions corresponding to the computed inputs
        predicted = self.reg.predict(x)

        # We find which group the predictions belong to, and
        # store how often the participant belongs to each group
        vector = np.zeros(3)
        for i in range(len(x)):
            threshold2 = [tuple(l) for l in threshold.tolist()]
            vector[threshold2.index(tuple(_findMin(predicted[i], threshold)))] += 1
        
        vector /= float(len(x))

        return vector




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



def getCategory(id):
    """Finds the clinical group a given participant belongs to.

    Parameters
    ----------
    id : int
        ID of the participant.
    
    Returns
    -------
    str
        Clinical group that the participant with the given
        ID belongs to.

    """


    file = open("data/"+str(id)+"/os.obj",'rb')
    collection = pickle.load(file)
    file.close()

    categories = ["borderline", "healthy", "bipolar"]

    return categories[collection[0].diagnosis]


def train(path, order=2, is_sig=False):
    """Trains the model, as specified in the original paper.

    Parameters
    ----------
    path : str
        Path of the training set.
    order : int, optional
        Order of the signature.
        Default is 2.
    is_sig : bool, optional
        Whether the test set files contain signatures.
        Default is false, in which case conversion to signatures will be carried out here.

    Returns
    -------
    Model
        Trained model.

    """


    file = open(path,'rb')
    collection = pickle.load(file)
    file.close()


    # Each clinical group is associated with a point on the
    # plane. These points were found using cross-validation.
    threshold = np.array([[1, 0],    # Point for borderline participants
                        [0, 1],   # Point for healthy participants
                        [-1/np.sqrt(2), -1/np.sqrt(2)]]) # Point for bipolar participants




    # x will contain the inputs of the model, while y
    # will contain the outputs.
    x = []
    y = []

    for participant in collection:
        # The input will be the signature of the stream of
        # the participant.
        if is_sig:
            # If using synthetic data, the data is already stored as a signature
            x.append(participant.data)
        else:
            # If using the original data, we convert the normalised path into the signature here
            x.append(tosig.stream2sig(np.array(participant.data), order))


        # The output, on the other hand, will be te point
        # on the plane corresponding to the clinical group
        # of the participant.
        y.append(threshold[participant.diagnosis])

    # We train the model using Random Forests.
    reg = RandomForestRegressor(n_estimators=100)
    reg.fit(x, y)

    # Return the trained model.
    return Model(reg)



def plotDensityMap(scores, plot_name):
    """Plots, given a set of scores, the density map on a triangle.

    Parameters
    ----------
    scores : list
        List of scores, where each score is a 3-dimensional list.
    plot_name : string
        Name given to the saved plot
    """


    TRIANGLE = np.array([[math.cos(math.pi*0.5), math.sin(math.pi*0.5)],
                        [math.cos(math.pi*1.166), math.sin(math.pi*1.166)],
                        [math.cos(math.pi*1.833), math.sin(math.pi*1.833)]])


    pointsX = [score.dot(TRIANGLE)[0] for score in scores]
    pointsY = [score.dot(TRIANGLE)[1] for score in scores]

    vertices = []
    vertices.append(np.array([1,0,0]).dot(TRIANGLE))
    vertices.append(np.array([0,1,0]).dot(TRIANGLE))
    vertices.append(np.array([0,0,1]).dot(TRIANGLE))
    for i in range(3):
        p1 = vertices[i]
        if i == 2:
            p2 = vertices[0]
        else:
            p2 = vertices[i+1]
        c = 0.5 * (p1 + p2)
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='k', linestyle='-', linewidth=2)
        plt.plot([0, c[0]], [0, c[1]], color='k', linestyle='-', linewidth=1)



    ax = plt.gca()
    ax.set_xlim([-1.2, 1.32])
    ax.set_ylim([-0.7,1.3])

    ax.text(0.8, -0.6, 'Bipolar')
    ax.text(-1.1, -0.6, 'Healthy')
    ax.text(-0.15, 1.05, 'Borderline')


    data = [[pointsX[i], pointsY[i]] for i in range(len(pointsX))]

    H, _, _=np.histogram2d(pointsX,pointsY,bins=40,normed=True)
    norm=H.sum()
    contour1=0.75
    target1=norm*contour1
    def objective(limit, target):
        w = np.where(H>limit)
        count = H[w]
        return count.sum() - target

    level1 = scipy.optimize.bisect(objective, H.min(), H.max(), args=(target1,))
    levels = [level1]

    data = np.array(data)

    sns.kdeplot(np.array(pointsX), np.array(pointsY), shade=True, ax=ax)
    sns.kdeplot(np.array(pointsX), np.array(pointsY), n_levels=3, ax=ax, cmap="Reds")

    file_name = plot_name + datetime.datetime.now().strftime("-%Y-%m-%d-%H:%M") + ".png"
    plt.savefig(file_name)
    plt.clf()
    if os.path.isfile(file_name):
        logger.log("Saved plot as {}".format(file_name))
    else:
        logger.log("Could not save file {}".format(file_name))


def export(l, i, data_prepared=False, is_test=False):
    """Saves as a pickle file the training or testing sets.

    Parameters
    ----------
    l : list
        List of participants that should be exported. If the
        length of the list is 1, the set is the out-of-sample
        set. Otherwise, it is the training set.
    i : int
        A random ID that will be used to export the file.
        
    """

    size=20 # Number of observations of each stream of data

    if not os.path.exists("data/"+str(i)):
            os.makedirs("data/"+str(i))

    if len(l)==1 or (data_prepared and is_test):
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
    if data_prepared:
        dataset = l
    else:
        for participant in l:
            for v in range(0, len(participant.data)-size, size):
                p = psychiatry.Participant(participant.data[v:v+size],
                                            participant.idNumber,
                                            participant.diagnosis,
                                            participant.data[v+size])
                dataset.append(psychiatry.normalise(p))

    # Export the dataset.
    filehandler = open("data/"+str(i)+"/"+setType+".obj","wb")
    pickle.dump(dataset,filehandler)
    filehandler.close()

def get_folders(a_dir):
    """Finds all folders in a directory.

    Parameters
    ----------
    a_dir : str
        Directory path.

    Returns
    -------
    list of str
        List of all folders in the directory.

    """

    return [name for name in sorted(os.listdir(a_dir))
            if os.path.isdir(os.path.join(a_dir, name))]

def load_and_export_cohort():
    """ Loads cohort data and exports two files for each participant into a folder with a random ID.

    One file contains the test data, i.e. the normalised buckets of mood score data from that patient, and the second
    contains the training data, i.e. the normalised buckets of mood score data from all other patients.
    """

    # We load all participants in the study
    print("Loading cohort...")
    cohort = psychiatry.loadParticipants("../data")

    # Number of observations of each stream of data
    size = 20

    # Only consider participants that provided at least 5 buckets of data
    valid_participants = [participant for participant in cohort if len(participant.data)>5*size]

    print("Exporting participants...")
    for ref_participant in tqdm(valid_participants):
        # Use participant for testing
        test_participant=[ref_participant]

        # Use the remaining participants for training.
        train_participants=[participant for participant in cohort if participant!=ref_participant]

        # Check that ref_participant is not in train_participants
        assert ref_participant not in train_participants

        # Save the testing and training sets as a file
        random_id = random.randint(0, 1e8)
        export(test_participant, random_id)
        export(train_participants, random_id)

def load_and_export_synthetic_cohort(cohort):
    """ Loads cohort synthetic data and exports two files for each 'participant' into a folder with a random ID.

    The synthetic data does not correspond to specific participants, so here we choose to define a 'participant' as
    a group of synthetic signatures, with the group size set by the variable buckets_per_participant.

    One file contains the test data, i.e. the synthetic signatures from that 'participant', and the second contains the
    training data, i.e. the synthetic signatures from all other 'participants'.

     Parameters
     ----------
     cohort: int
        ID of the synthetic cohort to be analysed
    """

    # Load all synthetic signatures and diagnoses
    signatures = np.genfromtxt(os.path.join("..", "data", "synthetic_signatures",
                                            "cohort_" + str(cohort) + "_sigs.pickle"), delimiter=',')
    diagnoses = np.genfromtxt(os.path.join("..", "data", "synthetic_signatures",
                                           "cohort_" + str(cohort) + "_diagnosis.pickle"), delimiter=',')

    # We don't have distinguishable participants in the synthetic dataset, so we'll consider groups of seven signatures
    # as having come from each participant for now
    buckets_per_participant = 7

    # Work out how many buckets of data we have for each diagnosis, and how many "patients" we can generate
    diag_ids, diag_counts = np.unique(diagnoses, return_counts=True)
    bucket_counts = [int(c) for c in diag_counts]
    patient_counts = [int(c/buckets_per_participant) for c in bucket_counts]

    # Create list of participants. We take all signatures associated with a particular diagnosis,
    # then construct participants by taking seven of these signatures at a time (a couple remain unused)
    participants = []
    for d_ind, d in enumerate(diag_ids):
        signatures_d = signatures[diagnoses == d]

        print("Diagnosis {}: {} buckets of data available to create {} patients".format(d, bucket_counts[d_ind],
                                                                                        patient_counts[d_ind]))

        single_participant = []
        for s_ind, s in enumerate(signatures_d):
            p_id = sum(diag_counts[:d_ind]/buckets_per_participant) + s_ind/buckets_per_participant - 1
            if len(single_participant) < buckets_per_participant:
                single_participant.append(psychiatry.Participant(s, p_id, int(d+1), None))
            else:
                participants.extend(single_participant)
                single_participant = [psychiatry.Participant(s, p_id, int(d+1), None)]

        print("{} buckets of data were not used".format(len(single_participant)))

    # Check that we put the correct number of unique buckets into "Participant" form
    assert(len(participants) == sum(patient_counts)*buckets_per_participant)

    # Export train and test sets for each participant
    for id in range(0, sum(patient_counts)-1, 1):

        random_id = random.randint(0, 1e8)
        test_participant = [p for p in participants if p.idNumber == id]
        train_participants = [p for p in participants if p.idNumber != id]

        export(train_participants, random_id, data_prepared=True, is_test=False)
        export(test_participant, random_id, data_prepared=True, is_test=True)


if __name__ == "__main__":
    """Plots a triangle and the density map of the proportion
    of periods of time participants with a specific diagnosis
    spend in each clinical category.
    """

    """
    Step 0

    Set up command line argument parsers
    --seed (optional): Sets the random seed; default is the original value used in this script.
    --synth (optional): If not specified at all, mood score data is loaded.
                        If --synth is given without a value, cohort 772192 (synthetic signatures) is loaded.
                        If a value for --synth is given, the specified cohort of synthetic signatures is loaded.
    """
    parser = argparse.ArgumentParser(
        description="Plot the time participants spend in each clinical category (healthy, borderline and bipolar)")
    parser.add_argument("--seed", type=int, default=1,
        help="seed for the random number generators (int, default=1)")
    parser.add_argument("--synth", nargs="?", type=int, const=772192,
        help="ID of cohort of synthetic mood score signatures, if they are to be used (int, default=772192 if --synth \
              alone is provided, or None (i.e. load original mood score data) if not)")
    args = parser.parse_args()

    logger = Logger("heat_map")

    # Set the random seeds and report
    random.seed(args.seed)
    np.random.seed(args.seed)
    logger.log("Random seed has been set to {}\n".format(args.seed))

    #  Clean up the data folder (only delete files + folders produced by earlier runs of heat_map.py)
    folders = get_folders("data/")
    for folder in folders:
        contents = os.listdir(os.path.join("data/", folder))
        if folder.isdigit() and "ts.obj" in contents and "os.obj" in contents:
            shutil.rmtree(os.path.join("data/", folder))


    """
    Step 1

    Load the cohort. Then, take each participant
    and create a testing set (using the participant)
    and a training set (using the rest of the
    cohort) and save it as a file.
    """

    if args.synth is None:
        use_synth_sig = False
        logger.log("Preparing to load mood score data...")
        load_and_export_cohort()
        logger.log("Loaded and exported cohort\n")
    else:
        use_synth_sig = True
        logger.log("Preparing to load synthetic signatures from cohort {}...".format(args.synth))
        load_and_export_synthetic_cohort(args.synth)
        logger.log("Loaded and exported synthetic cohort {}\n".format(args.synth))



    """
    Step 2

    For each participant in the clinical group we are interested in,
    test the model with data from this participant. The model is
    trained using the remaining participants in the cohort.

    """



    folders = get_folders("data/")
    scores = []
    logger.log("Calculating points...")
    for folder in tqdm(folders):
        # Train the model
        model=train("data/"+folder+"/ts.obj", is_sig=use_synth_sig)

        # Test the model
        score=model.test("data/"+folder+"/os.obj", is_sig=use_synth_sig)

        # Save the score
        scores.append((folder, score))



    """
    Step 3

    Assign each score to the corresponding clinical group

    """

    trianglePoints={
                        "bipolar":     [],
                        "healthy":     [],
                        "borderline":  []
                    }

    logger.log("Assigning scores...")
    for id, score in scores:
        category = getCategory(id)
        trianglePoints[category].append(score)



    """
    Step 4

    Plot the triangle and the density map.

    """

    logger.log("Generating plots...")
    plotDensityMap(trianglePoints["bipolar"], "bipolar-heatmap")
    plotDensityMap(trianglePoints["healthy"], "healthy-heatmap")
    plotDensityMap(trianglePoints["borderline"], "borderline-heatmap")
