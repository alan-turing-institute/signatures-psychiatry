import csv
import numpy as np
import psychiatry
import matplotlib.pyplot as plt
from matplotlib import gridspec

def plot_data(path):
    try:
        data = list(csv.reader(open(path)))
    except:
        raise RuntimeError("Could not load the data.")

    data = [(psychiatry.string2datenum(l[0].split(".")[0], "%Y-%m-%d %H:%M:%S"),
             int(l[1]), int(l[2]), int(l[3]), int(l[4]), int(l[5]), int(l[6])) for l in data]

    participant = psychiatry.Participant(data, 0, None, [])
    raw_stream = np.array(participant.data)
    participant = psychiatry.normalise(participant)

    stream = np.array(participant.data)

    time = np.linspace(0, 1, len(data))
    

    plt.figure(figsize=(6, 6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    plt.subplot(gs[0])
    plt.plot(time, stream[:, 1], "r")
    plt.ylabel("Anxiety")
    plt.xticks([])
    
    plt.subplot(gs[1])
    plt.plot(time, raw_stream[:, 1], "b")
    plt.plot(time, [4.] * len(raw_stream), "k--", alpha=0.3)
    plt.yticks([1, 2, 3, 4, 5, 6, 7])
    plt.xticks([0., 0.5, 1.])
    plt.xlabel("Time")
    plt.ylabel("Anxiety")
    plt.show()


    plt.figure(figsize=(8, 8))
    moods = ["Anxious", "Elated", "Sad", "Angry", "Irritable", "Energetic"]
    total = 0
    for i in range(len(moods)):
        for j in range(i + 1, len(moods)):
            total += 1
            plt.subplot(5, 3, total)
            plt.plot(stream[:, i + 1], stream[:, j + 1], "k")
            plt.scatter(stream[0, i + 1], stream[0, j + 1], c="r", s=20)
            plt.xlabel(moods[i])
            plt.ylabel(moods[j])
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_data("data/fake_patient.csv")
