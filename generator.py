import pickle
from esig import tosig
from tqdm import tqdm
import numpy as np
import os
import argparse
import random

import cvae
import psychiatry
import TensorAlgebra as TA


class Generator:
    def __init__(self, n_latent=14, seed=None):
        # Load data
        data_raw = psychiatry.buildData(20, "../data", training=1)
        data = np.array([tosig.stream2logsig(np.array(participant.data), 2) for participant in
                         tqdm(data_raw[0], desc="Loading data")])
        self.data_sig = np.array([tosig.stream2sig(np.array(participant.data), 2) for participant in
                                  tqdm(data_raw[0], desc="Computing signatures")])

        # We get the diagnosis:
        #   (1, 0, 0) -> Healthy
        #   (0, 1, 0) -> Bipolar
        #   (0, 0, 1) -> Borderline

        self.diagnosis = np.array([np.eye(3)[participant.diagnosis] for participant in data_raw[0]])

        self.M = np.max(data, axis=0).reshape(1, -1)
        self.m = np.min(data, axis=0).reshape(1, -1)
        self.m[self.M == self.m] = 0.

        # Normalise the data
        self.data = (data - self.m) / (self.M - self.m)
        
        self.generator = cvae.CVAE(n_latent=n_latent, seed=seed)

    def _flatten(self, ta):
        return np.concatenate([ta[i].flatten() for i in range(ta.order() + 1)])        
        
    def train(self, n_epochs=10000):
        self.generator.train(self.data, self.diagnosis, n_epochs=n_epochs)
        
    def generate(self, n_healthy=797, n_bipolar=851, n_borderline=544):

        DIAGNOSIS = {
                        (1, 0, 0): "Healthy",
                        (0, 1, 0): "Bipolar",
                        (0, 0, 1): "Borderline"
                    }
        
        n_samples = {
                        "Bipolar":     n_bipolar,
                        "Healthy":     n_healthy,
                        "Borderline":  n_borderline
                    }
        
        generated = {}
        
        for group in np.eye(3):
            group_name = DIAGNOSIS[tuple(group)]
            group_idx = (self.diagnosis == group).all(axis=1)

            data_diagnosis = self.data[group_idx]

            data_diagnosis_sig = self.data_sig[group_idx]
            generated_samples = self.generator.generate(cond=group, n_samples=n_samples[group_name])
            generated_samples = generated_samples * (self.M - self.m) + self.m
            
            logsig_ta = [TA.logsig2tensor(logsig, 7, 2) for logsig in generated_samples]
            sig_ta = [TA.exp(ta, max_n=3).truncate(2) for ta in logsig_ta]
            sigs = np.array(list(map(self._flatten, sig_ta)))
        
            generated[group_name] = sigs
            
            
        return generated
    
def run(directory="generated_data/", **kwargs):
    # Create a random cohort ID number
    cohort_id = np.random.randint(100000, 999999)
    
    generator = Generator()
    generator.train()
    generated = generator.generate(**kwargs)
    
    # Generate dataset: inputs
    X = np.r_[generated["Healthy"], generated["Bipolar"], generated["Borderline"]]
    
    # Generate dataset: outputs
    healthy_labels = [0]*len(generated["Healthy"])
    bipolar_labels = [1]*len(generated["Bipolar"])
    borderline_labels = [-1]*len(generated["Borderline"])
    Y = np.r_[healthy_labels, bipolar_labels, borderline_labels]
    
    # Export dataset
    np.savetxt(os.path.join(directory, "cohort_{}_sigs.pickle".format(cohort_id)),
               X, delimiter=",")
    np.savetxt(os.path.join(directory, "cohort_{}_diagnosis.pickle".format(cohort_id)),
               Y, delimiter=",")
    

    print("{} healthy, {} bipolar and {} borderline streams of data generated. Simulated cohort ID: {}.".format(len(healthy_labels), len(bipolar_labels), len(borderline_labels), cohort_id))
          
    
if __name__ == "__main__":
    parser=argparse.ArgumentParser()

    parser.add_argument('--healthy', help='Number of healthy streams')
    parser.add_argument('--bipolar', help='Number of bipolar streams')
    parser.add_argument('--borderline', help='Number of borderline streams')

    args=parser.parse_args()
    
    kwargs = {}
    
    if args.healthy:
        kwargs["n_healthy"] = int(args.healthy)
    if args.healthy:
        kwargs["n_bipolar"] = int(args.bipolar)
    if args.healthy:
        kwargs["n_borderline"] = int(args.borderline)
    
    
    run(**kwargs)