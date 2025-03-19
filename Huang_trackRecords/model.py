from mesa import Agent, Model
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector
import random
import numpy as np
import pandas as pd
import networkx as nx
import scipy.stats

import agent as ag


class TRModel(Model):
    def __init__(self, truth, feedback_rate):
        self.truth = truth
        self.schedule = SimultaneousActivation(self)
        self.fbr = feedback_rate # How often are track records public? Always 1 in current model.
        self.schedule = SimultaneousActivation(self)
        for i in range(30):
            self.schedule.add(ag.TRScientist(i, self))
        self.datacollector = DataCollector(
            agent_reporters={"pr": "pr", "ev": "ev",
                             "id": "id", "hub": "hub", "authority": "authority",
                             "Brier": "Brier", "BrierT": "BrierT", "crps": "crps",
                             "neighbors": get_neighbors, 
                             "c": "c", "m": "m", "noise": "noise",
                             "cred": "cred", "social": "social", "evidential": "evidential"},
            model_reporters={"truth": "truth"})
    def centrality(self):
        G = nx.DiGraph()
        for a in self.schedule.agents:
            G.add_node(a)
            for n in a.neighbors:
                if a != n:
                    G.add_edge(a, n)
        idc = nx.in_degree_centrality(G)
        evc = nx.eigenvector_centrality_numpy(G)
        pr = nx.pagerank(G)
        hub, authority = nx.hits(G)
        for a in self.schedule.agents:
            a.pr = round(pr[a], 4)
            a.ev = round(evc[a], 4)
            a.id = round(idc[a], 4)
            a.hub = round(hub[a], 4)
            a.authority = round(authority[a], 4)
    def step(self):
        self.schedule.step()
        if np.random.binomial(1, self.fbr):
            for a in self.schedule.agents:
                a.record.append(a.Brier[-1])
        self.centrality()
        self.datacollector.collect(self)

###HELPER FUNCTIONS###
def get_neighbors(agent):
    return [a.unique_id for a in agent.neighbors]

##################
###RANDOM MODEL###
##################
class RandomModel(TRModel):
    def __init__(self, truth, feedback_rate):
        super().__init__(truth, feedback_rate)
        self.schedule = SimultaneousActivation(self)
        for i in range(15):
            self.schedule.add(ag.TRScientist(i, self))
        for i in range(15, 30):
            self.schedule.add(ag.RandomScientist(i, self))
        self.datacollector = DataCollector(
            agent_reporters={"pr": "pr", "ev": "ev", 
                             "id": "id", "hub": "hub", "authority": "authority",
                             "Brier": "Brier", "BrierT": "BrierT", "crps": "crps",
                             "neighbors": get_neighbors, 
                             "c": "c", "m": "m", "noise": "noise",
                             "cred": "cred", "social": "social", "evidential": "evidential"},
            model_reporters={"truth": "truth"})

###################      
###PATIENT MODEL###
###################
class PatientModel(TRModel):
    def __init__(self, truth, feedback_rate):
        super().__init__(truth, feedback_rate)
        self.schedule = SimultaneousActivation(self)
        for i in range(30):
            self.schedule.add(ag.PatientScientist(i, self))
        self.datacollector = DataCollector(
            agent_reporters={"pr": "pr", "ev": "ev", 
                             "id": "id", "hub": "hub", "authority": "authority",
                             "Brier": "Brier", "BrierT": "BrierT", "crps": "crps",
                             "neighbors": get_neighbors, 
                             "c": "c", "m": "m", "noise": "noise",
                             "cred": "cred", "social": "social", "evidential": "evidential"},
            model_reporters={"truth": "truth"})
    def activate(self):
        for a in self.schedule.agents:
            a.c = round(random.random(), 2)

