from mesa import Agent, Model
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector
import random
import numpy as np
import pandas as pd
import networkx as nx
import scipy.stats


#################
###AGENT CLASS###
#################
class TRScientist(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.record = [] # Track record
        self.m = round(random.uniform(0.05, 0.5), 2) # Open-mindedness
        self.model = model
        self.unique_id = unique_id
        self.hyp = np.round(np.arange(0, 1.001, 1/5), 2)
        self.cred = np.round(np.full(len(self.hyp), 1/len(self.hyp)), 2) # Credence for each hyp
        self.noise = random.uniform(0.001, 0.2) # equivalent to sigma in paper
        self.c = round(random.random(), 2) # weight for evidence vs testimony
        self.neighbors = [] # trusted informants
        self.social = None 
        self.evidential = None
        self.pr = 0
        self.ev = 0
        self.id = 0
        self.hub = 0
        self.authority = 0
        self.Brier = [] # Prediction at previous time step against new toss
        self.BrierT = None # Cred against truth (God's eye view)
        self.crps = None
    def __hash__(self):
        return hash((self.model, self.unique_id))
    def __eq__(self, other):
        return (self.model, self.unique_id) == (other.model, other.unique_id)
    def r_avg(self):
        if len(self.record) > 0:
            # Mean Brier so far for toss predictions
            return round(sum(self.record)/len(self.record), 4)
        else:
            return 1
    def update_social(self):
        self.social = np.round(sum([a.cred for a in self.neighbors])/len(self.neighbors), 4)
    def update_evidence(self):
        toss = np.random.binomial(1, self.model.truth)
        self.Brier.append(round((toss - sum(self.cred*self.hyp))**2, 4)) # Cred at previous time step against new toss
        Pr_E_H = np.absolute((1-toss)-self.hyp)
        posterior = Pr_E_H*self.cred/np.sum(self.cred*Pr_E_H)
        loc = posterior
        scale = self.noise
        # No neg credence
        noisy = scipy.stats.truncnorm.rvs((0.0001-loc)/scale, (9.9999-loc)/scale, loc=loc, scale=scale)
        # Normalize
        self.evidential = noisy/sum(noisy)
    def update_neighbors(self):
        n = round(len(self.model.schedule.agents)*self.m)
        if n < 1:
            # Agent trust no one
            self.neighbors = [self]
        elif len(self.record) == 0:
            # No track records yet
            self.neighbors = random.sample(self.model.schedule.agents, n)
        else:
            # Choose best performing agents so far
            temp = []
            ls = self.model.schedule.agents
            random.shuffle(ls)
            temp = sorted(ls, key=lambda x: x.r_avg())[:n]
            if len(temp) < 1:
                temp.append(self)
            self.neighbors = temp
    def step(self):
        self.update_evidence()
        self.update_neighbors()
        self.update_social()
    def advance(self):
        # linear combination of social and evidential components
        new_cred = np.round((1-self.c)*self.social + self.c*self.evidential, 2) 
        self.cred = new_cred
        # calculate inaccuracy
        t = np.zeros((len(self.hyp),)) 
        t[int(self.model.truth*5)] = 1 # array of truth value for each hypothesis
        self.BrierT = round(sum((self.cred-t)**2), 4)
        self.crps = crps(self.cred, self.model.truth)


####Helper Functions###
def crps(cred, truth):
    penalty = 0
    for i in range(len(cred)):
        if i<(truth*5):
            penalty += (sum(cred[:i+1])-0)**2
        else:
            penalty += (sum(cred[:i+1])-1)**2
    return(round(penalty, 4))

def Euclidean(x, y):
    return sum((x-y)**2)

######################
###RANDOM SCIENTIST###
######################
class RandomScientist(TRScientist):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
    def update_neighbors(self):
        n = round(len(self.model.schedule.agents)*self.m)
        if n < 1:
            self.neighbors = [self]
        else:
            self.neighbors = random.sample(self.model.schedule.agents, n)

#######################
###PATIENT SCIENTIST###
#######################
class PatientScientist(TRScientist):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.c = 1