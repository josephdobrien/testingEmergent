from mesa import Agent, Model
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector
import random
import numpy as np
import pandas as pd
import networkx as nx
import scipy.stats

import model as md


import argparse
parser = argparse.ArgumentParser('TRACKSIM')
parser.add_argument('--m', required=True, type=str, help='m: model type (either random or patient)')
parser.add_argument('--s', required=True, type=int, help='s: number of simulations')
parser.add_argument('--r', required=True, type=int, help='r: rounds in each simulation')
args = parser.parse_args()

runs = args.s
rounds = args.r
model_type = args.m

main = pd.DataFrame() #initialize dataframe for tracking results

if model_type == 'random': 
##############
###BASELINE###
##############
    for run in range(runs):
        t = random.choice(np.round(np.arange(0, 1.001, 1/5), 2))
        f = 1
        m = md.RandomModel(t, f)
        for i in range(rounds):
            m.step()
        df = m.datacollector.get_agent_vars_dataframe()
        df['truth'] = t
        df['f_rate'] = f
        df['run'] = run
        main = pd.concat([main, df])

###################
###MORE_PATIENCE###
###################

elif model_type == 'patient':

    for run in range(runs):
        t = random.choice(np.round(np.arange(0, 1.001, 1/5), 2))
        f = 1
        m = md.PatientModel(t, f)
        for i in range(50):
            m.step()
        # Begin assessing peers after 50 time steps
        m.activate()
        for i in range(int(rounds-50)): #remaining number of rounds (e.g. rounds == 100, run loop for additional 50 rounds)
            m.step()
        df = m.datacollector.get_agent_vars_dataframe()
        df['truth'] = t
        df['f_rate'] = f
        df['run'] = run
        main = pd.concat([main, df])



main.dropna(inplace=True)
main.to_csv('test.csv')