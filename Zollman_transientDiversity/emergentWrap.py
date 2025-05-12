from modelpy_abm.main import AgentModel
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import stats 
import random
from typing import List, Tuple, Dict, Optional
import time

class Scientist: #agent learns via Bayesian updating w/ beta distributions

    def __init__(self, alpha1: float, beta1: float, alpha2: float, beta2: float): #initialize priors
        
        self.alpha1 = alpha1 #parameters for the beta distribution of methodology 1
        self.beta1 = beta1
        self.alpha2 = alpha2 #parameters for the beta distribution of methodology 2
        self.beta2 = beta2
        
    def expected_values(self) -> Tuple[float, float]: #calculate EV for each method
        return (self.alpha1 / (self.alpha1 + self.beta1), 
                self.alpha2 / (self.alpha2 + self.beta2))
    
    def choose_action(self) -> int: #choose method based on EV
        ev1, ev2 = self.expected_values()
        if ev1 > ev2:
            return 0  # Choose methodology 1
        else:
            return 1  # Choose methodology 2
            
    def update_beliefs(self, action: int, successes: int, trials: int): #bayesian updating
        if action == 0:  # Update methodology 1
            self.alpha1 += successes
            self.beta1 += (trials - successes)
        else:  # Update methodology 2
            self.alpha2 += successes
            self.beta2 += (trials - successes)
            
class ZollmanModel(AgentModel):  # Inherit from AgentModel
    def __init__(self, num_agents: int, network_type: str, true_probs: List[float],
                 trials_per_experiment: int = 1000, max_prior_value: float = 4.0,
                 max_iterations: int = 10000):
        super().__init__()  # Initialize the AgentModel
        self.num_agents = num_agents
        self.true_probs = true_probs
        self.trials_per_experiment = trials_per_experiment
        self.max_prior_value = max_prior_value
        self.max_iterations = max_iterations

        # Initialize the graph
        self.initialize_graph()
        if isinstance(network_type, nx.Graph):
            self.network = network_type
        else:
            self.network = self._create_network(network_type)

        # Initialize scientists with random priors
        self.scientists = []
        for _ in range(num_agents):
            alpha1 = np.random.uniform(0, max_prior_value)
            beta1 = np.random.uniform(0, max_prior_value)
            alpha2 = np.random.uniform(0, max_prior_value)
            beta2 = np.random.uniform(0, max_prior_value)
            self.scientists.append(Scientist(alpha1, beta1, alpha2, beta2))

        self.action_history = []
        self.variance_history = []

    def _create_network(self, network_type: str) -> nx.Graph:
        G = nx.Graph()
        G.add_nodes_from(range(self.num_agents))

        if network_type == 'cycle':
            for i in range(self.num_agents):
                G.add_edge(i, (i + 1) % self.num_agents)
        elif network_type == 'wheel':
            for i in range(1, self.num_agents):
                G.add_edge(0, i)
        elif network_type == 'complete':
            for i in range(self.num_agents):
                for j in range(i + 1, self.num_agents):
                    G.add_edge(i, j)

        return G

    def run_experiment(self, agent_id: int, action: int) -> Tuple[int, int]:
        successes = np.random.binomial(self.trials_per_experiment, self.true_probs[action])
        return successes, self.trials_per_experiment

    def run_iteration(self):
        actions = [scientist.choose_action() for scientist in self.scientists]
        self.action_history.append(actions)
        self.variance_history.append(np.var(actions))

        results = []
        for agent_id, action in enumerate(actions):
            successes, trials = self.run_experiment(agent_id, action)
            results.append((agent_id, action, successes, trials))

        for agent_id in range(self.num_agents):
            own_action = actions[agent_id]
            own_results = next((r for r in results if r[0] == agent_id), None)
            if own_results:
                _, _, successes, trials = own_results
                self.scientists[agent_id].update_beliefs(own_action, successes, trials)

            for neighbor in self.network.neighbors(agent_id):
                neighbor_action = actions[neighbor]
                neighbor_results = next((r for r in results if r[0] == neighbor), None)
                if neighbor_results:
                    _, _, successes, trials = neighbor_results
                    self.scientists[agent_id].update_beliefs(neighbor_action, successes, trials)

    def run_simulation(self) -> bool:
        correct_method = np.argmax(self.true_probs)

        for i in range(self.max_iterations):
            self.run_iteration()

            actions = [scientist.choose_action() for scientist in self.scientists]
            if len(set(actions)) == 1:
                return actions[0] == correct_method

        final_actions = [scientist.choose_action() for scientist in self.scientists]
        if len(set(final_actions)) == 1:
            return final_actions[0] == correct_method
        else:
            return False

    def plot_variance_over_time(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.variance_history)
        plt.xlabel('Iterations')
        plt.ylabel('Variance in Actions')
        plt.title('Diversity over Time')
        plt.grid(True)
        plt.show()
