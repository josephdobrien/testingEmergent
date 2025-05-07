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


class ZollmanModel: #bandit model class
    
    def __init__(self, 
                 num_agents: int, #number of scientists
                 network_type: str, #'cycle', 'wheel', 'complete', or cutrom graph (NetworkX)
                 true_probs: List[float], #ground truth prob for each method
                 trials_per_experiment: int = 1000, #number of trials
                 max_prior_value: float = 4.0, #max alpha and beta vals for priors
                 max_iterations: int = 10000): #iterations of simulation
        
        self.num_agents = num_agents
        self.true_probs = true_probs
        self.trials_per_experiment = trials_per_experiment
        self.max_prior_value = max_prior_value
        self.max_iterations = max_iterations
        
        if isinstance(network_type, nx.Graph): #create social network
            self.network = network_type
        else:
            self.network = self._create_network(network_type)
            
        self.scientists = [] #initialize scientists with random priors
        for _ in range(num_agents):
            alpha1 = np.random.uniform(0, max_prior_value)
            beta1 = np.random.uniform(0, max_prior_value)
            alpha2 = np.random.uniform(0, max_prior_value)
            beta2 = np.random.uniform(0, max_prior_value)
            self.scientists.append(Scientist(alpha1, beta1, alpha2, beta2))
            
        self.action_history = [] #tracking
        self.variance_history = []
    
    def _create_network(self, network_type: str) -> nx.Graph: #create netowrk of specified type
        G = nx.Graph()
        G.add_nodes_from(range(self.num_agents))
        
        if network_type == 'cycle': #connect neighbors
            for i in range(self.num_agents):
                G.add_edge(i, (i + 1) % self.num_agents)
                
        elif network_type == 'wheel': #connect center (node 0) to all other nodes
            for i in range(1, self.num_agents):
                G.add_edge(0, i)
                
        elif network_type == 'complete': #fully connected graph
            for i in range(self.num_agents):
                for j in range(i + 1, self.num_agents):
                    G.add_edge(i, j)
        
        return G
    
    def run_experiment(self, agent_id: int, action: int) -> Tuple[int, int]:
        
        #sample from binomial distribution with n trials and p probability of success
        successes = np.random.binomial(self.trials_per_experiment, self.true_probs[action])
        return successes, self.trials_per_experiment
    
    def run_iteration(self): #single model iteration
        actions = [scientist.choose_action() for scientist in self.scientists] #action choice for each agent
        
        #store actions for tracking variance
        self.action_history.append(actions)
        self.variance_history.append(np.var(actions))
        
        #run experiment
        results = []
        for agent_id, action in enumerate(actions):
            successes, trials = self.run_experiment(agent_id, action)
            results.append((agent_id, action, successes, trials))
        
        #update beliefs
        for agent_id in range(self.num_agents):
            own_action = actions[agent_id] #update based on agent's experiment
            own_results = next((r for r in results if r[0] == agent_id), None)
            if own_results:
                _, _, successes, trials = own_results
                self.scientists[agent_id].update_beliefs(own_action, successes, trials)
            
            for neighbor in self.network.neighbors(agent_id): #update on neighbor's experiments
                neighbor_action = actions[neighbor]
                neighbor_results = next((r for r in results if r[0] == neighbor), None)
                if neighbor_results:
                    _, _, successes, trials = neighbor_results
                    self.scientists[agent_id].update_beliefs(neighbor_action, successes, trials)
    
    def run_simulation(self) -> bool: #returns True if community converges to correct method
        correct_method = np.argmax(self.true_probs) #find method with highest probability
        
        for i in range(self.max_iterations):
            self.run_iteration()
            
            #check for convergence
            actions = [scientist.choose_action() for scientist in self.scientists]
            if len(set(actions)) == 1:
                #check for convergence to the correct method
                return actions[0] == correct_method
                
        #if convergence fails, check final state
        final_actions = [scientist.choose_action() for scientist in self.scientists]
        if len(set(final_actions)) == 1:
            return final_actions[0] == correct_method
        else:
            #no consensus
            return False
    
    def plot_variance_over_time(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.variance_history)
        plt.xlabel('Iterations')
        plt.ylabel('Variance in Actions')
        plt.title('Diversity over Time')
        plt.grid(True)
        plt.show()


def run_experiment_batch(num_trials: int, #number of trials for each network type
                         network_types: List[str], #list of network types to test
                         num_agents: int, #number of agents in simulation
                         true_probs: List[float], #true probabilities for each method
                         max_prior_value: float = 4.0, #max alpha and beta values, see footnote 10 in Zollman (2010)
                         max_iterations: int = 10000) -> Dict[str, float]: #maxiterations per simulation
 
    results = {network_type: 0 for network_type in network_types}
    
    for network_type in network_types:
        print(f"Running {num_trials} trials for {network_type} network...")
        successes = 0
        for _ in range(num_trials):
            model = ZollmanModel(num_agents=num_agents,
                                network_type=network_type,
                                true_probs=true_probs,
                                max_prior_value=max_prior_value,
                                max_iterations=max_iterations)
            if model.run_simulation():
                successes += 1
        
        success_rate = successes / num_trials
        results[network_type] = success_rate
        print(f"{network_type}: {success_rate:.3f} success rate")
    
    return results


def plot_varying_max_prior(network_types: List[str],  #plot success rate vs. initial priors (figure 6)
                          num_agents: int,
                          true_probs: List[float],
                          max_prior_values: List[float], #test changing maximum priors
                          num_trials: int = 20):
   
    results = {network_type: [] for network_type in network_types}
    
    for max_prior in max_prior_values:
        print(f"Testing with max_prior = {max_prior}")
        batch_results = run_experiment_batch(
            num_trials=num_trials,
            network_types=network_types,
            num_agents=num_agents,
            true_probs=true_probs,
            max_prior_value=max_prior
        )
        
        for network_type in network_types:
            results[network_type].append(batch_results[network_type])
    
    #plot results
    plt.figure(figsize=(12, 8))
    for network_type in network_types:
        plt.plot(max_prior_values, results[network_type], marker='o', label=f"{network_type}")
    
    plt.xlabel('Maximum alpha/beta Value')
    plt.ylabel('Probability of Successful Learning')
    plt.title('Effect of Prior Strength on Learning Success')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_network_densities(max_agents: int = 6, num_trials: int = 20): #plot success rate vs network density (figure 4)
    true_probs = [0.5, 0.499]  #true probabilities from paper
    
    #for networks of size 6, there are 2^15 possible networks (sample a subset)
    densities = []
    success_rates = []
    
    #generate random networks with varying densities
    for _ in range(50):
        #random probability for edge creation
        p = np.random.uniform(0.1, 0.9)
        G = nx.erdos_renyi_graph(max_agents, p)
        
        #ensure connected graph
        if not nx.is_connected(G):
            continue
        
        #calculate density
        density = nx.density(G)
        densities.append(density)
        
        #run simulation
        successes = 0
        for _ in range(num_trials):
            model = ZollmanModel(
                num_agents=max_agents,
                network_type=G,
                true_probs=true_probs
            )
            if model.run_simulation():
                successes += 1
        
        success_rate = successes / num_trials
        success_rates.append(success_rate)
        print(f"Network with density {density:.3f}: {success_rate:.3f} success rate")
    
    #plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(densities, success_rates, alpha=0.7)
    
    #fit a regression line
    z = np.polyfit(densities, success_rates, 1)
    p = np.poly1d(z)
    plt.plot(sorted(densities), p(sorted(densities)), "r--")
    
    plt.xlabel('Density')
    plt.ylabel('Probability of Correct Convergence')
    plt.title('Network Density vs. Learning Success')
    plt.grid(True)
    plt.show()


def compare_diversity_over_time(network_types: List[str], #diversity over time varying network and priors (figure 7)
                               num_agents: int,
                               true_probs: List[float],
                               high_prior_value: float = 7000.0,
                               low_prior_value: float = 4.0,
                               max_iterations: int = 300):

    variance_data = {
        'high': {network_type: [] for network_type in network_types},
        'low': {network_type: [] for network_type in network_types}
    }
    
    #run simulations with high priors
    for network_type in network_types:
        model = ZollmanModel(
            num_agents=num_agents,
            network_type=network_type,
            true_probs=true_probs,
            max_prior_value=high_prior_value,
            max_iterations=max_iterations
        )
        
        for _ in range(max_iterations):
            model.run_iteration()
        
        variance_data['high'][network_type] = model.variance_history
    
    #run simulations with low priors
    for network_type in network_types:
        model = ZollmanModel(
            num_agents=num_agents,
            network_type=network_type,
            true_probs=true_probs,
            max_prior_value=low_prior_value,
            max_iterations=max_iterations
        )
        
        for _ in range(max_iterations):
            model.run_iteration()
        
        variance_data['low'][network_type] = model.variance_history
    
    #plot results
    plt.figure(figsize=(12, 8))
    
    #plot high prior results
    for network_type in network_types:
        plt.plot(
            variance_data['high'][network_type], 
            label=f"High-{network_type}"
        )
    
    #plot low prior results
    for network_type in network_types:
        plt.plot(
            variance_data['low'][network_type], 
            label=f"Low-{network_type}"
        )
    
    plt.xlabel('Generations')
    plt.ylabel('Variance in Actions')
    plt.title('Diversity Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    #example usage
    #reproduce the experiment from Figure 3
    network_types = ['cycle', 'wheel', 'complete']
    true_probs = [0.5, 0.499]  # As used in the paper
    
    results = run_experiment_batch(
        num_trials=100,
        network_types=network_types,
        num_agents=10,
        true_probs=true_probs
    )
    
    print("\nResults:")
    for network_type, success_rate in results.items():
        print(f"{network_type}: {success_rate:.3f}")
    
    #figure 6 (varying max_prior)
    # plot_varying_max_prior(
    #     network_types=network_types,
    #     num_agents=7,
    #     true_probs=true_probs,
    #     max_prior_values=[1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000],
    #     num_trials=20
    # )
    
    #figure 4 (network densities)
    # plot_network_densities(max_agents=6, num_trials=20)
    
    #figure 7 (diversity over time)
    # compare_diversity_over_time(
    #     network_types=network_types,
    #     num_agents=7,
    #     true_probs=true_probs,
    #     high_prior_value=7000,
    #     low_prior_value=4,
    #     max_iterations=300
    # )