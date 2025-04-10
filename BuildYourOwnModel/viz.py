import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

def plot_fitness_over_time(time_points, fitness_values): #community fitness over time as a line graph
    plt.figure(figsize=(8, 5))
    plt.plot(time_points, fitness_values, marker='o', linestyle='-', color='b')
    plt.xlabel("Time")
    plt.ylabel("Community Fitness")
    plt.title("Community Fitness Over Time")
    plt.grid(True)
    plt.show()

def plot_community_fitness(community_labels, fitness_values): #bar graph showing fitness values for different communities
    plt.figure(figsize=(8, 5))
    plt.bar(community_labels, fitness_values, color='green', alpha=0.7)
    plt.xlabel("Community")
    plt.ylabel("Fitness")
    plt.title("Community Fitness Comparison")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def plot_fitness_graph(graph, fitness_values): #node colors correspond to fitness values
    plt.figure(figsize=(6, 6))
    pos = nx.spring_layout(graph)  # Layout for visualization
    node_colors = [fitness_values[node] for node in graph.nodes]
    nx.draw(graph, pos, with_labels=True, node_color=node_colors, cmap=plt.cm.viridis, node_size=300, edge_color='gray')
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
    plt.colorbar(sm, label='Fitness')
    plt.title("Graph Visualization with Fitness Coloring")
    plt.show()

def plot_fitness_distribution(fitness_values, bins=10): #histogram showing the distribution of fitness values
    plt.figure(figsize=(8, 5))
    plt.hist(fitness_values, bins=bins, color='blue', alpha=0.7, edgecolor='black')
    plt.xlabel("Fitness")
    plt.ylabel("Frequency")
    plt.title("Distribution of Fitness Values")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def plot_fitness_heatmap(fitness_matrix, labels): #heatmap showing community similarities based on fitness
    plt.figure(figsize=(8, 6))
    sns.heatmap(fitness_matrix, annot=True, xticklabels=labels, yticklabels=labels, cmap='coolwarm', linewidths=0.5)
    plt.xlabel("Community")
    plt.ylabel("Community")
    plt.title("Fitness Similarity Heatmap")
    plt.show()

def plot_fitness_scatter(x_values, y_values, xlabel, ylabel): #scatter plot to visualize relationships between fitness and another variable
    plt.figure(figsize=(8, 5))
    plt.scatter(x_values, y_values, color='red', alpha=0.7, edgecolors='black')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs {xlabel}")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
