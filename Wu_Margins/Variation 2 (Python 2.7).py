import networkx as nx 
import random
import numpy as np
from numpy import random


nodes_complete = [3,6,12,18]
m_values = [0.2,0.5,0.8]

d=3
di=1
cutoff=300

noofpulls=5
objectiveB=.6

def P_E_D(d, m, P_i_E):
	return 1-min(1,d*m)*(1-P_i_E)

def AgentChoice(B_probability,m,average_cumulative_payoff, pull_B, pull_B_no_cutoff, t, G):
	evidence_givenB=[-1 for n in G.nodes()]
	B_posterior_probability=[0 for n in G.nodes()]
	for n in G.nodes():
		B_posterior_probability[n]=B_probability[n]
	for n in G.nodes():
			if B_probability[n]>.5:
				evidence_givenB[n]=(np.random.binomial(noofpulls, objectiveB, size=None))
				average_cumulative_payoff[n]+=evidence_givenB[n]
				if t>0:
					pull_B_no_cutoff[n]+=1
				if t<cutoff:
					pull_B[n]+=1
			else:
				average_cumulative_payoff[n]+=.5

	distance=[[0 for n in G.nodes()] for n in G.nodes()]
	P_i_E=[[0 for n in G.nodes()] for n in G.nodes()]

	lst_2=list(G.nodes())
	lst=list(G.nodes())
	random.shuffle(lst_2)
	random.shuffle(lst)
	for a in lst_2:
		for n in lst:
			if (B_probability[n]>.5):
				if ((a>=(len(lst)/d)) and (n<(len(lst)/d))):
					distance[a][n]=di
					P_i_E[a][n]=((objectiveB**evidence_givenB[n])*((1-objectiveB)**(noofpulls-evidence_givenB[n]))*B_posterior_probability[a])+(((1-objectiveB)**evidence_givenB[n])*(objectiveB**(noofpulls-evidence_givenB[n]))*(1-B_posterior_probability[a]))
					P_i_H_given_E=((objectiveB**evidence_givenB[n])*((1-objectiveB)**(noofpulls-evidence_givenB[n]))*B_posterior_probability[a])/P_i_E[a][n]
					P_i_H_given_not_E=(B_posterior_probability[a]-((objectiveB**evidence_givenB[n])*((1-objectiveB)**(noofpulls-evidence_givenB[n]))*B_posterior_probability[a]))/(1-P_i_E[a][n])
					B_posterior_probability[a]=P_i_H_given_E*P_E_D(distance[a][n], m, P_i_E[a][n])+P_i_H_given_not_E*(1-P_E_D(distance[a][n], m, P_i_E[a][n]))
				else:
					P_i_E[a][n]=((objectiveB**evidence_givenB[n])*((1-objectiveB)**(noofpulls-evidence_givenB[n]))*B_posterior_probability[a])+(((1-objectiveB)**evidence_givenB[n])*(objectiveB**(noofpulls-evidence_givenB[n]))*(1-B_posterior_probability[a]))
					B_posterior_probability[a]=((objectiveB**evidence_givenB[n])*((1-objectiveB)**(noofpulls-evidence_givenB[n]))*B_posterior_probability[a])/P_i_E[a][n]

	for n in G.nodes():
		B_probability[n]=B_posterior_probability[n]

polarization_count=0.0
win_count=0.0
lose_count=0.0
simulation_runs=10000.0
max_iterations=100000

polarization_tally_1_total=0.0
polarization_tally_1_minority=0.0
polarization_tally_2_total=0.0
polarization_tally_2_minority=0.0
total_tally_true_belief_all=0.0
total_tally_true_belief_minority=0.0
total_tally_false_belief_total=0.0
total_tally_false_belief_minority=0.0

success_matrix_complete=[]
polarization_data=[[] for k in nodes_complete]

def run_complete():
	for k in nodes_complete:
		for m in m_values:
			global polarization_count
			#global stuck_count
			global win_count
			global lose_count
			global polarization_tally_1_total
			global polarization_tally_1_minority
			global polarization_tally_2_total
			global polarization_tally_2_minority
			global total_tally_true_belief_all
			global total_tally_true_belief_minority
			global total_tally_false_belief_total
			global total_tally_false_belief_minority
			polarization_tally_1_total=0.0
			polarization_tally_1_minority=0.0
			polarization_tally_2_total=0.0
			polarization_tally_2_minority=0.0
			total_tally_true_belief_all=0.0
			total_tally_true_belief_minority=0.0
			total_tally_false_belief_total=0.0
			total_tally_false_belief_minority=0.0
			average_cumulative_payoff=[0 for n in range(k)]
			win_count=0.0
			lose_count=0.0
			win_turn=[[0 for n in range(k)],[0 for n in range(k)]]
			community_win_turn=0
			polarization_count=0.0
			pull_B=[0 for n in range(k)]
			pull_B_no_cutoff=[0 for n in range(k)]
			total_rounds=0
			s=0
			while s<simulation_runs:
				G=nx.complete_graph(k)
				t=0
				B_probability=[]
				B_probability=[random.uniform(0,1) for n in G.nodes()]
				B_initial=[0 for n in G.nodes()]
				for n in G.nodes():
					B_initial[n]=B_probability[n]
				polarized=[False for n in G.nodes()]
				win=[False for n in G.nodes()]
				lose=[False for n in G.nodes()]

				while t<=max_iterations:
					AgentChoice(B_probability,m,average_cumulative_payoff, pull_B, pull_B_no_cutoff, t, G)
					t+=1
					for n in G.nodes():
						win[n]=(B_probability[n]>.99)
						lose[n]=(B_probability[n]<=.5)
					for n in G.nodes():
						if n>=(k/d):
							polarized[n]=(B_probability[n]<=.5)
						if n<(k/d):
							polarized[n]=(B_probability[n]>.99)

					for n in G.nodes():
						if win[n]==True:
							if win_turn[0][n]==0:
								win_turn[0][n]=t
						else:
							win_turn[0][n]=0

					if all(win):
						win_count+=1
						for n in G.nodes():
							if B_probability[n]>.99:
								total_tally_true_belief_all+=1
								if n<(k/d):
									total_tally_true_belief_minority+=1
							else:
								total_tally_false_belief_total+=1
								if n<(k/d):
									total_tally_false_belief_minority+=1
						for n in G.nodes():
							win_turn[1][n]+=win_turn[0][n]
							win_turn[0][n]=0
						community_win_turn+=t
						total_rounds+=(t-1)
						if t<cutoff:
							for n in range(k):
								pull_B[n]+=(cutoff-t)
						break
					elif all(lose):
						lose_count+=1
						for n in G.nodes():
							if B_probability[n]>.99:
								total_tally_true_belief_all+=1
								if n<(k/d):
									total_tally_true_belief_minority+=1
							else:
								total_tally_false_belief_total+=1
								if n<(k/d):
									total_tally_false_belief_minority+=1
						total_rounds+=(t-1)
						break
				s+=1
			success_matrix_complete.append([k,[m,(polarization_count/simulation_runs),(win_count/simulation_runs),(lose_count/simulation_runs),polarization_count, win_count, win_turn[1], community_win_turn, average_cumulative_payoff, pull_B, pull_B_no_cutoff, total_rounds]])

run_complete()

print success_matrix_complete