#!/usr/bin/env python

"""
Get and plot data saved for QIF neurons

Developped by Raphael BERGOIN

Run : python3 plotQIF.py
"""

#export ARCHFLAGS="-arch x86_64"
#python3 -mpip install numpy --user
import matplotlib
from math import *
import string
import numpy as np
from io import StringIO
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation
import codecs
import random
import warnings
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import time
import copy
import cmath
import operator
from scipy import stats
from sklearn.metrics import mutual_info_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import community as community_louvain
import itertools
import cv2
#import pycochleagram.cochleagram as cgram
#from pycochleagram import utils
from scipy.io.wavfile import write
from playsound import playsound
#import hdbscan


#warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)



def orderParameter(theta, m, N) :
    """
        Calculate the order parameter
        
        Parameters : 
        theta -- phase vector
        m -- order
        N -- number of neurons
    """
    
    theta = np.array(theta)  # Convert theta to a numpy array
    R = np.sum(np.exp(1.0j * m * theta))
    return abs((1.0 / N) * R)


def getMeanFrequenciesAllTimes(N, startT, T, P, spikesMatrix, timeConstant) :
    """
        Create a matrix of mean frequencies of each neurons at each time step
        
        Parameters : 
        N -- number of neurons
        startT -- iteration of start calculation
        T -- number of iterations of the simulation
        P -- period for the mean (in second)
        spikesMatrix -- matrix for each spikes of each neurons
        timeConstant -- constant factor time per iteration (precision)
    """
    
    meanFrequencies = np.zeros((N, T))
    
    for i in range(N) :     #For each neurons
        for t in range(startT+int(P/timeConstant), T) :         #For each time
            meanFrequencies[i][t] = np.count_nonzero((spikesMatrix[i] >= t*timeConstant-P) & (spikesMatrix[i] <= t*timeConstant))/P
    return meanFrequencies
    


""" Arguments """
    
if len(sys.argv) > 1 :
    if str(sys.argv[1]) == "1" :
        save = True
    else :
        save = False
else :
    save = False



""" Parameters simulation """
    
adimensional = False    #If adimensional time
animated = False        #If the figure are animated    True
dt = 0.1                #Time step (depends on time step used to register data)
vp = 10.0			    #peak value 
vr = -10.0				#reset value 
tfinal = 50.0           #duration simulation in s   //50.0 1000.0 18000.0 4000.0 86400 172800.0 18000.0
nbNeurons = 100         # 100 200 500 1000 2000 20000
iterations = int(tfinal/dt)  
tpoints = np.arange(0.0, (tfinal+0.000001), dt)



"""Get data saved"""   
    
matrices = np.loadtxt('weights_matrices.txt', dtype=float )
#matrices = matrices.reshape(2,nbNeurons,nbNeurons)
matrices = matrices.reshape(4,nbNeurons,nbNeurons)
#matrices = matrices.reshape(6,nbNeurons,nbNeurons)
#matrices = matrices.reshape(8,nbNeurons,nbNeurons)


f = open("changeRates.txt", "r")
changeRates = []
for x in f:
    lst = x.split()
    changeRates.append([float(i) for i in lst])

f.close()
changeRates = np.array(changeRates)
changeRates = changeRates[0:, :]
changeRates = changeRates * 10.0  #normalize by delta t = 0.1s
#changeRates = changeRates * 1.0  #normalize by delta t = 1.0s

f = open("inhibitory.txt", "r")
inhibitory = []
for x in f:
    lst = x.split()
    inhibitory = [int(i) for i in lst]

f.close()
inhibitory = np.array(inhibitory)

f = open("spikes.txt", "r")
spikes = [[] for i in range(len(inhibitory))]
for x in f:
    lst = x.split()
    spikes[int(lst[0])].append(float(lst[1]))

f.close()
spikes = [x + [tfinal] for x in spikes] #artificially create spikes last iteration to calculate phase order parameters
spikes = np.array(spikes,dtype=object)



""" Sorting neurons """

#Order according to states
order = list(range(0, len(inhibitory)))
nbExcit = (inhibitory == 1).sum()
nbInhib = (inhibitory == -1).sum()
nbInhibCluster = int(nbInhib/2.0)
excit = np.arange(nbExcit)
inhib = [x for x in order if x not in excit]

# Sort Hebbian and anti-Hebbian thanks to parity
#antiHebb1 = [x for x in inhib if nbExcit <= x < nbExcit+nbInhibCluster and x % 2 == 0]          #even first half inhibitory
#hebb1 = [x for x in inhib if nbExcit <= x < nbExcit+nbInhibCluster and x % 2 != 0]              #odd first half inhibitory
#hebb2 = [x for x in inhib if nbExcit+nbInhibCluster <= x < nbNeurons and x % 2 != 0]            #odd second half inhibitory
#antiHebb2 = [x for x in inhib if nbExcit+nbInhibCluster <= x < nbNeurons and x % 2 == 0]        #even second half inhibitory

#inhib = antiHebb1 + hebb1 + hebb2 + antiHebb2


if len(inhib) != 0:
    order = np.concatenate((excit, inhib))
else:
    order = excit




""" Process data """ 

startIterations = int(0.0/dt)
#startIterations = int((tfinal-2000.0)/dt) #time to start calculating order parameters and firing rates

#Calculate order parameters through the time
orderParameter1 = np.zeros(iterations+1)
orderParameter1_1 = np.zeros(iterations+1)
orderParameter1_2 = np.zeros(iterations+1)
phases_network = np.empty(iterations+1, dtype=object)


for t in range(startIterations, iterations) :
    #Get indices of closest spikes after time t
    indices = [np.searchsorted(spikes[i], t*dt, side='right') for i in range(0, nbNeurons)]
            
    #calculate times spikes, calculate phases and order parameter
    tn = [spikes[i][indices[i]-1] for i in range(0, nbNeurons)]
    tn1 = [spikes[i][indices[i]] for i in range(0, nbNeurons)]
    phases = [2.0 * np.pi * ((t * dt) - tn[i]) / (tn1[i] - tn[i] + 0.0000001) for i in range(0, nbNeurons)]
    
    phases_network[t] = phases
        
    orderParameter1[t] = orderParameter(phases, 1, nbNeurons)
    orderParameter1_1[t] = orderParameter(phases[0:int(nbExcit/2)], 1, int(nbExcit/2))
    orderParameter1_2[t] = orderParameter(phases[int(nbExcit/2):nbExcit], 1, int(nbExcit/2))

#Artificially create order parameters last iteration to fit dimensions

orderParameter1_1[-1] = orderParameter(phases[0:int(nbExcit/2)], 1, int(nbExcit/2))
orderParameter1_2[-1] = orderParameter(phases[int(nbExcit/2):nbExcit], 1, int(nbExcit/2))
orderParameter1[-1] = orderParameter(phases, 1, nbNeurons)
phases_network[-1] = phases


spikes = [np.delete(x, -1) for x in spikes]     #Remove artificiacial last spike
spikes = np.array(spikes,dtype=object)
phases_network = np.array(phases_network)

#Calculate mean frequencies and order of neurons
allMeanFrequencies = getMeanFrequenciesAllTimes(len(inhibitory), startIterations, iterations+1, 0.05, spikes, dt)   #mean for periods of 0.05s



"""Plot data""" 

newcmap = np.genfromtxt('./Matplotlib_colourmap_div.csv', delimiter=',')
cmap_div = ListedColormap(newcmap)
newblue = (0.21961,0.33333,0.64706)
newred = (0.7098,0.070588,0.070588)






"""
#m(t) m(t+1) phase
plt.scatter(np.mean(phases_network, axis=1), np.roll(np.mean(phases_network, axis=1), -1), s=2, color='grey')
plt.scatter(np.mean(phases_network[:, 0:int(nbExcit/2)], axis=1), np.roll(np.mean(phases_network[:, 0:int(nbExcit/2)], axis=1), -1), s=2, color='green')
plt.scatter(np.mean(phases_network[:, int(nbExcit/2):nbExcit], axis=1), np.roll(np.mean(phases_network[:, int(nbExcit/2):nbExcit], axis=1), -1), s=2, color='orange')
plt.gca().set_ylabel('phase(t+1)', fontsize=20)
plt.gca().set_xlabel('phase(t)', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
if save : 
   plt.savefig('results/state.png', dpi=300, bbox_inches='tight')
   plt.close()
else :
   plt.gca().set_title('State represented in mean phase space', fontsize=25)
   plt.show()
   
   
#m(t) m(t+1) R
plt.scatter(orderParameter1, np.roll(orderParameter1, -1), s=2, color='grey')
plt.scatter(orderParameter1_1, np.roll(orderParameter1_1, -1), s=2, color='green')
plt.scatter(orderParameter1_2, np.roll(orderParameter1_2, -1), s=2, color='orange')
plt.gca().set_ylabel('R(t+1)', fontsize=20)
plt.gca().set_xlabel('R(t)', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
if save : 
   plt.savefig('results/state.png', dpi=300, bbox_inches='tight')
   plt.close()
else :
   plt.gca().set_title('State represented in R space', fontsize=25)
   plt.show()


#m(t) m(t+1) F
plt.scatter(np.mean(allMeanFrequencies, axis=0), np.roll(np.mean(allMeanFrequencies, axis=0), -1), s=2, color='grey')
plt.scatter(np.mean(allMeanFrequencies[0:int(nbExcit/2), :], axis=0), np.roll(np.mean(allMeanFrequencies[0:int(nbExcit/2), :], axis=0), -1), s=2, color='green')
plt.scatter(np.mean(allMeanFrequencies[int(nbExcit/2):nbExcit, :], axis=0), np.roll(np.mean(allMeanFrequencies[int(nbExcit/2):nbExcit, :], axis=0), -1), s=2, color='orange')
plt.gca().set_ylabel('F(t+1)', fontsize=20)
plt.gca().set_xlabel('F(t)', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
if save : 
   plt.savefig('results/state.png', dpi=300, bbox_inches='tight')
   plt.close()
else :
   plt.gca().set_title('State represented in mean frequency space', fontsize=25)
   plt.show()
"""   

"""
######## Plot states ########

R = orderParameter1[:-1]
R1 = orderParameter1_1[:-1]
R2 = orderParameter1_2[:-1]

F = np.mean(allMeanFrequencies, axis=0)[:-1]
F1 = np.mean(allMeanFrequencies[0:int(nbExcit/2), :-1], axis=0)
F2 = np.mean(allMeanFrequencies[int(nbExcit/2):nbExcit, :-1], axis=0)

W = changeRates[:,0]
W1 = changeRates[:,1]
W2 = changeRates[:,2]

points = np.column_stack((R, F, W))
points_1 = np.column_stack((R1, F1, W1))
points_2 = np.column_stack((R2, F2, W2))

# Normalize
scaler = MinMaxScaler()
points = scaler.fit_transform(points) 
points_1 = scaler.fit_transform(points_1) 
points_2 = scaler.fit_transform(points_2) 

# Clusterize
#clusterer = hdbscan.HDBSCAN(min_cluster_size=100) 
#labels = clusterer.fit_predict(points)

#gmm = GaussianMixture(n_components=4, covariance_type='full')
#labels = gmm.fit_predict(points)

kmeans = KMeans(n_clusters=8, random_state=42)
labels = kmeans.fit_predict(np.vstack((points_1, points_2)))
centers = kmeans.cluster_centers_  


fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
points_replaced = centers[labels]

#Plot clusters
ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='red', marker='x', s=100)  

#Plot clusters directed edges
seen_links = set()
for i in range(len(points_replaced) - 1):
    start = tuple(points_replaced[i])    
    end = tuple(points_replaced[i + 1])  

    link = (start, end)  
    
    if link in seen_links:
        continue  
    seen_links.add(link)

    ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], c='red', alpha=0.5)

    mid_point = (np.array(start) + np.array(end)) / 2  
    direction = (np.array(end) - np.array(start)) * 0.3

    ax.quiver(mid_point[0], mid_point[1], mid_point[2],  
              direction[0], direction[1], direction[2],  
              color='blue', arrow_length_ratio=0.2, alpha=0.7)

#Plot points
ax.scatter(np.vstack((points_1, points_2))[:, 0], np.vstack((points_1, points_2))[:, 1], np.vstack((points_1, points_2))[:, 2], c=labels, cmap='plasma', s=20)

#ax.scatter(R, F, W, color='grey', lw=2)
#ax.scatter(R1, F1, W1, color='green', lw=2)
#ax.scatter(R2, F2, W2, color='orange', lw=2)

#ax.plot(R, F, W, color='grey', lw=2)
#ax.plot(R1, F1, W1, color='green', lw=2)
#ax.plot(R2, F2, W2, color='orange', lw=2)

ax.set_xlabel('R')
ax.set_ylabel('F')
ax.set_zlabel('Delta w')
plt.show()
"""



#Connectivity

#Weights matrix 
#kij : j presynaptic to i postsynaptic neurons
plt.figure(figsize=(1.8,1.5), dpi=300)
plt.imshow(matrices[0][order, :][:, order], cmap=cmap_div, vmin=-1, vmax=1, aspect='auto', origin='lower')
plt.clim(-1,1)
#plt.gca().set_xlabel('Presynaptic neurons j', fontsize=20)
#plt.gca().set_ylabel('Postsynaptic neurons i', fontsize=20)
plt.gca().xaxis.tick_bottom()
#plt.gca().invert_yaxis()
#plt.xticks((0,int(nbExcit/2),nbExcit,int(nbNeurons-nbInhib/2),nbNeurons),(0,int(nbExcit/2),nbExcit,int(nbNeurons-nbInhib/2),nbNeurons), fontsize=7)
#plt.yticks((0,int(nbExcit/2),nbExcit,int(nbNeurons-nbInhib/2),nbNeurons),(0,int(nbExcit/2),nbExcit,int(nbNeurons-nbInhib/2),nbNeurons), fontsize=7)
plt.xticks((0,int(nbExcit/2),nbExcit),(0,int(nbExcit/2),nbExcit), fontsize=7)
plt.yticks((0,int(nbExcit/2),nbExcit),(0,int(nbExcit/2),nbExcit), fontsize=7)
cbar = plt.colorbar()
#cbar.ax.tick_params(labelsize=20)
cbar.set_ticks((-1,0,1))
cbar.set_ticklabels((-1,0,1))
cbar.ax.tick_params(labelsize=7)
plt.tight_layout()
if save : 
    plt.savefig('results/weights_matrix_sorted_T0.pdf', dpi=300, bbox_inches='tight')
    plt.close()
else :
    plt.gca().set_title('Weight matrix (sorted) T0', fontsize=25)
    plt.show() 

  
#Weights matrix 
#kij : j presynaptic to i postsynaptic neurons
plt.figure(figsize=(1.8,1.5), dpi=300)
plt.imshow(matrices[1][order, :][:, order], cmap=cmap_div, vmin=-1, vmax=1, aspect='auto', origin='lower')
plt.clim(-1,1)
#plt.gca().set_xlabel('Presynaptic neurons j', fontsize=20)
#plt.gca().set_ylabel('Postsynaptic neurons i', fontsize=20)
plt.gca().xaxis.tick_bottom()
#plt.gca().invert_yaxis()
#plt.xticks((0,int(nbExcit/2),nbExcit,int(nbNeurons-nbInhib/2),nbNeurons),(0,int(nbExcit/2),nbExcit,int(nbNeurons-nbInhib/2),nbNeurons), fontsize=7)
#plt.yticks((0,int(nbExcit/2),nbExcit,int(nbNeurons-nbInhib/2),nbNeurons),(0,int(nbExcit/2),nbExcit,int(nbNeurons-nbInhib/2),nbNeurons), fontsize=7)
plt.xticks((0,int(nbExcit/2),nbExcit),(0,int(nbExcit/2),nbExcit), fontsize=7)
plt.yticks((0,int(nbExcit/2),nbExcit),(0,int(nbExcit/2),nbExcit), fontsize=7)
cbar = plt.colorbar()
#cbar.ax.tick_params(labelsize=20)
cbar.set_ticks((-1,0,1))
cbar.set_ticklabels((-1,0,1))
cbar.ax.tick_params(labelsize=7)
plt.tight_layout()
if save : 
    plt.savefig('results/weights_matrix_sorted_T1.pdf', dpi=300, bbox_inches='tight')
    plt.close()
else :
    plt.gca().set_title('Weight matrix (sorted) T1', fontsize=25)
    plt.show()
 
   
#Weights matrix  
#kij : j presynaptic to i postsynaptic neurons
plt.figure(figsize=(1.8,1.5), dpi=300)
plt.imshow(matrices[2][order, :][:, order], cmap=cmap_div, vmin=-1, vmax=1, aspect='auto', origin='lower')
plt.clim(-1,1)
#plt.gca().set_xlabel('Presynaptic neurons j', fontsize=20)
#plt.gca().set_ylabel('Postsynaptic neurons i', fontsize=20)
plt.gca().xaxis.tick_bottom()
#plt.gca().invert_yaxis()
#plt.xticks((0,int(nbExcit/2),nbExcit,int(nbNeurons-nbInhib/2),nbNeurons),(0,int(nbExcit/2),nbExcit,int(nbNeurons-nbInhib/2),nbNeurons), fontsize=7)
#plt.yticks((0,int(nbExcit/2),nbExcit,int(nbNeurons-nbInhib/2),nbNeurons),(0,int(nbExcit/2),nbExcit,int(nbNeurons-nbInhib/2),nbNeurons), fontsize=7)
plt.xticks((0,int(nbExcit/2),nbExcit),(0,int(nbExcit/2),nbExcit), fontsize=7)
plt.yticks((0,int(nbExcit/2),nbExcit),(0,int(nbExcit/2),nbExcit), fontsize=7)
cbar = plt.colorbar()
#cbar.ax.tick_params(labelsize=20)
cbar.set_ticks((-1,0,1))
cbar.set_ticklabels((-1,0,1))
cbar.ax.tick_params(labelsize=7)
plt.tight_layout()
if save : 
    plt.savefig('results/weights_matrix_sorted_T2.pdf', dpi=300, bbox_inches='tight')
    plt.close()
else :
    plt.gca().set_title('Weight matrix (sorted) T2', fontsize=25)
    plt.show()  
    

#Weights matrix  
#kij : j presynaptic to i postsynaptic neurons
plt.figure(figsize=(1.8,1.5), dpi=300)
plt.imshow(matrices[3][order, :][:, order], cmap=cmap_div, vmin=-1, vmax=1, aspect='auto', origin='lower')
plt.clim(-1,1)
#plt.gca().set_xlabel('Presynaptic neurons j', fontsize=20)
#plt.gca().set_ylabel('Postsynaptic neurons i', fontsize=20)
plt.gca().xaxis.tick_bottom()
#plt.gca().invert_yaxis()
#plt.xticks((0,int(nbExcit/2),nbExcit,int(nbNeurons-nbInhib/2),nbNeurons),(0,int(nbExcit/2),nbExcit,int(nbNeurons-nbInhib/2),nbNeurons), fontsize=7)
#plt.yticks((0,int(nbExcit/2),nbExcit,int(nbNeurons-nbInhib/2),nbNeurons),(0,int(nbExcit/2),nbExcit,int(nbNeurons-nbInhib/2),nbNeurons), fontsize=7)
plt.xticks((0,int(nbExcit/2),nbExcit),(0,int(nbExcit/2),nbExcit), fontsize=7)
plt.yticks((0,int(nbExcit/2),nbExcit),(0,int(nbExcit/2),nbExcit), fontsize=7)
cbar = plt.colorbar()
#cbar.ax.tick_params(labelsize=20)
cbar.set_ticks((-1,0,1))
cbar.set_ticklabels((-1,0,1))
cbar.ax.tick_params(labelsize=7)
plt.tight_layout()
if save : 
    plt.savefig('results/weights_matrix_sorted_T3.pdf', dpi=300, bbox_inches='tight')
    plt.close()
else :
    plt.gca().set_title('Weight matrix (sorted) T3', fontsize=25)
    plt.show()  
    
"""     
#Weights matrix  
#kij : j presynaptic to i postsynaptic neurons
plt.figure(figsize=(1.8,1.5), dpi=300)
plt.imshow(matrices[4][order, :][:, order], cmap=cmap_div, vmin=-1, vmax=1, aspect='auto', origin='lower')
plt.clim(-1,1)
#plt.gca().set_xlabel('Presynaptic neurons j', fontsize=20)
#plt.gca().set_ylabel('Postsynaptic neurons i', fontsize=20)
plt.gca().xaxis.tick_bottom()
#plt.gca().invert_yaxis()
#plt.xticks((0,int(nbExcit/2),nbExcit,int(nbNeurons-nbInhib/2),nbNeurons),(0,int(nbExcit/2),nbExcit,int(nbNeurons-nbInhib/2),nbNeurons), fontsize=7)
#plt.yticks((0,int(nbExcit/2),nbExcit,int(nbNeurons-nbInhib/2),nbNeurons),(0,int(nbExcit/2),nbExcit,int(nbNeurons-nbInhib/2),nbNeurons), fontsize=7)
plt.xticks((0,int(nbExcit/2),nbExcit),(0,int(nbExcit/2),nbExcit), fontsize=7)
plt.yticks((0,int(nbExcit/2),nbExcit),(0,int(nbExcit/2),nbExcit), fontsize=7)
cbar = plt.colorbar()
#cbar.ax.tick_params(labelsize=20)
cbar.set_ticks((-1,0,1))
cbar.set_ticklabels((-1,0,1))
cbar.ax.tick_params(labelsize=7)
plt.tight_layout()
if save : 
    plt.savefig('results/weights_matrix_sorted_T4.pdf', dpi=300, bbox_inches='tight')
    plt.close()
else :
    plt.gca().set_title('Weight matrix (sorted) T4', fontsize=25)
    plt.show()      
  
  

#Weights matrix 
#kij : j presynaptic to i postsynaptic neurons
plt.figure(figsize=(1.8,1.5), dpi=300)
plt.imshow(matrices[5][order, :][:, order], cmap=cmap_div, vmin=-1, vmax=1, aspect='auto', origin='lower')
plt.clim(-1,1)
#plt.gca().set_xlabel('Presynaptic neurons j', fontsize=20)
#plt.gca().set_ylabel('Postsynaptic neurons i', fontsize=20)
plt.gca().xaxis.tick_bottom()
#plt.gca().invert_yaxis()
#plt.xticks((0,int(nbExcit/2),nbExcit,int(nbNeurons-nbInhib/2),nbNeurons),(0,int(nbExcit/2),nbExcit,int(nbNeurons-nbInhib/2),nbNeurons), fontsize=7)
#plt.yticks((0,int(nbExcit/2),nbExcit,int(nbNeurons-nbInhib/2),nbNeurons),(0,int(nbExcit/2),nbExcit,int(nbNeurons-nbInhib/2),nbNeurons), fontsize=7)
plt.xticks((0,int(nbExcit/2),nbExcit),(0,int(nbExcit/2),nbExcit), fontsize=7)
plt.yticks((0,int(nbExcit/2),nbExcit),(0,int(nbExcit/2),nbExcit), fontsize=7)
cbar = plt.colorbar()
#cbar.ax.tick_params(labelsize=20)
cbar.set_ticks((-1,0,1))
cbar.set_ticklabels((-1,0,1))
cbar.ax.tick_params(labelsize=7)
plt.tight_layout()
if save : 
    plt.savefig('results/weights_matrix_sorted_T5.pdf', dpi=300, bbox_inches='tight')
    plt.close()
else :
    plt.gca().set_title('Weight matrix (sorted) T5', fontsize=25)
    plt.show() 
    
             
         
#Weights matrix 
#kij : j presynaptic to i postsynaptic neurons
plt.figure(figsize=(1.8,1.5), dpi=300)
plt.imshow(matrices[6][order, :][:, order], cmap=cmap_div, vmin=-1, vmax=1, aspect='auto', origin='lower')
plt.clim(-1,1)
#plt.gca().set_xlabel('Presynaptic neurons j', fontsize=20)
#plt.gca().set_ylabel('Postsynaptic neurons i', fontsize=20)
plt.gca().xaxis.tick_bottom()
#plt.gca().invert_yaxis()
#plt.xticks((0,int(nbExcit/2),nbExcit,int(nbNeurons-nbInhib/2),nbNeurons),(0,int(nbExcit/2),nbExcit,int(nbNeurons-nbInhib/2),nbNeurons), fontsize=7)
#plt.yticks((0,int(nbExcit/2),nbExcit,int(nbNeurons-nbInhib/2),nbNeurons),(0,int(nbExcit/2),nbExcit,int(nbNeurons-nbInhib/2),nbNeurons), fontsize=7)
plt.xticks((0,int(nbExcit/2),nbExcit),(0,int(nbExcit/2),nbExcit), fontsize=7)
plt.yticks((0,int(nbExcit/2),nbExcit),(0,int(nbExcit/2),nbExcit), fontsize=7)
cbar = plt.colorbar()
#cbar.ax.tick_params(labelsize=20)
cbar.set_ticks((-1,0,1))
cbar.set_ticklabels((-1,0,1))
cbar.ax.tick_params(labelsize=7)
plt.tight_layout()
if save : 
    plt.savefig('results/weights_matrix_sorted_T6.pdf', dpi=300, bbox_inches='tight')
    plt.close()
else :
    plt.gca().set_title('Weight matrix (sorted) T6', fontsize=25)
    plt.show() 




#Weights matrix 
#kij : j presynaptic to i postsynaptic neurons
plt.figure(figsize=(1.8,1.5), dpi=300)
plt.imshow(matrices[7][order, :][:, order], cmap=cmap_div, vmin=-1, vmax=1, aspect='auto', origin='lower')
plt.clim(-1,1)
#plt.gca().set_xlabel('Presynaptic neurons j', fontsize=20)
#plt.gca().set_ylabel('Postsynaptic neurons i', fontsize=20)
plt.gca().xaxis.tick_bottom()
#plt.gca().invert_yaxis()
#plt.xticks((0,int(nbExcit/2),nbExcit,int(nbNeurons-nbInhib/2),nbNeurons),(0,int(nbExcit/2),nbExcit,int(nbNeurons-nbInhib/2),nbNeurons), fontsize=7)
#plt.yticks((0,int(nbExcit/2),nbExcit,int(nbNeurons-nbInhib/2),nbNeurons),(0,int(nbExcit/2),nbExcit,int(nbNeurons-nbInhib/2),nbNeurons), fontsize=7)
plt.xticks((0,int(nbExcit/2),nbExcit),(0,int(nbExcit/2),nbExcit), fontsize=7)
plt.yticks((0,int(nbExcit/2),nbExcit),(0,int(nbExcit/2),nbExcit), fontsize=7)
cbar = plt.colorbar()
#cbar.ax.tick_params(labelsize=20)
cbar.set_ticks((-1,0,1))
cbar.set_ticklabels((-1,0,1))
cbar.ax.tick_params(labelsize=7)
plt.tight_layout()
if save : 
    plt.savefig('results/weights_matrix_sorted_T7.pdf', dpi=300, bbox_inches='tight')
    plt.close()
else :
    plt.gca().set_title('Weight matrix (sorted) T7', fontsize=25)
    plt.show() 
"""
#Weights matrix 
#kij : j presynaptic to i postsynaptic neurons
plt.figure(figsize=(1.8,1.5), dpi=300)
plt.imshow(matrices[-1][order, :][:, order], cmap=cmap_div, vmin=-1, vmax=1, aspect='auto', origin='lower')
plt.clim(-1,1)
#plt.gca().set_xlabel('Presynaptic neurons j', fontsize=20)
#plt.gca().set_ylabel('Postsynaptic neurons i', fontsize=20)
plt.gca().xaxis.tick_bottom()
#plt.gca().invert_yaxis()
#plt.xticks((0,int(nbExcit/2),nbExcit,int(nbNeurons-nbInhib/2),nbNeurons),(0,int(nbExcit/2),nbExcit,int(nbNeurons-nbInhib/2),nbNeurons), fontsize=7)
#plt.yticks((0,int(nbExcit/2),nbExcit,int(nbNeurons-nbInhib/2),nbNeurons),(0,int(nbExcit/2),nbExcit,int(nbNeurons-nbInhib/2),nbNeurons), fontsize=7)
plt.xticks((0,int(nbExcit/2),nbExcit),(0,int(nbExcit/2),nbExcit), fontsize=7)
plt.yticks((0,int(nbExcit/2),nbExcit),(0,int(nbExcit/2),nbExcit), fontsize=7)
cbar = plt.colorbar()
#cbar.ax.tick_params(labelsize=20)
cbar.set_ticks((-1,0,1))
cbar.set_ticklabels((-1,0,1))
cbar.ax.tick_params(labelsize=7)
plt.tight_layout()
if save : 
    plt.savefig('results/weights_matrix_sorted_TF.pdf', dpi=300, bbox_inches='tight')
    plt.close()
else :
    plt.gca().set_title('Weight matrix (sorted) TF', fontsize=25)
    plt.show() 

#times = [0.0, 40.0, 400.0, 4000.0]
times = [0, 1, 2, 4, 6, 12, 18, 24]

#Weights matrices (sorted)
if animated:
    fig, ax = plt.subplots()
    img = ax.matshow(matrices[0], cmap=cmap_div, vmin=-1, vmax=1)
    plt.gca().set_xlabel('Presynaptic neurons j', fontsize=20)
    plt.gca().set_ylabel('Postsynaptic neurons i', fontsize=20)
    ax.set_title('Weight matrix (sorted) T0', fontsize=25)
    plt.gca().xaxis.tick_bottom()
    plt.gca().invert_yaxis()
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    cbar = fig.colorbar(img)
    cbar.ax.tick_params(labelsize=20)
    def animateWeightsMatrices(frame):
        #ax.set_title('Weight matrix (sorted) T%d' % (frame%len(matrices)), fontsize=25)
        #0.0 40.0 400.0 4000.0 
        ax.set_title('Weight matrix t=%dh' % times[(frame%len(matrices))] , fontsize=25)
        ax.matshow(matrices[frame%len(matrices)], cmap=cmap_div, vmin=-1, vmax=1)
        plt.gca().xaxis.tick_bottom()
        plt.gca().invert_yaxis()
    ani = animation.FuncAnimation(fig, animateWeightsMatrices, interval=2000, blit=False, frames=20, cache_frame_data=False, repeat=True)
    if save : 
        plt.gcf().set_size_inches(16, 9)
        #writer = animation.FFMpegWriter(bitrate=-1, codec="libx264", extra_args=['-pix_fmt', 'yuv420p'])
        #ani.save("results/weights_matrix_sorted.mp4")
        ani.save("results/weights_matrix_sorted.gif", writer="pillow", fps=0.5, dpi=80)
        plt.close()
    else :
        plt.show()


#Distribution of the weights
plt.figure(figsize=(4.0,3.2), dpi=300)
plt.hist(matrices[0].flatten(), bins=100, histtype=u'step', density=True, color='lime', label='T=0s', log=True) #Beginning of the simulation
plt.hist(matrices[2].flatten(), bins=100, histtype=u'step', density=True, color='cyan', label='T=400s', log=True) #in the middle
plt.hist(matrices[-1].flatten(), bins=100, histtype=u'step', density=True, color='magenta', label='T=4000s', log=True) #End of the simulation

#plt.hist(matrices[0].flatten(), bins=100, histtype=u'step', density=True, color='lime', label='T=0s', log=True) #Beginning of the simulation
#plt.hist(matrices[2].flatten(), bins=100, histtype=u'step', density=True, color='cyan', label='T=15min', log=True) #in the middle
#plt.hist(matrices[-1].flatten(), bins=100, histtype=u'step', density=True, color='magenta', label='T=24h', log=True) #End of the simulation
#plt.gca().set_xlabel('Weights', fontsize=20)
#plt.gca().set_ylabel('Population', fontsize=20)
plt.xticks([-1, -0.5, 0, 0.5, 1], fontsize=13)
plt.yticks([], fontsize=13)
plt.tight_layout()

plt.gca().legend(bbox_to_anchor=(0.5, 1.18), loc='upper center', fontsize=13, ncol=3, columnspacing=0.5)
if save : 
    plt.savefig('results/distribution_weights.pdf', dpi=300, bbox_inches='tight')
    plt.close()
else :
    plt.gca().set_title('Distribution of the weights (log scale)', fontsize=25)
    plt.show() 
    
"""
#Evolution of the change rate of weights
fig, ax = plt.subplots(figsize=(9.0,3.0))

#plt.figure(figsize=(6.0,1.0))

#line = ax.plot(tpoints[0:len(tpoints)-1], changeRates[:,0], label='Absolute change network', color='grey')
#line2 = ax.plot(tpoints[0:len(tpoints)-1], changeRates[:,1], label='Cluster 1', color='green', linewidth=4) 
#line3 = ax.plot(tpoints[0:len(tpoints)-1], changeRates[:,2], label='Cluster 2', color='orange', linewidth=4)

line = ax.plot(tpoints[0:len(tpoints)-2], changeRates[:,0], label='intra E-E', linewidth=4, linestyle='-', color='darkred') 
line = ax.plot(tpoints[0:len(tpoints)-2], changeRates[:,1], label='inter E-E', linewidth=4, linestyle=':', color='darkred') 
line = ax.plot(tpoints[0:len(tpoints)-2], changeRates[:,2], label='intra E-I', linewidth=4, linestyle='-', color='red') 
line = ax.plot(tpoints[0:len(tpoints)-2], changeRates[:,3], label='inter E-I', linewidth=4, linestyle=':', color='red') 
line = ax.plot(tpoints[0:len(tpoints)-2], changeRates[:,4], label='intra A-E', linewidth=4, linestyle='-', color='blue') 
line = ax.plot(tpoints[0:len(tpoints)-2], changeRates[:,5], label='inter A-E', linewidth=4, linestyle=':', color='blue') 
line = ax.plot(tpoints[0:len(tpoints)-2], changeRates[:,6], label='intra A-I', linewidth=4, linestyle='-', color='deepskyblue') 
line = ax.plot(tpoints[0:len(tpoints)-2], changeRates[:,7], label='inter A-I', linewidth=4, linestyle=':', color='deepskyblue') 
line = ax.plot(tpoints[0:len(tpoints)-2], changeRates[:,8], label='intra H-E', linewidth=4, linestyle='-', color='dodgerblue') 
line = ax.plot(tpoints[0:len(tpoints)-2], changeRates[:,9], label='inter H-E', linewidth=4, linestyle=':', color='dodgerblue') 
line = ax.plot(tpoints[0:len(tpoints)-2], changeRates[:,10], label='intra H-I', linewidth=4, linestyle='-', color='steelblue') 
line = ax.plot(tpoints[0:len(tpoints)-2], changeRates[:,11], label='inter H-I', linewidth=4, linestyle=':', color='steelblue') 

 
plt.gca().legend(loc='upper right', fontsize=7, frameon=True, fancybox=True, shadow=True)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.gca().set_ylabel('Mean change rate of weights', fontsize=7)

if animated:
    def animateRateChangeWeights(frame):
        ax.set_xlim(frame*0.02, frame*0.02+0.5)   #0.01s step, 0.5s window
    speedFactor = 10
    ani = animation.FuncAnimation(fig, animateRateChangeWeights, interval=10*speedFactor, blit=False, frames=50*round(dt*len(changeRates[:,0])), cache_frame_data=False, repeat=False) #50 frames per simulation second

if adimensional :
    plt.gca().set_xlabel('Time', fontsize=7)
else :
    plt.gca().set_xlabel('Time (h)', fontsize=7)
if save : 
    #plt.gca().set_xlim(2768.0, 2770.0)
    #plt.gca().set_xlim(2729, 2731)
    #plt.gca().set_xlim(2690.5, 2692.5)
    #plt.gca().set_ylim(-0.003, 0.003)
    plt.gca().set_xlim(1800, 16200)
    
    plt.xticks((1800,9000,16200),(0,2,4), fontsize=7)
    plt.yticks((-1,-0.5,0,0.5,1),(-1,-0.5,0,0.5,1), fontsize=7)
    
    
    plt.savefig('results/rateChange_weights.pdf', dpi=300, bbox_inches='tight')
   # plt.savefig('results/rateChange_weights.png', dpi=300, bbox_inches='tight')
    if animated:
        plt.gcf().set_size_inches(16, 9)
        writer = animation.FFMpegWriter(bitrate=-1, codec="libx264", extra_args=['-pix_fmt', 'yuv420p'])
        #ani.save("results/rateChange_weights.mp4", writer=writer, dpi=80)
        ani.save("results/rateChange_weights.gif", writer="pillow", fps=1, dpi=80)
    plt.close()
else :
    plt.gca().set_title('Average evolution of the change rate of weights', fontsize=25)
    plt.show() 
  
"""
    
"""


#Dynamics


"""
"""   
#Time development of the order parameters
fig, ax = plt.subplots()
ax.plot(tpoints[0:len(orderParameter1_1)], orderParameter1_1, label='Cluster 1', color='green') 
ax.plot(tpoints[0:len(orderParameter1_2)], orderParameter1_2, label='Cluster 2', color='orange')
ax.plot(tpoints[0:len(orderParameter1)], orderParameter1, label='Network', color='grey') 
#ax.plot(tpoints[0:len(orderParameter1)], orderParameter1, label='Network', color='magenta', linewidth=4) 
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.gca().set_ylabel('Order parameter', fontsize=20)
#add orders if necessary
plt.gca().legend(fontsize=20)

if animated:
    def animateOrderParameters(frame):
        ax.set_xlim(frame*0.02, frame*0.02+0.5)   #0.02s step, 0.5s window
    speedFactor = 10
    ani = animation.FuncAnimation(fig, animateOrderParameters, interval=10*speedFactor, blit=False, frames=50*round(dt*len(orderParameter1_1)), cache_frame_data=False, repeat=False) #50 frames per simulation second

if adimensional :
    plt.gca().set_xlabel('Time', fontsize=20)
else :
    plt.gca().set_xlabel('Time (s)', fontsize=20)
if save :
    
    #plt.gca().set_xlim(950.0, 1000)
    #plt.gca().set_xlim(0.0, 50)
    #plt.gca().set_ylim(0.0, 1.0)
    plt.gca().set_xlim(10.0, 25.0) 
    #plt.axis('off')
    #plt.gcf().set_size_inches(25.0, 2.0)

    plt.savefig('results/order_parameters.png', dpi=300, bbox_inches='tight')
    if animated:
        plt.gcf().set_size_inches(16, 9)
        writer = animation.FFMpegWriter(bitrate=-1, codec="libx264", extra_args=['-pix_fmt', 'yuv420p'])
        #ani.save("results/order_parameters.mp4", writer=writer)
        ani.save("results/order_parameters.gif", writer="pillow", fps=1)
    plt.close()
else :
    plt.gca().set_title('Evolution of the Kuramoto order parameters', fontsize=25)
    plt.show() 
"""


#Spikes evolution

colorsCodes = [newred if inhibitory[i]==1 else newblue for i in range(len(inhibitory))]
fig, ax = plt.subplots()

spikes_sorted = [spikes[i] for i in order]
colorsCodes_sorted = [colorsCodes[i] for i in order]
#ax.eventplot(spikes_sorted, colors=colorsCodes_sorted, lineoffsets=1, linelengths=1.0, linewidths=2)

# Assuming spikes_sorted is a list of lists, where each list represents spike times for a specific event
spike_times = []
spike_ids = []
colors = []

# Flatten the list of events and assign corresponding y-axis values
for i, spike_train in enumerate(spikes_sorted):
    spike_times.extend(spike_train)
    spike_ids.extend([i + 1] * len(spike_train))  # y-axis values (1-based indexing)
    colors.extend([colorsCodes_sorted[i]] * len(spike_train))  # same color per event


ax.scatter(spike_times, spike_ids, c=colors, s=10.0, edgecolors='none', marker='.')  # 's' is the size of the points #[::5] to have 1/5 data s=1.0 for 200,000N



#ax.eventplot(spikes, colors=colorsCodes, lineoffsets=1, linelengths=1.0, linewidths=2)
#if adimensional :
#    plt.gca().set_xlabel('Time', fontsize=20)
#else :
plt.gca().set_xlabel('Time (s)', fontsize=15)
plt.gca().set_ylabel('Neurons', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

if animated:
    def animateSpikesEvolution(frame):
        #ax.set_xlim(frame*0.02, frame*0.02+0.5)   #0.02s step, 0.5s window
        ax.set_xlim(2660+frame*0.25, 2690+frame*0.25)   #0.02s step, 30s window
    speedFactor = 10
    ani = animation.FuncAnimation(fig, animateSpikesEvolution, interval=10*speedFactor, blit=False, frames=200, cache_frame_data=False, repeat=False) #50 frames per simulation second 50*round(dt*iterations)

if save : 
    
    plt.gca().set_xlim(0.0, 50.0)
    plt.gcf().set_size_inches(25.0, 2.0)
    #plt.gca().set_xlim(10.0, 50.0)
    #plt.gcf().set_size_inches(9.0, 2.0)
    plt.savefig('results/spikes_evolution_simu.pdf', dpi=300, bbox_inches='tight')
    plt.gca().autoscale()
    if animated:
        plt.gca().set_title('Spike trains', fontsize=15)
        plt.gcf().set_size_inches(9, 5)
        writer = animation.FFMpegWriter(bitrate=-1, codec="libx264", extra_args=['-pix_fmt', 'yuv420p'])
        #ani.save("results/spikes_evolution.mp4", writer=writer)
        ani.save("results/spikes_evolution.gif", writer="pillow", fps=4, dpi=80)
    plt.close()
else : 
    plt.gca().set_title('Spikes of neurons through the time', fontsize=25)
    plt.show()




plt.figure(figsize=(6.0,1.0))
plt.axis('off')
#plt.scatter(spikes_exc_ha[0], spikes_exc_ha[1], s=1, lw=0, color=newred)
#plt.scatter(spikes_inh_ha[0], spikes_inh_ha[1], s=1, lw=0, color=newblue)
#plt.scatter(spikesMatrix[i], spikesMatrix[i], s=1, lw=0, color=colorsCodes)
#plt.eventplot(spikes, colors=colorsCodes, lineoffsets=1, linelengths=1.0, linewidths=2)
plt.scatter(spike_times, spike_ids, c=colors, s=10.0, edgecolors='none', marker='.')  # 's' is the size of the points, s=1.0 for 200,000N
# plt.xlabel( 'Time (seconds)', fontsize=10)
# plt.ylabel( 'Neuron index', fontsize=10)
# plt.tick_params(labelsize=8)
plt.xlim(0,50)
#plt.xlim(25,55)
plt.tight_layout()

if save:
    #plt.savefig('results/spikes_evolution_simu_paper.pdf', bbox_inches='tight', dpi=300 )
    plt.savefig('results/spikes_evolution_simu_paper.png', bbox_inches='tight', dpi=300 )
   
    



plt.figure(figsize=(6.0,2.0))
plt.axis('off')
plt.scatter(spike_times, spike_ids, c=colors, s=10.0, edgecolors='none', marker='.')  # 's' is the size of the points, s=1.0 for 200,000N
plt.xlim(1800,1820)
#plt.xlim(25,55)
plt.tight_layout()

if save:
    plt.savefig('results/spikes_evolution_simu_paper_0H.png', bbox_inches='tight', dpi=300 )


plt.figure(figsize=(6.0,2.0))
plt.axis('off')
plt.scatter(spike_times, spike_ids, c=colors, s=10.0, edgecolors='none', marker='.')  # 's' is the size of the points, s=1.0 for 200,000N
plt.xlim(9000,9020)
plt.tight_layout()

if save:
    plt.savefig('results/spikes_evolution_simu_paper_2H.png', bbox_inches='tight', dpi=300 )
    
    
plt.figure(figsize=(6.0,2.0))
plt.axis('off')
plt.scatter(spike_times, spike_ids, c=colors, s=10.0, edgecolors='none', marker='.')  # 's' is the size of the points, s=1.0 for 200,000N
plt.xlim(16200,16220)
plt.tight_layout()

if save:
    plt.savefig('results/spikes_evolution_simu_paper_4H.png', bbox_inches='tight', dpi=300 )



"""  
#Spike counts 
i=8     #index of the memory
selected_neurons = [0+i, 33+2*i, 34+2*i]
selected_spikes = np.concatenate([spikes[i] for i in selected_neurons])
all_spikes = np.concatenate(spikes)

# Bins
#bin_size = 0.15  # Bin size in seconds
bin_size = 0.2  # Bin size in seconds
bins = np.arange(0, tfinal + bin_size, bin_size)

# histograms
hist_selected, _ = np.histogram(selected_spikes, bins=bins)
hist_total, _ = np.histogram(all_spikes, bins=bins)
ratio = np.where(hist_total != 0, hist_selected / hist_total, np.nan)

plt.figure(figsize=(10, 2))
plt.hist(all_spikes, bins=bins, color='blue', alpha=0.7, edgecolor='black')
#plt.bar(bins[:-1], ratio, width=bin_size, color='purple', alpha=0.7, edgecolor='black')

#plt.gca().set_xlim(950.0, 960.0)
#plt.gca().set_ylim(0.0, 1.0)
#plt.axis('off')
#plt.grid(True, linestyle="--", alpha=0.6)

if save:
    plt.savefig('results/spike_counts.png', dpi=300, bbox_inches='tight')
else:
    plt.show()   

plt.figure(figsize=(6.0,1.0))
plt.axis('off')
plt.hist(all_spikes, bins=bins, color='purple', alpha=0.7, edgecolor='black')
plt.gca().set_xlim(1800,1820)
plt.gca().set_ylim(0,100)
plt.tight_layout()

if save:
    plt.savefig('results/hist_evolution_simu_paper_0H.png', bbox_inches='tight', dpi=300 )


plt.figure(figsize=(6.0,1.0))
plt.axis('off')
plt.hist(all_spikes, bins=bins, color='purple', alpha=0.7, edgecolor='black')
plt.gca().set_xlim(9000,9020)
plt.gca().set_ylim(0,100)
plt.tight_layout()

if save:
    plt.savefig('results/hist_evolution_simu_paper_2H.png', bbox_inches='tight', dpi=300 )
    
    
plt.figure(figsize=(6.0,1.0))
plt.axis('off')
plt.hist(all_spikes, bins=bins, color='purple', alpha=0.7, edgecolor='black')
plt.gca().set_xlim(16200,16220)
plt.gca().set_ylim(0,100)
plt.tight_layout()

if save:
    plt.savefig('results/hist_evolution_simu_paper_4H.png', bbox_inches='tight', dpi=300 )

"""    


#Firing rates  

#Evolution mean firing rates of the network
meanFrequenciesNetwork = np.sum(allMeanFrequencies, axis=0)/len(allMeanFrequencies)

allMeanFrequencies1 = allMeanFrequencies[0:int(nbExcit/2), :]
allMeanFrequencies2 = allMeanFrequencies[int(nbExcit/2):nbExcit, :]
#allMeanFrequencies3 = allMeanFrequencies[40:60, :]
#allMeanFrequencies4 = allMeanFrequencies[60:80, :]

meanFrequenciesNetwork1 = np.sum(allMeanFrequencies1, axis=0)
meanFrequenciesNetwork1 = meanFrequenciesNetwork1/len(allMeanFrequencies1)

meanFrequenciesNetwork2 = np.sum(allMeanFrequencies2, axis=0)
meanFrequenciesNetwork2 = meanFrequenciesNetwork2/len(allMeanFrequencies2)

#meanFrequenciesNetwork3 = np.sum(allMeanFrequencies3, axis=0)
#meanFrequenciesNetwork3 = meanFrequenciesNetwork3/len(allMeanFrequencies3)

#meanFrequenciesNetwork4 = np.sum(allMeanFrequencies4, axis=0)
#meanFrequenciesNetwork4 = meanFrequenciesNetwork4/len(allMeanFrequencies4)
"""
fig, ax = plt.subplots()
#ax.plot(tpoints, meanFrequenciesNetwork, label='Network')
ax.plot(tpoints, meanFrequenciesNetwork1, label='Cluster 1', color='green', linewidth=4) 
ax.plot(tpoints, meanFrequenciesNetwork2, label='Cluster 2', color='orange', linewidth=4)  
#ax.plot(tpoints, meanFrequenciesNetwork3, label='Cluster 3') 
#ax.plot(tpoints, meanFrequenciesNetwork4, label='Cluster 4') 
if adimensional :
    plt.gca().set_xlabel('Time', fontsize=20)
else :
    plt.gca().set_xlabel('Time (s)', fontsize=20)
plt.gca().set_ylabel('Mean firing rate (Hz)', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.gca().legend(fontsize=20)

if animated:
    def animateMeanFiringRatesEvolutionNetwork(frame):
        ax.set_xlim(frame*0.02, frame*0.02+0.5)   #0.02s step, 0.5s window
    speedFactor = 10
    ani = animation.FuncAnimation(fig, animateMeanFiringRatesEvolutionNetwork, interval=10*speedFactor, blit=False, frames=50*round(dt*len(tpoints)), cache_frame_data=False, repeat=False) #50 frames per simulation second
    
if save : 
    #plt.gca().set_xlim(2768.0, 2770.0)
    #plt.gca().set_xlim(2729, 2731)
    plt.gca().set_xlim(2690.5, 2692.5)
    plt.gca().set_ylim(-0.2, 4.0)
    plt.savefig('results/mean_firing_rate_evolution_network.png', dpi=300, bbox_inches='tight')
    if animated:
        plt.gcf().set_size_inches(16, 9)
        writer = animation.FFMpegWriter(bitrate=-1, codec="libx264", extra_args=['-pix_fmt', 'yuv420p'])
        #ani.save("results/mean_firing_rate_evolution_network.mp4", writer=writer)
        ani.save("results/mean_firing_rate_evolution_network.gif", writer="pillow", fps=1, dpi=80)
    plt.close()
else :
    plt.gca().set_title('Mean firing rates of the network', fontsize=25)
    plt.show() 
""" 
   
    
    
#Calculate PDF firing rate
cst = 0.00000000001 #to avoid null PDF (for log in free energy)
counts, bins = np.histogram(meanFrequenciesNetwork[startIterations:], bins=30)
counts = counts+cst
pdf_Network = counts / sum(counts)

counts, bins = np.histogram(meanFrequenciesNetwork1[startIterations:], bins=30)
counts = counts+cst
pdf_cluster1= counts / sum(counts)

counts, bins = np.histogram(meanFrequenciesNetwork2[startIterations:], bins=30)
counts = counts+cst
pdf_cluster2 = counts / sum(counts)

std_cluster1 = np.std(meanFrequenciesNetwork1[startIterations:])
std_cluster2 = np.std(meanFrequenciesNetwork2[startIterations:])
std_Network = np.std(meanFrequenciesNetwork[startIterations:])

mean_cluster1 = np.mean(meanFrequenciesNetwork1[startIterations:])
mean_cluster2 = np.mean(meanFrequenciesNetwork2[startIterations:])
mean_Network = np.mean(meanFrequenciesNetwork[startIterations:])


#Plot pdf
fig, ax = plt.subplots(figsize=(4.0,2.0), dpi=300) 
ax.plot(bins[:-1], pdf_cluster1, label='Cluster 1', color='green')
ax.plot(bins[:-1], pdf_cluster2, label='Cluster 2', color='orange')
#ax.plot(bins[:-1], pdf_Network, label='Network', color='grey')

plt.axvline(mean_cluster1, linestyle='dashed', linewidth=1, label='Mean', color='green')
plt.errorbar(mean_cluster1, max(pdf_cluster1), xerr=std_cluster1, fmt='-o', capsize=5, label='Std Dev', color='green')
plt.axvline(mean_cluster2, linestyle='dashed', linewidth=1, label='Mean', color='orange')
plt.errorbar(mean_cluster2, max(pdf_cluster2), xerr=std_cluster2, fmt='-o', capsize=5, label='Std Dev', color='orange')

#plt.gca().set_xlabel('R', fontsize=20)
#plt.gca().set_ylabel('Probability', fontsize=20)
plt.yscale('log')
plt.xticks([0.0, 5.0, 10.0, 15.0], fontsize=13)
plt.yticks([], fontsize=13)     #[0.0, 0.4, 0.8]
fig.tight_layout()

plt.gca().legend(bbox_to_anchor=(0.5, 1.52), loc='upper center', fontsize=13, ncol=3, columnspacing=0.5)
if save :
    plt.savefig('results/pdf_firing_rate.pdf', dpi=300, bbox_inches='tight')
    plt.close()
else :
    plt.gca().set_title('Probability Distribution Function', fontsize=25)
    plt.show() 



#Calculate PDF mean change rate of weights
cst = 0.00000000001 #to avoid null PDF (for log in free energy)
bin_min = -0.02
bin_max = 0.04
num_bins = 100
counts, bins = np.histogram(changeRates[:,0], bins=np.linspace(bin_min, bin_max, num_bins))
counts = counts+cst
pdf_Network = counts / sum(counts)

counts, bins = np.histogram(changeRates[:,1], bins=np.linspace(bin_min, bin_max, num_bins))
counts = counts+cst
pdf_cluster1= counts / sum(counts)

counts, bins = np.histogram(changeRates[:,2], bins=np.linspace(bin_min, bin_max, num_bins))
counts = counts+cst
pdf_cluster2 = counts / sum(counts)

std_cluster1 = np.std(changeRates[:,1])
std_cluster2 = np.std(changeRates[:,2])
std_Network = np.std(changeRates[:,0])

mean_cluster1 = np.mean(changeRates[:,1])
mean_cluster2 = np.mean(changeRates[:,2])
mean_Network = np.mean(changeRates[:,0])


#Plot pdf
fig, ax = plt.subplots(figsize=(4.0,2.0), dpi=300) 
ax.plot(bins[:-1], pdf_cluster1, label='Cluster 1', color='green')
ax.plot(bins[:-1], pdf_cluster2, label='Cluster 2', color='orange')
#ax.plot(bins[:-1], pdf_Network, label='Network', color='grey')

plt.axvline(mean_cluster1, linestyle='dashed', linewidth=1, label='Mean', color='green')
plt.errorbar(mean_cluster1, max(pdf_cluster1), xerr=std_cluster1, fmt='-o', capsize=5, label='Std Dev', color='green')
plt.axvline(mean_cluster2, linestyle='dashed', linewidth=1, label='Mean', color='orange')
plt.errorbar(mean_cluster2, max(pdf_cluster2), xerr=std_cluster2, fmt='-o', capsize=5, label='Std Dev', color='orange')

#plt.gca().set_xlabel('R', fontsize=20)
#plt.gca().set_ylabel('Probability', fontsize=20)
plt.yscale('log')
plt.xticks([-0.02, 0.0, 0.02, 0.04], fontsize=13)
plt.yticks([], fontsize=13)
fig.tight_layout()

plt.gca().legend(bbox_to_anchor=(0.5, 1.52), loc='upper center', fontsize=13, ncol=3, columnspacing=0.5)
if save :
    plt.savefig('results/pdf_change_weights.pdf', dpi=300, bbox_inches='tight')
    plt.close()
else :
    plt.gca().set_title('Probability Distribution Function', fontsize=25)
    plt.show() 
    


#Calculate the PDF and the free energy of the Kuramoto Order parameter
#Calculate PDF
cst = 0.00000000001 #to avoid null PDF (for log in free energy)
counts, bins = np.histogram(orderParameter1[startIterations:], bins=100)
counts = counts+cst
pdf_Network = counts / sum(counts)

counts, bins = np.histogram(orderParameter1_1[startIterations:], bins=100)
counts = counts+cst
pdf_cluster1= counts / sum(counts)

counts, bins = np.histogram(orderParameter1_2[startIterations:], bins=100)
counts = counts+cst
pdf_cluster2 = counts / sum(counts)

std_cluster1 = np.std(orderParameter1_1[startIterations:])
std_cluster2 = np.std(orderParameter1_2[startIterations:])
std_Network = np.std(orderParameter1[startIterations:])

mean_cluster1 = np.mean(orderParameter1_1[startIterations:])
mean_cluster2 = np.mean(orderParameter1_2[startIterations:])
mean_Network = np.mean(orderParameter1[startIterations:])


#Plot pdf
fig, ax = plt.subplots(figsize=(4.0,1.75), dpi=300) 
ax.plot(bins[:-1], pdf_cluster1, label='Cluster 1', color='green')
ax.plot(bins[:-1], pdf_cluster2, label='Cluster 2', color='orange')
ax.plot(bins[:-1], pdf_Network, label='Network', color='grey')

plt.axvline(mean_cluster1, linestyle='dashed', linewidth=1, label='Mean', color='green')
plt.errorbar(mean_cluster1, max(pdf_cluster1), xerr=std_cluster1, fmt='-o', capsize=5, label='Std Dev', color='green')
plt.axvline(mean_cluster2, linestyle='dashed', linewidth=1, label='Mean', color='orange')
plt.errorbar(mean_cluster2, max(pdf_cluster2), xerr=std_cluster2, fmt='-o', capsize=5, label='Std Dev', color='orange')
plt.axvline(mean_Network, linestyle='dashed', linewidth=1, label='Mean', color='grey')
plt.errorbar(mean_Network, max(pdf_Network), xerr=std_Network, fmt='-o', capsize=5, label='Std Dev', color='grey')

#plt.gca().set_xlabel('R', fontsize=20)
#plt.gca().set_ylabel('Probability', fontsize=20)
plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=13)
plt.yticks([], fontsize=13)
fig.tight_layout()

plt.gca().legend(bbox_to_anchor=(0.5, 1.85), loc='upper center', fontsize=13, ncol=3, columnspacing=0.5) #1.83 trop petit
if save :
    plt.savefig('results/pdf_order_parameter.pdf', dpi=300, bbox_inches='tight')
    plt.close()
else :
    plt.gca().set_title('Probability Distribution Function', fontsize=25)
    plt.show() 
    
  

#MIX PLOTS RESULTS  
colorsCodes = [newred if inhibitory[i]==1 else newblue for i in range(len(inhibitory))]
fig, axs = plt.subplots(4, sharex=True)
plt.gca().set_xlabel('Time (s)', fontsize=15) #, fontsize=20

#axs[0].eventplot(spikes, colors=colorsCodes, lineoffsets=1, linelengths=1.0, linewidths=2)
axs[0].scatter(spike_times, spike_ids, c=colors, s=10.0, edgecolors='none', marker='.')  # 's' is the size of the points #[::5] to have 1/5 data s=1.0 for 200,000N
axs[0].set_ylabel('Neurons', fontsize=15)  #, fontsize=20
axs[0].set_yticks((0,int(nbExcit/2),nbExcit,nbNeurons))
axs[0].set_yticklabels((0,int(nbExcit/2),nbExcit,nbNeurons), fontsize=15)


axs[1].plot(tpoints, orderParameter1_1, label='Cluster 1', color='green') 
axs[1].plot(tpoints, orderParameter1_2, label='Cluster 2', color='orange')
axs[1].plot(tpoints, orderParameter1, label='Network', color='grey') 
axs[1].set_yticks((0,0.5,1))
axs[1].set_yticklabels((0,0.5,1), fontsize=15)
axs[1].set_ylabel('Order \nparameter', fontsize=15)       #, fontsize=20
#axs[1].legend(fontsize=20)

axs[2].plot(tpoints, meanFrequenciesNetwork1, color='green') #, label='Cluster 1'
axs[2].plot(tpoints, meanFrequenciesNetwork2, color='orange')  #, label='Cluster 2'
#axs[2].plot(tpoints, meanFrequenciesNetwork, color='grey') 
axs[2].set_yticks((0,10,20))
axs[2].set_yticklabels((0,10,20), fontsize=15)
axs[2].set_ylabel('Mean firing \nrate (Hz)', fontsize=15) #, fontsize=20
#axs[2].legend(fontsize=20)

axs[3].plot(tpoints[0:len(tpoints)-1], changeRates[:,1], color='green')  #, label='Cluster 1'
axs[3].plot(tpoints[0:len(tpoints)-1], changeRates[:,2], color='orange')  #, label='Cluster 2'
#axs[3].plot(tpoints[0:len(tpoints)-1], changeRates[:,0], color='grey')
axs[3].set_yticks((0.00,0.01,0.02))
axs[3].set_yticklabels((0.00,0.01,0.02), fontsize=15)
#axs[3].legend(fontsize=20)
axs[3].set_ylabel('Mean change \nrate of weights', fontsize=15) #, fontsize=20

#axs[4].plot(tpoints[0:len(phases_network)], np.sum(phases_network[:, 0:40], axis=1)/40.0)  #, label='Cluster 1'
#axs[4].plot(tpoints[0:len(phases_network)], np.sum(phases_network[:, 40:80], axis=1)/40.0)  #, label='Cluster 2'
#axs[4].set_ylabel('Phase') #, fontsize=20

#fig.legend(bbox_to_anchor=(0.5, -0.09), loc='lower center', fontsize=4, ncol=3)
fig.legend(fontsize=15)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

if animated:
    def animateMix(frame):
        for ax in axs.flat:
            axs[0].set_xlim(2660+frame*0.25, 2690+frame*0.25)   #1.0s step, 0.5s window
        axs[3].set_ylim(top=0.025)
    speedFactor = 10
    ani = animation.FuncAnimation(fig, animateMix, interval=10*speedFactor, blit=False, frames=200, cache_frame_data=False, repeat=False) #50 frames per simulation second 50*round(dt*len(tpoints))
    
if save : 
    #plt.gca().set_xlim(1830, 1860)
    #axs[3].set_xlim(3320, 3360)
    #axs[3].set_xlim(2740, 2780)
    #axs[3].set_xlim(2700, 2735)
    axs[3].set_xlim(2670, 2700)
    #axs[3].set_xlim(86360, 86400)
    #axs[3].set_xlim(172600, 172630)
    axs[3].set_ylim(top=0.025)
    fig.tight_layout()
    fig.set_size_inches(3.28, 3.88) 
    #plt.savefig('results/mix.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/mix.pdf', dpi=300, bbox_inches='tight')
    plt.gca().autoscale()
    
   
    if animated:
        plt.gcf().set_size_inches(16, 9)
        writer = animation.FFMpegWriter(bitrate=-1, codec="libx264", extra_args=['-pix_fmt', 'yuv420p'])
        #ani.save("results/mix.mp4", writer=writer)
        ani.save("results/mix.gif", writer="pillow", fps=4, dpi=80)
    plt.close()
else :
    plt.show() 
   


#Calculate the ISIs by subtracting each spike time from the preceding spike time.
IsIs = [np.diff(st) for st in spikes]

#Calculate the CV as the ratio of the mean (μ) and standard deviation (σ) of the ISIs
CVs = [np.std(i) / np.mean(i) for i in IsIs]

poissonDistribution = np.ones(len(CVs))


plt.figure(figsize=(4.0,2.7), dpi=300)
#Plot distribution of the coefficient of variation of the neurons 
plt.hist(CVs, bins=30, histtype=u'step', density=True, color='grey', label='Network')
plt.hist(poissonDistribution, bins=30, histtype=u'step', density=True, color='gold', label='Poisson process')

#plt.gca().set_xlabel('Coefficient of variation interspike intervals', fontsize=20)
#plt.gca().set_ylabel('Population', fontsize=20)
plt.xticks([0.5, 1, 1.5], fontsize=13)
plt.yticks([], fontsize=13)

plt.tight_layout()
#plt.gca().legend(loc='upper right', fontsize=13)
plt.gca().legend(bbox_to_anchor=(0.5, 1.23), loc='upper center', fontsize=13, ncol=2, columnspacing=0.5) #1.2 pas assez 1.24 trop large
if save :
    #plt.xlim([0.0, 1.5]) 
    plt.savefig('results/distribution_CVs.pdf', dpi=300, bbox_inches='tight')
    plt.close()
else :
    plt.gca().set_title('Distribution of the coefficient of variation of the interspike intervals of neurons', fontsize=25)
    plt.show() 


