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
from datetime import datetime
import community as community_louvain
import itertools
import cv2
#import pycochleagram.cochleagram as cgram
#from pycochleagram import utils
from scipy.io.wavfile import write
from playsound import playsound


warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


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


def getMeanFrequenciesAllTimes(N, T, P, spikesMatrix, timeConstant) :
    """
        Create a matrix of mean frequencies of each neurons at each time step
        
        Parameters : 
        N -- number of neurons
        T -- number of iterations of the simulation
        P -- period for the mean (in second)
        spikesMatrix -- matrix for each spikes of each neurons
        timeConstant -- constant factor time per iteration (precision)
    """
    
    meanFrequencies = np.zeros((N, T))
    
    for i in range(N) :     #For each neurons
        for t in range(int(P/timeConstant), T) :         #For each time
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
tfinal = 40.0         #duration simulation in s   //50.0 4000.0
iterations = int(tfinal/dt)  
tpoints = np.arange(0.0, (tfinal+0.00001), dt)



"""Get data saved"""   
    
matrices = np.loadtxt('weights_matrices.txt', dtype=float )
matrices = matrices.reshape(4,100,100)


f = open("changeRates.txt", "r")
changeRates = []
for x in f:
    lst = x.split()
    changeRates.append([float(i) for i in lst])

f.close()
changeRates = np.array(changeRates)
changeRates = changeRates[1:, :]
changeRates = changeRates * 10.0  #normalize by delta t = 0.1s

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



""" Process data """ 


#Calculate order parameters through the time
orderParameter1 = []    #list of order parameters (m=1)
orderParameter1_1 = []  #list of order parameters (m=1) cluster 1
orderParameter1_2 = []  #list of order parameters (m=1) cluster 2
phases_network = []
for t in range(0, iterations) :
    #Get indices of closest spikes after time t
    indices = [np.searchsorted(spikes[i], t*dt, side='right') for i in range(0, 100)]
            
    #calculate times spikes, calculate phases and order parameter
    tn = [spikes[i][indices[i]-1] for i in range(0, 100)]
    tn1 = [spikes[i][indices[i]] for i in range(0, 100)]
    phases = [2.0 * np.pi * ((t * dt) - tn[i]) / (tn1[i] - tn[i] + 0.0000001) for i in range(0, 100)]
    phases_network.append(phases)
    
    orderParameter1.append(orderParameter(phases, 1, 100))      
    orderParameter1_1.append(orderParameter(phases[0:40], 1, 40))    #calculate and register the order parameters (m=1)
    orderParameter1_2.append(orderParameter(phases[40:80], 1, 40))    #calculate and register the order parameters (m=1)

#Artificially create order parameters last iteration to fit dimensions
orderParameter1_1.append(orderParameter(phases[0:40], 1, 40))    #calculate and register the order parameters (m=1)
orderParameter1_2.append(orderParameter(phases[40:80], 1, 40))    #calculate and register the order parameters (m=1)
orderParameter1.append(orderParameter(phases, 1, 100))       #calculate and register the order parameters (m=1)
phases_network.append(phases)

spikes = [np.delete(x, -1) for x in spikes]     #Remove artifiacial last spike
spikes = np.array(spikes,dtype=object)
phases_network = np.array(phases_network)

#Calculate mean frequencies and order of neurons
allMeanFrequencies = getMeanFrequenciesAllTimes(len(inhibitory), iterations+1, 0.05, spikes, dt)   #mean for periods of 0.05s




"""Plot data""" 

newcmap = np.genfromtxt('./Matplotlib_colourmap_div.csv', delimiter=',')
cmap_div = ListedColormap(newcmap)
newblue = (0.21961,0.33333,0.64706)
newred = (0.7098,0.070588,0.070588)



#Connectivity

#Weights matrix 
#kij : j presynaptic to i postsynaptic neurons
plt.figure(figsize=(1.8,1.5), dpi=300)
plt.imshow(matrices[0], cmap=cmap_div, vmin=-1, vmax=1, aspect='auto', origin='lower')
plt.clim(-1,1)
#plt.gca().set_xlabel('Presynaptic neurons j', fontsize=20)
#plt.gca().set_ylabel('Postsynaptic neurons i', fontsize=20)
plt.gca().xaxis.tick_bottom()
#plt.gca().invert_yaxis()
plt.xticks((0,40,80),(0,40,80), fontsize=7)
plt.yticks((0,40,80),(0,40,80), fontsize=7)
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
plt.imshow(matrices[1], cmap=cmap_div, vmin=-1, vmax=1, aspect='auto', origin='lower')
plt.clim(-1,1)
#plt.gca().set_xlabel('Presynaptic neurons j', fontsize=20)
#plt.gca().set_ylabel('Postsynaptic neurons i', fontsize=20)
plt.gca().xaxis.tick_bottom()
#plt.gca().invert_yaxis()
plt.xticks((0,40,80),(0,40,80), fontsize=7)
plt.yticks((0,40,80),(0,40,80), fontsize=7)
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
plt.imshow(matrices[2], cmap=cmap_div, vmin=-1, vmax=1, aspect='auto', origin='lower')
plt.clim(-1,1)
#plt.gca().set_xlabel('Presynaptic neurons j', fontsize=20)
#plt.gca().set_ylabel('Postsynaptic neurons i', fontsize=20)
plt.gca().xaxis.tick_bottom()
#plt.gca().invert_yaxis()
plt.xticks((0,40,80),(0,40,80), fontsize=7)
plt.yticks((0,40,80),(0,40,80), fontsize=7)
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
plt.imshow(matrices[-1], cmap=cmap_div, vmin=-1, vmax=1, aspect='auto', origin='lower')
plt.clim(-1,1)
#plt.gca().set_xlabel('Presynaptic neurons j', fontsize=20)
#plt.gca().set_ylabel('Postsynaptic neurons i', fontsize=20)
plt.gca().xaxis.tick_bottom()
#plt.gca().invert_yaxis()
plt.xticks((0,40,80),(0,40,80), fontsize=7)
plt.yticks((0,40,80),(0,40,80), fontsize=7)
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
        ax.set_title('Weight matrix (sorted) T%d' % (frame%len(matrices)), fontsize=25)
        ax.matshow(matrices[frame%len(matrices)], cmap=cmap_div, vmin=-1, vmax=1)
        plt.gca().xaxis.tick_bottom()
        plt.gca().invert_yaxis()
    ani = animation.FuncAnimation(fig, animateWeightsMatrices, interval=2000, blit=False, frames=20, cache_frame_data=False, repeat=True)
    if save : 
        plt.gcf().set_size_inches(16, 9)
        #writer = animation.FFMpegWriter(bitrate=-1, codec="libx264", extra_args=['-pix_fmt', 'yuv420p'])
        ani.save("results/weights_matrix_sorted.mp4")
        plt.close()
    else :
        plt.show()


#Distribution of the weights
plt.figure(figsize=(4.0,3.2), dpi=300)
plt.hist(matrices[0].flatten(), bins=100, histtype=u'step', density=True, color='lime', label='T=0s', log=True) #Beginning of the simulation
plt.hist(matrices[2].flatten(), bins=100, histtype=u'step', density=True, color='cyan', label='T=400s', log=True) #in the middle
plt.hist(matrices[-1].flatten(), bins=100, histtype=u'step', density=True, color='magenta', label='T=4000s', log=True) #End of the simulation
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
    

#Evolution of the change rate of weights
fig, ax = plt.subplots()
#line = ax.plot(tpoints[0:len(tpoints)-1], changeRates[:,0], label='Absolute change network', color='grey')
line2 = ax.plot(tpoints[0:len(tpoints)-1], changeRates[:,1], label='Cluster 1', color='green', linewidth=4) 
line3 = ax.plot(tpoints[0:len(tpoints)-1], changeRates[:,2], label='Cluster 2', color='orange', linewidth=4) 
plt.gca().legend(fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.gca().set_ylabel('Mean change rate of weights', fontsize=20)

if animated:
    def animateRateChangeWeights(frame):
        ax.set_xlim(frame*0.02, frame*0.02+0.5)   #0.01s step, 0.5s window
    speedFactor = 10
    ani = animation.FuncAnimation(fig, animateRateChangeWeights, interval=10*speedFactor, blit=False, frames=50*round(dt*len(changeRates[:,0])), cache_frame_data=False, repeat=False) #50 frames per simulation second

if adimensional :
    plt.gca().set_xlabel('Time', fontsize=20)
else :
    plt.gca().set_xlabel('Time (s)', fontsize=20)
if save : 
    #plt.gca().set_xlim(2768.0, 2770.0)
    #plt.gca().set_xlim(2729, 2731)
    plt.gca().set_xlim(2690.5, 2692.5)
    plt.gca().set_ylim(-0.003, 0.003)
    plt.savefig('results/rateChange_weights.png', dpi=300, bbox_inches='tight')
    if animated:
        plt.gcf().set_size_inches(16, 9)
        writer = animation.FFMpegWriter(bitrate=-1, codec="libx264", extra_args=['-pix_fmt', 'yuv420p'])
        ani.save("results/rateChange_weights.mp4", writer=writer)
    plt.close()
else :
    plt.gca().set_title('Average evolution of the change rate of weights', fontsize=25)
    plt.show() 
  

    
"""


#Dynamics


"""
    
#Time development of the order parameters
fig, ax = plt.subplots()
ax.plot(tpoints[0:len(orderParameter1_1)], orderParameter1_1, label='Cluster 1', color='green') 
ax.plot(tpoints[0:len(orderParameter1_2)], orderParameter1_2, label='Cluster 2', color='orange')
ax.plot(tpoints[0:len(orderParameter1)], orderParameter1, label='Network', color='grey') 
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
    plt.gca().set_xlim(10.0, 25.0) 
    plt.savefig('results/order_parameters.png', dpi=300, bbox_inches='tight')
    if animated:
        plt.gcf().set_size_inches(16, 9)
        writer = animation.FFMpegWriter(bitrate=-1, codec="libx264", extra_args=['-pix_fmt', 'yuv420p'])
        ani.save("results/order_parameters.mp4", writer=writer)
    plt.close()
else :
    plt.gca().set_title('Evolution of the Kuramoto order parameters', fontsize=25)
    plt.show() 



#Spikes evolution
colorsCodes = [newred if inhibitory[i]==1 else newblue for i in range(len(inhibitory))]
fig, ax = plt.subplots()
ax.eventplot(spikes, colors=colorsCodes, lineoffsets=1, linelengths=1.0, linewidths=2)
#if adimensional :
#    plt.gca().set_xlabel('Time', fontsize=20)
#else :
#    plt.gca().set_xlabel('Time (s)', fontsize=20)
#plt.gca().set_ylabel('Neurons', fontsize=20)
#plt.xticks(fontsize=20)
#plt.yticks(fontsize=20)

if animated:
    def animateSpikesEvolution(frame):
        ax.set_xlim(frame*0.02, frame*0.02+0.5)   #0.02s step, 0.5s window
    speedFactor = 10
    ani = animation.FuncAnimation(fig, animateSpikesEvolution, interval=10*speedFactor, blit=False, frames=50*round(dt*iterations), cache_frame_data=False, repeat=False) #50 frames per simulation second

if save : 
    
    plt.gca().set_xlim(0.0, 50.0)
    plt.gcf().set_size_inches(25.0, 2.0)
    #plt.gca().set_xlim(10.0, 50.0)
    #plt.gcf().set_size_inches(9.0, 2.0)
    plt.savefig('results/spikes_evolution_simu.pdf', dpi=300, bbox_inches='tight')
    plt.gca().autoscale()
    if animated:
        plt.gca().set_title('Spike trains', fontsize=25)
        plt.gcf().set_size_inches(16, 9)
        writer = animation.FFMpegWriter(bitrate=-1, codec="libx264", extra_args=['-pix_fmt', 'yuv420p'])
        ani.save("results/spikes_evolution.mp4", writer=writer)
    plt.close()
else : 
    plt.gca().set_title('Spikes of neurons through the time', fontsize=25)
    plt.show()




plt.figure(figsize=(6.0,1.0))
plt.axis('off')
#plt.scatter(spikes_exc_ha[0], spikes_exc_ha[1], s=1, lw=0, color=newred)
#plt.scatter(spikes_inh_ha[0], spikes_inh_ha[1], s=1, lw=0, color=newblue)
#plt.scatter(spikesMatrix[i], spikesMatrix[i], s=1, lw=0, color=colorsCodes)
plt.eventplot(spikes, colors=colorsCodes, lineoffsets=1, linelengths=1.0, linewidths=2)
# plt.xlabel( 'Time (seconds)', fontsize=10)
# plt.ylabel( 'Neuron index', fontsize=10)
# plt.tick_params(labelsize=8)
plt.xlim(0,50)
#plt.xlim(25,55)
plt.tight_layout()

if save:
    plt.savefig('results/spikes_evolution_simu_paper.pdf', bbox_inches='tight', dpi=300 )



    


#Firing rates  

#Evolution mean firing rates of the network
meanFrequenciesNetwork = np.sum(allMeanFrequencies, axis=0)/len(allMeanFrequencies)

allMeanFrequencies1 = allMeanFrequencies[0:40, :]
allMeanFrequencies2 = allMeanFrequencies[40:80, :]
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
        ani.save("results/mean_firing_rate_evolution_network.mp4", writer=writer)
    plt.close()
else :
    plt.gca().set_title('Mean firing rates of the network', fontsize=25)
    plt.show() 
 
   
    
    
#Calculate PDF firing rate
cst = 0.0000000 #to avoid null PDF (for log in free energy)
counts, bins = np.histogram(meanFrequenciesNetwork, bins=30)
counts = counts+cst
pdf_Network = counts / sum(counts)

counts, bins = np.histogram(meanFrequenciesNetwork1, bins=30)
counts = counts+cst
pdf_cluster1= counts / sum(counts)

counts, bins = np.histogram(meanFrequenciesNetwork2, bins=30)
counts = counts+cst
pdf_cluster2 = counts / sum(counts)

std_cluster1 = np.std(meanFrequenciesNetwork1)
std_cluster2 = np.std(meanFrequenciesNetwork2)
std_Network = np.std(meanFrequenciesNetwork)

mean_cluster1 = np.mean(meanFrequenciesNetwork1)
mean_cluster2 = np.mean(meanFrequenciesNetwork2)
mean_Network = np.mean(meanFrequenciesNetwork)


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
cst = 0.0000000 #to avoid null PDF (for log in free energy)
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
cst = 0.0000000 #to avoid null PDF (for log in free energy)
counts, bins = np.histogram(orderParameter1, bins=100)
counts = counts+cst
pdf_Network = counts / sum(counts)

counts, bins = np.histogram(orderParameter1_1, bins=100)
counts = counts+cst
pdf_cluster1= counts / sum(counts)

counts, bins = np.histogram(orderParameter1_2, bins=100)
counts = counts+cst
pdf_cluster2 = counts / sum(counts)

std_cluster1 = np.std(orderParameter1_1)
std_cluster2 = np.std(orderParameter1_2)
std_Network = np.std(orderParameter1)

mean_cluster1 = np.mean(orderParameter1_1)
mean_cluster2 = np.mean(orderParameter1_2)
mean_Network = np.mean(orderParameter1)


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
#plt.gca().set_xlabel('Time (s)') #, fontsize=20

axs[0].eventplot(spikes, colors=colorsCodes, lineoffsets=1, linelengths=1.0, linewidths=2)
#axs[0].set_ylabel('Neurons')  #, fontsize=20
axs[0].set_yticks((0,40,80,100))
axs[0].set_yticklabels((0,40,80,100), fontsize=4)


axs[1].plot(tpoints, orderParameter1_1, label='Cluster 1', color='green') 
axs[1].plot(tpoints, orderParameter1_2, label='Cluster 2', color='orange')
axs[1].plot(tpoints, orderParameter1, label='Network', color='grey') 
axs[1].set_yticks((0,0.5,1))
axs[1].set_yticklabels((0,0.5,1), fontsize=4)
#axs[1].set_ylabel('Order \nparameter')       #, fontsize=20
#axs[1].legend(fontsize=20)

axs[2].plot(tpoints, meanFrequenciesNetwork1, color='green') #, label='Cluster 1'
axs[2].plot(tpoints, meanFrequenciesNetwork2, color='orange')  #, label='Cluster 2'
#axs[2].plot(tpoints, meanFrequenciesNetwork, color='grey') 
axs[2].set_yticks((0,10,20))
axs[2].set_yticklabels((0,10,20), fontsize=4)
#axs[2].set_ylabel('Mean firing \nrate (Hz)') #, fontsize=20
#axs[2].legend(fontsize=20)

axs[3].plot(tpoints[0:len(tpoints)-1], changeRates[:,1], color='green')  #, label='Cluster 1'
axs[3].plot(tpoints[0:len(tpoints)-1], changeRates[:,2], color='orange')  #, label='Cluster 2'
#axs[3].plot(tpoints[0:len(tpoints)-1], changeRates[:,0], color='grey')
axs[3].set_yticks((0.00,0.01,0.02))
axs[3].set_yticklabels((0.00,0.01,0.02), fontsize=4)
#axs[3].legend(fontsize=20)
#axs[3].set_ylabel('Mean change \nrate of weights') #, fontsize=20

#axs[4].plot(tpoints[0:len(phases_network)], np.sum(phases_network[:, 0:40], axis=1)/40.0)  #, label='Cluster 1'
#axs[4].plot(tpoints[0:len(phases_network)], np.sum(phases_network[:, 40:80], axis=1)/40.0)  #, label='Cluster 2'
#axs[4].set_ylabel('Phase') #, fontsize=20

fig.legend(bbox_to_anchor=(0.5, -0.09), loc='lower center', fontsize=4, ncol=3)

plt.xticks(fontsize=4)
#plt.yticks(fontsize=8)

if animated:
    def animateMix(frame):
        for ax in axs.flat:
            axs[0].set_xlim(frame*0.02, frame*0.02+0.5)   #0.02s step, 0.5s window
    speedFactor = 10
    ani = animation.FuncAnimation(fig, animateMix, interval=10*speedFactor, blit=False, frames=50*round(dt*len(tpoints)), cache_frame_data=False, repeat=False) #50 frames per simulation second
    
if save : 
    #plt.gca().set_xlim(1830, 1860)
    #axs[3].set_xlim(3320, 3360)
    #axs[3].set_xlim(2740, 2780)
    #axs[3].set_xlim(2700, 2735)
    axs[3].set_xlim(2670, 2700)
    axs[3].set_ylim(top=0.025)
    fig.tight_layout()
    fig.set_size_inches(3.28, 3.88) 
    #plt.savefig('results/mix.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/mix.pdf', dpi=300, bbox_inches='tight')
    plt.gca().autoscale()
    
   
    if animated:
        plt.gcf().set_size_inches(16, 9)
        writer = animation.FFMpegWriter(bitrate=-1, codec="libx264", extra_args=['-pix_fmt', 'yuv420p'])
        ani.save("results/mix.mp4", writer=writer)
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


