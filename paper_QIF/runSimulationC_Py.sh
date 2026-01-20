#!/bin/sh

#Compile and execute C program and run visualization with python program
#Developped by Raphael BERGOIN
#Run :  ./runSimulationC_Py.sh

echo "Compilation..."
gcc -W -Wall -o simulationQIF simulationQIF.c -lm




echo "Running..."

# Choose one of these executions
# Parameters : duration of simulation in s | number of neurons | peak value | reset value | global coupling | learning rate | integration time step | membrane time constant | time decay excitatory | time decay inhibitory | inhibitory policy | adjacency policy | weight policy | bifurcation parameter policy | membrane potential policy | ratio of excitatory neurons | saved network


./simulationQIF 50 100 10.0 -10.0 100.0 5.0 0.001 0.02 0.002 0.005 o f g g r 80 0 	#exp 1 learning

#./simulationQIF 50 20000 10.0 -10.0 100.0 5.0 0.001 0.02 0.002 0.005 o f g g r 80 0 	#exp 1 learning large network


#./simulationQIF 1000 100 10.0 -10.0 100.0 5.0 0.001 0.02 0.002 0.005 o f g g r 33 0 	#exp 1 learning max stimuli



#./simulationQIF 18000 2000 10.0 -10.0 100.0 5.0 0.001 0.02 0.002 0.005 o f g g r 80 0 	#exp 1 learning nb stimuli with 1000 neurons with 5hours (18000sec rest)



#./simulationQIF 4000 100 10.0 -10.0 100.0 5.0 0.001 0.02 0.002 0.005 o f p g r 80 0 	#exp 2 consolidation proto modules



#./simulationQIF 86400 100 10.0 -10.0 100.0 5.0 0.001 0.02 0.002 0.005 o f m g r 80 0 	#exp 2.2 long-term reconstruction minimum conditions (24 hours)



#./simulationQIF 86400 100 10.0 -10.0 100.0 5.0 0.001 0.02 0.002 0.005 o f d g r 80 0 	#exp 2.2 long-term maintenance learned modules (24 hours)

#./simulationQIF 172800 100 10.0 -10.0 100.0 5.0 0.001 0.02 0.002 0.005 o f d g r 80 0 	#exp 2.2 long-term maintenance learned modules (48 hours)



#./simulationQIF 86400 100 10.0 -10.0 100.0 5.0 0.001 0.02 0.002 0.005 o f m g r 80 0 	#exp 3 reconstruction minimum conditions




#./simulationQIF 4000 100 10.0 -10.0 100.0 5.0 0.001 0.02 0.002 0.005 o f f g r 92 0 	#exp sup 4 clusters stability




#./simulationQIF 50 100 10.0 -10.0 100.0 5.0 0.001 0.02 0.002 0.005 o rre g g r 80 0 	#exp 1 learning, random connectivity

#./simulationQIF 18000 200 10.0 -10.0 100.0 5.0 0.001 0.02 0.002 0.005 o rre g g r 80 0 	#exp 1 learning nb stimuli with 200 neurons with 5hours (18000sec rest), random connectivity





#./simulationQIF 50 100 10.0 -10.0 100.0 5.0 0.001 0.02 0.002 0.005 o f d g r 80 0 	#exp evoked recall





echo "Plotting..."
python3 plotQIF.py 1

