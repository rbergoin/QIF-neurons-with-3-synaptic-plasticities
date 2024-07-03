#!/bin/sh

#Compile and execute C program and run visualization with python program
#Developped by Raphael BERGOIN
#Run :  ./runSimulationC_Py.sh

save=1

echo "Compilation..."
gcc -W -Wall -o simulationQIF simulationQIF.c -lm




echo "Running..."

# Choose one of these executions
# Parameters : duration of simulation in s | number of neurons | peak value | reset value | global coupling | learning rate slow adaptation | learning rate fast adaptation | integration time step | membrane time constant | time decay excitatory | time decay inhibitory | inhibitory policy | adjacency policy | weight policy | bifurcation parameter policy | membrane potential policy | ratio of excitatory neurons | saved network


./simulationQIF 50 100 10.0 -10.0 100.0 5.0 5.0 0.001 0.02 0.002 0.005 o f g g r 80 0 	#exp 1 learning

#./simulationQIF 4000 100 10.0 -10.0 100.0 5.0 5.0 0.001 0.02 0.002 0.005 o f p g r 80 0 	#exp 2 consolidation proto modules

#./simulationQIF 4000 100 10.0 -10.0 100.0 5.0 5.0 0.001 0.02 0.002 0.005 o f m g r 80 0 	#exp 3 reconstruction minimum conditions

#./simulationQIF 50 100 10.0 -10.0 100.0 5.0 5.0 0.001 0.02 0.002 0.005 o f d g r 80 0 	#exp recall

#./simulationQIF 4000 100 10.0 -10.0 100.0 5.0 5.0 0.001 0.02 0.002 0.005 o f f g r 92 0 	#exp sup 4 clusters stability


echo "Plotting..."
python3 plotQIF.py save

