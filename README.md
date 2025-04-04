# QIF neurons with 3 synaptic plasticities

Source code of the paper: "Emergence and maintenance of modularity in neural networks with Hebbian and anti-Hebbian inhibitory STDP"

To run a simulation execute the script "runSimulationC_Py.sh". To save figures, change the value of the variable "save" in this script. The figures generated are put in the "results" folder.

The parameters of each simulation (i.e. duration, number of neurons...) can be changed by adapting the corresponding arguments in the execution of the C program. C code don't require any particular external library.

By default the experiment excuted is the "the learning of 2 memories". To execute another experiment, comment the lines of the current experiment and uncomment the lines between /* */ of the chosen experiment in the simulation C program. The ".h" files containing the networks class and the utils functions should not be modified.

During each simulations raw data (i.e. spikes time, weights matrices, parameters...) are generated in the corresponding ".txt" files, notably used to display the plots.
Data used to generate paper figures are present in Fig* folders.

The program "plotQIF.py" requires external libraries that can be easily installed via the "pip" command. This python3 program reads the raw data and plot them. No particular changes are required in this program except the duration of the simulation, the nunmber of neurons etc. Additionally, in some plots (such as spikes, or mean firing rate evolution), the x-axis is rescale for better visualization and can therefore be changed.

For any questions and additional requests, contact: raphael.bergoin@gmail.com
