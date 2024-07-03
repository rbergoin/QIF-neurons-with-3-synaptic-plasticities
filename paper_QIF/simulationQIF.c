// gcc -W -Wall -o simulationQIF simulationQIF.c
// ./simulationQIF

#include "QIF.h"


/*
* Calculate the rate of change of weights
*
* @param	w		coupling weights matrix at time 1
* @param	w2		coupling weights matrix at time 2
* @param	ki		index postsynaptic neuron i start
* @param	ni		index postsynaptic neuron i stop (not included)
* @param	kj		index presynaptic neuron j start
* @param	nj		index presynaptic neuron j stop (not included)
* @param	type	calculation type of the rate: 0 = absolute variation rate, 1 = exact variation rate, 2 = exact variation rate reverse (for inhbitory)
*/
long double ChangeRate(long double **w, long double ** w2, int ki, int ni, int kj, int nj, int type)
{
    int i, j, normalization = 0;
    long double K = 0.0;

	//for the selected neurons of the network
	for (i=ki; i<ni; i++) 
	{
		for (j=kj; j<nj; j++)
		{
			if(type==0)
			{
				K += fabsl(w2[i][j]-w[i][j]);
			}
			else if(type==1)
			{
				K += w2[i][j]-w[i][j];
			}
			else if(type==2)
			{
				K += -(w2[i][j]-w[i][j]);
			}
			
			normalization++;
		}
	}
    
    return K/normalization;
}


/*
* Save a matrix in a file
*
* @param	fptr	file
* @param	m		matrix
* @param	l		number of line
* @param	c		number of column
*/
void saveMatrix(FILE *fptr, long double **m, int l, int c)
{
	int i, j;
	
	//for each element of the matrix
	for(i=0; i < l; i++)
    {
		for(j=0; j < c; j++)
	    {
			fprintf(fptr,"%3.4Lf ", m[i][j]);
		}
		fprintf(fptr, "\n");
	}
	fprintf(fptr, "\n");
}


int main(int argc, char *argv[])
{	
    srand(time(NULL)); // randomize seed
	
	int t, i, j, degree;
	FILE *fptr1;
	FILE *fptr2;
	long double **w1;
	int nbIterations;					//Number of interations step of the simulation
	
	int n;								//number of neurons
	float vp;							//peak value
	float vr; 							//reset value
	double g;							//(chemical) global coupling strength
	float epsilon1;						//learning rate for the slow adaptation
	float epsilon2;						//learning rate for the fast adaptation
	float dt;							//integration time step
	float tau_m;						//membrane time constant
	float tau_d_e;						//time decay excitatory/AMPA
	float tau_d_i;						//time decay inhibitory/GABA
	char inhibitoryPolicy;				//type of organization policy of inhibitory neurons;
	char* adjacencyPolicy;				//type of policy for the adjacency matrix initialization
	char weightPolicy;					//type of policy for the coupling weights matrix initialization
	char etaPolicy;						//type of policy for the bifurcation parameter array initialization
	char vPolicy;						//type of policy for the membrane potential array initialization
	float ratio;						//ratio of excitatory neurons
	
	int save;							//If we load a save
	float timeStepSave = 0.1;			//Time step to save data (in second)
	double saveData = timeStepSave;		//Variable next time to save data
	int nbInputNeurons;					//The number of input neurons
	
	if(argc < 19 || argc > 19) 
	{
		n = 100;
		vp = 10.0;
		vr = -10.0;
		g = 100.0;
		epsilon1 = 1.0;
		epsilon2 = 1.0;
		dt = 0.001;
		tau_m = 0.02;
		tau_d_e = 0.002;
		tau_d_i = 0.005;
		nbIterations = 50/dt;
		inhibitoryPolicy = 'o';
		adjacencyPolicy = "f";
		weightPolicy = 'r';
		etaPolicy = 'g';
		vPolicy = 'r';
		ratio = 80.0;			
		save = 0;
	}
	else
	{	
		n = atoi(argv[2]);
		vp = atof(argv[3]);
		vr = atof(argv[4]);
		g = atof(argv[5]);
		epsilon1 = atof(argv[6]);
		epsilon2 = atof(argv[7]);
		dt = atof(argv[8]);
		tau_m = atof(argv[9]);
		tau_d_e = atof(argv[10]);
		tau_d_i = atof(argv[11]);
		nbIterations = atoi(argv[1])/dt;
		inhibitoryPolicy = argv[12][0];
		adjacencyPolicy = argv[13];
		weightPolicy = argv[14][0];
		etaPolicy = argv[15][0];
		vPolicy = argv[16][0];
		ratio = atof(argv[17]);
		save = atoi(argv[18]);
	}
	
	
	struct neurons neurons;
	if(save)
	{
		neurons =  initneuronsSaved(n, vp, vr, g, epsilon1, epsilon2, dt, tau_m, tau_d_e, tau_d_i, ratio);	//Create a network of n neurons from a save		
	}
	else
	{
		neurons = initNeurons(n, vp, vr, g, epsilon1, epsilon2, dt, tau_m, tau_d_e, tau_d_i, inhibitoryPolicy, adjacencyPolicy, weightPolicy, etaPolicy, vPolicy, ratio); 		//Create a network of n neurons		
	}
	
	
	degree =  graphDegree(neurons.a, n);
	int *spikes;
	spikes =  (int *) calloc(n, sizeof(int));
	nbInputNeurons = neurons.n/100.0*ratio;
		
	
	/**** Save change rate of weights through the time ****/
	fptr2 = fopen("changeRates.txt","w");
	
	if(fptr2 == NULL)
	{
		printf("Error!");   
		exit(1);             
	}
	fprintf(fptr2,"%f ", 0.0); //register change rate of weights of the whole network at time 0 (=0)
	
	fprintf(fptr2,"%f ", 0.0); //register change rate of weights of the cluster 1 at time 0 (=0)
	
	fprintf(fptr2,"%f ", 0.0); //register change rate of weights of the cluster 2 at time 0 (=0)
	
	fprintf(fptr2, "\n");
	
	
	
	/**** Save spikes of neurons ****/
	fptr1 = fopen("spikes.txt","w");
	
	if(fptr1 == NULL)
	{
		printf("Error!");   
		exit(1);             
	}
	fclose(fptr1);
	
	
	/**** Save weights matrix ****/
	fptr1 = fopen("weights_matrices.txt" ,"w");
	
	if(fptr1 == NULL)
	{
		printf("Error!");   
		exit(1);             
	}
	
	saveMatrix(fptr1, neurons.w, neurons.n, neurons.n);
	w1 = copyMatrix(neurons.w, neurons.n);	//save first weights matrix
	
	
	//Parameters duration (numerical values in second)
	//Convert duration in seconds to iterations
	int periodInput = 1.0/dt;
	int periodStimulation = 0.8/dt;
	int durationSpontaneousActivity = 5/dt;
	int durationLearning = 40/dt;	//400
	float inputFrequency = 50.0; //in Hz
	
	
	//Simulate for nbIterations iterations
	for(t=0; t < nbIterations; t++)
	{	
		//Provoke external recall
		/*if((t>=(20/dt)) && (t<(20.01/dt)))
		{
			neurons.inputs[0] = pow(100.0*M_PI*tau_m, 2.0);
			neurons.inputs[1] = pow(100.0*M_PI*tau_m, 2.0);
			neurons.inputs[2] = pow(100.0*M_PI*tau_m, 2.0);
			neurons.inputs[3] = pow(100.0*M_PI*tau_m, 2.0);
			neurons.inputs[4] = pow(100.0*M_PI*tau_m, 2.0);
			neurons.inputs[5] = pow(100.0*M_PI*tau_m, 2.0);
			neurons.inputs[6] = pow(100.0*M_PI*tau_m, 2.0);
			neurons.inputs[7] = pow(100.0*M_PI*tau_m, 2.0);
		}
		else if (t>=(20.01/dt))
		{
			neurons.inputs[0] = 0.0;
			neurons.inputs[1] = 0.0;
			neurons.inputs[2] = 0.0;
			neurons.inputs[3] = 0.0;
			neurons.inputs[4] = 0.0;
			neurons.inputs[5] = 0.0;
			neurons.inputs[6] = 0.0;
			neurons.inputs[7] = 0.0;
		}*/
		
		
		if(((t%periodInput)==0) && (t<durationLearning) && (t>=durationSpontaneousActivity))
		{	
			//Exp 2 clusters
			j = rand()%2;
			j = (j*nbInputNeurons/2.0)+(nbInputNeurons/2.0)/2;
			addBinaryLocalized(neurons.inputs, n, j, nbInputNeurons/2.0, neurons.tau_m, inputFrequency);
			//addBinaryLocalizedRandom(neurons.inputs, n, j, nbInputNeurons/2.0, neurons.tau_m);
			
			if(j<40)
			{
				neurons.inputs[80] = pow(inputFrequency*M_PI*tau_m, 2.0);
				neurons.inputs[81] = pow(inputFrequency*M_PI*tau_m, 2.0);
				neurons.inputs[82] = pow(inputFrequency*M_PI*tau_m, 2.0);
				neurons.inputs[83] = pow(inputFrequency*M_PI*tau_m, 2.0);
				neurons.inputs[84] = pow(inputFrequency*M_PI*tau_m, 2.0);
				neurons.inputs[85] = pow(inputFrequency*M_PI*tau_m, 2.0);
				neurons.inputs[86] = pow(inputFrequency*M_PI*tau_m, 2.0);
				neurons.inputs[87] = pow(inputFrequency*M_PI*tau_m, 2.0);
				neurons.inputs[88] = pow(inputFrequency*M_PI*tau_m, 2.0);
				neurons.inputs[89] = pow(inputFrequency*M_PI*tau_m, 2.0);
				
			}
			else
			{		
				neurons.inputs[90] = pow(inputFrequency*M_PI*tau_m, 2.0);
				neurons.inputs[91] = pow(inputFrequency*M_PI*tau_m, 2.0);
				neurons.inputs[92] = pow(inputFrequency*M_PI*tau_m, 2.0);
				neurons.inputs[93] = pow(inputFrequency*M_PI*tau_m, 2.0);
				neurons.inputs[94] = pow(inputFrequency*M_PI*tau_m, 2.0);
				neurons.inputs[95] = pow(inputFrequency*M_PI*tau_m, 2.0);
				neurons.inputs[96] = pow(inputFrequency*M_PI*tau_m, 2.0);
				neurons.inputs[97] = pow(inputFrequency*M_PI*tau_m, 2.0);
				neurons.inputs[98] = pow(inputFrequency*M_PI*tau_m, 2.0);
				neurons.inputs[99] = pow(inputFrequency*M_PI*tau_m, 2.0);
				
			}
			
			//Exp 3 clusters
			/*j = rand()%3;
			//j = rand()%2;
			j = (j*nbInputNeurons/3.0)+(nbInputNeurons/3.0)/2;
			addBinaryLocalized(neurons.inputs, n, j, nbInputNeurons/3.0, neurons.type_neuron);*/
			
			
			//Exp 2 clusters with overlaps
			/*
			if(j>40)
			{
				j = 0;
			}
			else
			{
				j = 1;
			}
			j = (j*(nbInputNeurons/2.0-4))+((nbInputNeurons+8)/2.0)/2;
			addBinaryLocalized(neurons.inputs, n, j, (nbInputNeurons+8)/2.0, neurons.type_neuron);
			*/
		}
		else if(((t-periodStimulation)%periodInput)==0)
		{
			addNullInputs(neurons.inputs, n);
		}
		
		else if(t==durationLearning)
		{
			addNullInputs(neurons.inputs, n);
			//reset_excitatory_weights_null(&neurons);
		}
		
		
		
		update_states(&neurons, spikes, t);
		update_weights(&neurons, spikes); 
		
		
		/***** Save data *****/
		if((t*dt >= saveData) || ((t+1)==nbIterations))
		{
			saveData += timeStepSave;	//Save every timeStepSave second
		
			fprintf(fptr2,"%10.15Lf ", ChangeRate(w1, neurons.w, 0, neurons.n, 0, neurons.n, 0)); //calculate and register change rate of weights of the whole network
			
			fprintf(fptr2,"%10.15Lf ", (ChangeRate(w1, neurons.w, 0, 40, 0, 40, 1) + ChangeRate(w1, neurons.w, 80, 90, 0, 40, 1))/2.0); //calculate and register change rate of weights of the cluster 1
			
			fprintf(fptr2,"%10.15Lf ", (ChangeRate(w1, neurons.w, 40, 80, 40, 80, 1) + ChangeRate(w1, neurons.w, 90, 100, 40, 80, 1))/2.0); //calculate and register change rate of weights of the cluster 2
			
			fprintf(fptr2, "\n");

			
			freeMatrix(w1, neurons.n);
			w1 = copyMatrix(neurons.w, neurons.n);	//copy last weights matrix
		}
		
		if(((t+1)==(durationLearning/3)) || ((t+1)==durationLearning) || ((t+1)==nbIterations))
		{
			saveMatrix(fptr1, neurons.w, neurons.n, neurons.n);
		}
	}
	
	fclose(fptr1);
	fclose(fptr2);
	freeMatrix(w1, neurons.n);
		
	
	/**** Save adjacency matrix ****/
	fptr1 = fopen("adjacency.txt" ,"w");
	
	if(fptr1 == NULL)
	{
		printf("Error!");   
		exit(1);             
	}
	
	//for each element of the matrix
	for(i=0; i < neurons.n; i++)
    {
		for(j=0; j < neurons.n; j++)
	    {
			fprintf(fptr1,"%d ", neurons.a[i][j]);
		}
		fprintf(fptr1, "\n");
	}
	
	fclose(fptr1);
	
	
	/**** Save bifurcation parameters ****/
	fptr1 = fopen("alpha.txt","w");
	
	if(fptr1 == NULL)
	{
		printf("Error!");   
		exit(1);             
	}
	
	//for each neurons of the network
	for(i=0; i < neurons.n; i++)
    {
		fprintf(fptr1,"%f ", neurons.eta[i]);
	}
	
	fclose(fptr1);
	
	
	/**** Save type of neurons ****/
	fptr1 = fopen("inhibitory.txt","w");
	
	if(fptr1 == NULL)
	{
		printf("Error!");   
		exit(1);             
	}
	
	//for each neurons of the network
	for(i=0; i < neurons.n; i++)
    {
		if(neurons.type_neuron[i]!=0)
		{
			fprintf(fptr1,"-1 ");
		}
		else
		{
			fprintf(fptr1,"1 ");
		}
	}
	
	fclose(fptr1);
	
	
	/**** Free memory ****/
	freeNeurons(&neurons);
    
	
    return 0;
}