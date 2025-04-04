// gcc -W -Wall -o simulationQIF simulationQIF.c
// ./simulationQIF

#include "QIF.h"


/*
* Calculate the rate of change of weights
*
* @param	w		coupling weights matrix at time 1
* @param	w2		coupling weights matrix at time 2
* @param	k		index start
* @param	n		index total
* @param	ki		index postsynaptic neuron i start
* @param	ni		index postsynaptic neuron i stop (not included)
* @param	kj		index presynaptic neuron j start
* @param	nj		index presynaptic neuron j stop (not included)
* @param	type	calculation type of the rate: 0 = absolute variation rate, 1 = exact variation rate, 2 = exact variation rate reverse (for inhbitory)
*/
long double ChangeRate(long double **w, long double ** w2, int k, int n, int ki, int ni, int kj, int nj, int type)
{
    int i, j, normalization = 0;
    long double K = 0.0;

	//for the selected neurons of the network
	
	for (i=k; i<n; i++) 
	{
		for (j=kj; j<nj; j++)
		{
			if(type==0) //all absolute difference time 2 and time 1
			{
				if(i>=ki && i<ni)
				{
					if(i!=j)
					{
						K += fabsl(w2[i][j]-w[i][j]);
						normalization++;
					}
				}
			}
			else if(type==1)	//all difference time 2 and time 1
			{
				if(i>=ki && i<ni)
				{
					if(i!=j)
					{
						K += w2[i][j]-w[i][j];
						normalization++;
					}
				}
			}
			else if(type==2)	//all difference time 2 and time 1 reverse
			{
				if(i>=ki && i<ni)
				{
					if(i!=j)
					{
						K += -(w2[i][j]-w[i][j]);
						normalization++;
					}
				}
			}
			else if(type==3) 	//all intra weights time 2
			{
				if(i>=ki && i<ni)
				{
					if(i!=j)
					{
						K += w2[i][j];
						normalization++;
					}
				}
			}
			else if(type==-3) 	//all inter weights time 2
			{
				if(i>=ni || i<ki)
				{
					if(i!=j)
					{
						K += w2[i][j];
						normalization++;
					}
				}
			}
		
			else if(type==4) 	//all intra anti-Hebbian weights time 2
			{
				if(i>=ki && i<ni && ((j%2)==0))
				{
					if(i!=j)
					{
						K += w2[i][j];
						normalization++;
					}
				}
			}
			
			else if(type==-4) 	//all inter anti-Hebbian weights time 2
			{
				if((i>=ni || i<ki) && ((j%2)==0))
				{
					if(i!=j)
					{
						K += w2[i][j];
						normalization++;
					}
				}
			}
			
			else if(type==5) 	//all intra Hebbian weights time 2
			{
				if(i>=ki && i<ni && ((j%2)==1))
				{
					if(i!=j)
					{
						K += w2[i][j];
						normalization++;
					}
				}
			}
			
			else if(type==-5) 	//all inter Hebbian weights time 2
			{
				if((i>=ni || i<ki) && ((j%2)==1))
				{
					if(i!=j)
					{
						K += w2[i][j];
						normalization++;
					}
				}
			}
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
		
	FILE *fptr;
	FILE *fptrMatrix;
	FILE *fptrRate;
	FILE *fptrSpike;
	
	long double **w1;
	long double intraEE;
	long double intraHI;
	long double intraAI;
	long double intraEI;
	long double intraHE;
	long double intraAE;
	long double interEE;
	long double interHI;
	long double interAI;
	long double interEI;
	long double interHE;
	long double interAE;
	
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
	
	int load;							//If we load a save of a network
	int save = 1;						//If we save data of the simulation
	float timeStepSave = 0.1;			//Time step to save data (in second)
	double saveData = timeStepSave;		//Variable next time to save data
	int nbExcitatoryNeurons;			//The number of excitatory neurons
	int nbInhibitoryNeurons;			//The number of inhibitory neurons
	
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
		load = 0;
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
		load = atoi(argv[18]);
	}
	
	
	struct neurons neurons;
	if(load)
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
	nbExcitatoryNeurons = neurons.n/100.0*ratio;
	nbInhibitoryNeurons = neurons.n - nbExcitatoryNeurons;
		
	
	/**** Save spikes of neurons ****/
	fptrSpike = fopen("spikes.txt","w");
	
	if(fptrSpike == NULL)
	{
		printf("Error!\n");   
		exit(1);             
	}
	
	
	/**** Save change rate of weights through the time ****/
	fptrRate = fopen("changeRates.txt","w");
	
	if(fptrRate == NULL)
	{
		printf("Error!");   
		exit(1);             
	}
	//fprintf(fptrRate,"%f ", 0.0); //register change rate of weights of the whole network at time 0 (=0)
	
	//fprintf(fptrRate,"%f ", 0.0); //register change rate of weights of the cluster 1 at time 0 (=0)
	
	//fprintf(fptrRate,"%f ", 0.0); //register change rate of weights of the cluster 2 at time 0 (=0)
	
	//fprintf(fptrRate, "\n");
	
	
	
	/**** Save weights matrix ****/
	fptrMatrix = fopen("weights_matrices.txt" ,"w");
	
	if(fptrMatrix == NULL)
	{
		printf("Error!\n");   
		exit(1);             
	}
	
	saveMatrix(fptrMatrix, neurons.w, neurons.n, neurons.n);
	w1 = copyMatrix(neurons.w, neurons.n);	//save first weights matrix
	
	
	//Parameters duration (numerical values in second)
	//Convert duration in seconds to iterations
	int periodInput = 1.0/dt;
	int periodStimulation = 0.8/dt;
	int durationSpontaneousActivity = 5/dt;
	int durationLearning = 40.0/dt;	//0 40	1800 3800
	int nbInput = 2;		//2, 4, 8, 10, 16, 20, 33
	float inputFrequency = 50.0; //in Hz
	
	
	//Simulate for nbIterations iterations
	for(t=0; t < nbIterations; t++)
	{	
		
		if(((t%periodInput)==0) && (t<durationLearning) && (t>=durationSpontaneousActivity))
		{	
			//Experiment non-overlapting nbInput clusters
			
			//Select a random cluster amount nbInput
			j = rand()%nbInput;
			
			//Stimulation excitatory neurons
			addBinaryLocalized(neurons.inputs, n, (j*nbExcitatoryNeurons/(float)nbInput)+(nbExcitatoryNeurons/(float)nbInput)/2, nbExcitatoryNeurons/nbInput, neurons.tau_m, inputFrequency);		
			//addBinaryLocalizedRandom(neurons.inputs, n, (j*nbExcitatoryNeurons/(float)nbInput)+(nbExcitatoryNeurons/(float)nbInput)/2, nbExcitatoryNeurons/nbInput, neurons.tau_m);	
				
			
			//Stimulation inhibitory neurons
			for(i=0;i<(nbInhibitoryNeurons/nbInput);i++)
			{
				neurons.inputs[j*(nbInhibitoryNeurons/nbInput)+nbExcitatoryNeurons+i] = pow(inputFrequency*M_PI*tau_m, 2.0);
			}	
				
			//Inhbitory neurons
			/*if(j<(int)(nbExcitatoryNeurons/2)) //40
			{
				for(i=nbExcitatoryNeurons; i < neurons.n - (neurons.n-nbExcitatoryNeurons)/2 ; i++)
			    {
					neurons.inputs[i] = pow(inputFrequency*M_PI*tau_m, 2.0);
				}	
			}
			else
			{		
				for(i=nbExcitatoryNeurons + (neurons.n-nbExcitatoryNeurons)/2; i < neurons.n; i++)
			    {
					neurons.inputs[i] = pow(inputFrequency*M_PI*tau_m, 2.0);
				}
			}*/	
				
				
				
				
			//Overlaps of 10 neurons between successive clusters
			/*if(j==0)
			{
				j = (j*nbExcitatoryNeurons/10.0)+(nbExcitatoryNeurons/10.0)/2;
				addBinaryLocalized(neurons.inputs, n, j, nbExcitatoryNeurons/10.0, neurons.tau_m, inputFrequency);
				
				neurons.inputs[nbExcitatoryNeurons-1] = pow(inputFrequency*M_PI*tau_m, 2.0);
				neurons.inputs[nbExcitatoryNeurons-2] = pow(inputFrequency*M_PI*tau_m, 2.0);
				neurons.inputs[nbExcitatoryNeurons-3] = pow(inputFrequency*M_PI*tau_m, 2.0);
				neurons.inputs[nbExcitatoryNeurons-4] = pow(inputFrequency*M_PI*tau_m, 2.0);
				neurons.inputs[nbExcitatoryNeurons-5] = pow(inputFrequency*M_PI*tau_m, 2.0);
				
				neurons.inputs[nbExcitatoryNeurons/10.0] = pow(inputFrequency*M_PI*tau_m, 2.0);
				neurons.inputs[nbExcitatoryNeurons/10.0+1] = pow(inputFrequency*M_PI*tau_m, 2.0);
				neurons.inputs[nbExcitatoryNeurons/10.0+2] = pow(inputFrequency*M_PI*tau_m, 2.0);
				neurons.inputs[nbExcitatoryNeurons/10.0+3] = pow(inputFrequency*M_PI*tau_m, 2.0);
				neurons.inputs[nbExcitatoryNeurons/10.0+4] = pow(inputFrequency*M_PI*tau_m, 2.0);
			}
			else if(j==9)
			{
				j = (j*nbExcitatoryNeurons/10.0)+(nbExcitatoryNeurons/10.0)/2;
				addBinaryLocalized(neurons.inputs, n, j, nbExcitatoryNeurons/10.0, neurons.tau_m, inputFrequency);
				
				neurons.inputs[nbExcitatoryNeurons-nbExcitatoryNeurons/10.0-1] = pow(inputFrequency*M_PI*tau_m, 2.0);
				neurons.inputs[nbExcitatoryNeurons-nbExcitatoryNeurons/10.0-2] = pow(inputFrequency*M_PI*tau_m, 2.0);
				neurons.inputs[nbExcitatoryNeurons-nbExcitatoryNeurons/10.0-3] = pow(inputFrequency*M_PI*tau_m, 2.0);
				neurons.inputs[nbExcitatoryNeurons-nbExcitatoryNeurons/10.0-4] = pow(inputFrequency*M_PI*tau_m, 2.0);
				neurons.inputs[nbExcitatoryNeurons-nbExcitatoryNeurons/10.0-5] = pow(inputFrequency*M_PI*tau_m, 2.0);
				
				neurons.inputs[0] = pow(inputFrequency*M_PI*tau_m, 2.0);
				neurons.inputs[1] = pow(inputFrequency*M_PI*tau_m, 2.0);
				neurons.inputs[2] = pow(inputFrequency*M_PI*tau_m, 2.0);
				neurons.inputs[3] = pow(inputFrequency*M_PI*tau_m, 2.0);
				neurons.inputs[4] = pow(inputFrequency*M_PI*tau_m, 2.0);
			}
			else
			{
				j = (j*nbExcitatoryNeurons/10.0)+(nbExcitatoryNeurons/10.0)/2;
				addBinaryLocalized(neurons.inputs, n, j, nbExcitatoryNeurons/10.0, neurons.tau_m, inputFrequency);
				
				neurons.inputs[j*nbExcitatoryNeurons/10.0-1] = pow(inputFrequency*M_PI*tau_m, 2.0);
				neurons.inputs[j*nbExcitatoryNeurons/10.0-2] = pow(inputFrequency*M_PI*tau_m, 2.0);
				neurons.inputs[j*nbExcitatoryNeurons/10.0-3] = pow(inputFrequency*M_PI*tau_m, 2.0);
				neurons.inputs[j*nbExcitatoryNeurons/10.0-4] = pow(inputFrequency*M_PI*tau_m, 2.0);
				neurons.inputs[j*nbExcitatoryNeurons/10.0-5] = pow(inputFrequency*M_PI*tau_m, 2.0);
				
				neurons.inputs[(j+1)*nbExcitatoryNeurons/10.0] = pow(inputFrequency*M_PI*tau_m, 2.0);
				neurons.inputs[(j+1)*nbExcitatoryNeurons/10.0+1] = pow(inputFrequency*M_PI*tau_m, 2.0);
				neurons.inputs[(j+1)*nbExcitatoryNeurons/10.0+2] = pow(inputFrequency*M_PI*tau_m, 2.0);
				neurons.inputs[(j+1)*nbExcitatoryNeurons/10.0+3] = pow(inputFrequency*M_PI*tau_m, 2.0);
				neurons.inputs[(j+1)*nbExcitatoryNeurons/10.0+4] = pow(inputFrequency*M_PI*tau_m, 2.0);
			}*/
					
			
			//Exp 2 clusters with overlaps
			/*
			j = (j*(nbExcitatoryNeurons/2.0-4))+((nbExcitatoryNeurons+8)/2.0)/2;
			addBinaryLocalized(neurons.inputs, n, j, (nbExcitatoryNeurons+8)/2.0, neurons.type_neuron);
			*/
		}
		else if(((t-periodStimulation)%periodInput)==0 && (t<durationLearning))
		{
			addNullInputs(neurons.inputs, n);
		}
		
		else if(t==durationLearning)
		{
			addNullInputs(neurons.inputs, n);
			//reset_excitatory_weights_null(&neurons);
		}
		
		
		
		/*if(t==(int)(25.0/dt))
		{
			neurons.inputs[0] = pow(inputFrequency*M_PI*tau_m, 2.0);
			neurons.inputs[1] = pow(inputFrequency*M_PI*tau_m, 2.0);
			neurons.inputs[2] = pow(inputFrequency*M_PI*tau_m, 2.0);
			neurons.inputs[3] = pow(inputFrequency*M_PI*tau_m, 2.0);
			neurons.inputs[4] = pow(inputFrequency*M_PI*tau_m, 2.0);
			neurons.inputs[5] = pow(inputFrequency*M_PI*tau_m, 2.0);
			neurons.inputs[6] = pow(inputFrequency*M_PI*tau_m, 2.0);
			neurons.inputs[7] = pow(inputFrequency*M_PI*tau_m, 2.0);
			neurons.inputs[8] = pow(inputFrequency*M_PI*tau_m, 2.0);
		}
		
		if(t==(int)(26.5/dt))
		{
			addNullInputs(neurons.inputs, n);
		}*/
		
		
		
		update_states(&neurons, spikes, t, fptrSpike, save);
		update_weights(&neurons, spikes, nbInput); 
		
		
		/***** Save data *****/
		if(((t*dt >= saveData) || ((t+1)==nbIterations)) && save)
		{
			saveData += timeStepSave;	//Save every timeStepSave second
		
			/*fprintf(fptrRate,"%10.15Lf ", ChangeRate(w1, neurons.w, 0, neurons.n, 0, neurons.n, 0, neurons.n, 0)); //calculate and register change rate of weights of the whole network
			
			fprintf(fptrRate,"%10.15Lf ", (ChangeRate(w1, neurons.w, 0, neurons.n, 0, 40, 0, 40, 1) + ChangeRate(w1, neurons.w, 80, 90, 0, 40, 1))/2.0); //calculate and register change rate of weights of the cluster 1
			
			fprintf(fptrRate,"%10.15Lf ", (ChangeRate(w1, neurons.w, 0, neurons.n, 40, 80, 40, 80, 1) + ChangeRate(w1, neurons.w, 90, 100, 40, 80, 1))/2.0); //calculate and register change rate of weights of the cluster 2
			*/
			
			//Mean intra and inter connections
			intraEE = 0.0;
			interEE = 0.0;
			
			intraEI = 0.0;
			interEI = 0.0;
			
			intraAI = 0.0;
			interAI = 0.0;
			
			intraHI = 0.0;
			interHI = 0.0;
			
			intraAE = 0.0;
			interAE = 0.0;
			
			intraHE = 0.0;
			interHE = 0.0;
			
			for(i=0; i < nbInput ; i++)
			{		
				intraEE += ChangeRate(w1, neurons.w, 0, nbExcitatoryNeurons, i*nbExcitatoryNeurons/(float)nbInput, nbExcitatoryNeurons/(float)nbInput + i*nbExcitatoryNeurons/(float)nbInput, i*nbExcitatoryNeurons/(float)nbInput, nbExcitatoryNeurons/(float)nbInput + i*nbExcitatoryNeurons/(float)nbInput, 3);
				interEE += ChangeRate(w1, neurons.w, 0, nbExcitatoryNeurons, i*nbExcitatoryNeurons/(float)nbInput, nbExcitatoryNeurons/(float)nbInput + i*nbExcitatoryNeurons/(float)nbInput, i*nbExcitatoryNeurons/(float)nbInput, nbExcitatoryNeurons/(float)nbInput + i*nbExcitatoryNeurons/(float)nbInput, -3);
			
				intraEI += ChangeRate(w1, neurons.w, nbExcitatoryNeurons, neurons.n, i*(nbInhibitoryNeurons/nbInput)+nbExcitatoryNeurons, i*(nbInhibitoryNeurons/nbInput)+nbExcitatoryNeurons+(nbInhibitoryNeurons/nbInput), i*nbExcitatoryNeurons/(float)nbInput, nbExcitatoryNeurons/(float)nbInput + i*nbExcitatoryNeurons/(float)nbInput, 3);
				interEI += ChangeRate(w1, neurons.w, nbExcitatoryNeurons, neurons.n, i*(nbInhibitoryNeurons/nbInput)+nbExcitatoryNeurons, i*(nbInhibitoryNeurons/nbInput)+nbExcitatoryNeurons+(nbInhibitoryNeurons/nbInput), i*nbExcitatoryNeurons/(float)nbInput, nbExcitatoryNeurons/(float)nbInput + i*nbExcitatoryNeurons/(float)nbInput, -3);
				
				intraAI += ChangeRate(w1, neurons.w, nbExcitatoryNeurons, neurons.n, i*(nbInhibitoryNeurons/nbInput)+nbExcitatoryNeurons, i*(nbInhibitoryNeurons/nbInput)+nbExcitatoryNeurons+(nbInhibitoryNeurons/nbInput), i*(nbInhibitoryNeurons/nbInput)+nbExcitatoryNeurons, i*(nbInhibitoryNeurons/nbInput)+nbExcitatoryNeurons+(nbInhibitoryNeurons/nbInput), 4);
				interAI += ChangeRate(w1, neurons.w, nbExcitatoryNeurons, neurons.n, i*(nbInhibitoryNeurons/nbInput)+nbExcitatoryNeurons, i*(nbInhibitoryNeurons/nbInput)+nbExcitatoryNeurons+(nbInhibitoryNeurons/nbInput), i*(nbInhibitoryNeurons/nbInput)+nbExcitatoryNeurons, i*(nbInhibitoryNeurons/nbInput)+nbExcitatoryNeurons+(nbInhibitoryNeurons/nbInput), -4);
				
				intraHI += ChangeRate(w1, neurons.w, nbExcitatoryNeurons, neurons.n, i*(nbInhibitoryNeurons/nbInput)+nbExcitatoryNeurons, i*(nbInhibitoryNeurons/nbInput)+nbExcitatoryNeurons+(nbInhibitoryNeurons/nbInput), i*(nbInhibitoryNeurons/nbInput)+nbExcitatoryNeurons, i*(nbInhibitoryNeurons/nbInput)+nbExcitatoryNeurons+(nbInhibitoryNeurons/nbInput), 5);
				interHI += ChangeRate(w1, neurons.w, nbExcitatoryNeurons, neurons.n, i*(nbInhibitoryNeurons/nbInput)+nbExcitatoryNeurons, i*(nbInhibitoryNeurons/nbInput)+nbExcitatoryNeurons+(nbInhibitoryNeurons/nbInput), i*(nbInhibitoryNeurons/nbInput)+nbExcitatoryNeurons, i*(nbInhibitoryNeurons/nbInput)+nbExcitatoryNeurons+(nbInhibitoryNeurons/nbInput), -5);
				
				intraAE += ChangeRate(w1, neurons.w, 0, nbExcitatoryNeurons, i*nbExcitatoryNeurons/(float)nbInput, nbExcitatoryNeurons/(float)nbInput + i*nbExcitatoryNeurons/(float)nbInput, i*(nbInhibitoryNeurons/nbInput)+nbExcitatoryNeurons, i*(nbInhibitoryNeurons/nbInput)+nbExcitatoryNeurons+(nbInhibitoryNeurons/nbInput), 4);
				interAE += ChangeRate(w1, neurons.w, 0, nbExcitatoryNeurons, i*nbExcitatoryNeurons/(float)nbInput, nbExcitatoryNeurons/(float)nbInput + i*nbExcitatoryNeurons/(float)nbInput, i*(nbInhibitoryNeurons/nbInput)+nbExcitatoryNeurons, i*(nbInhibitoryNeurons/nbInput)+nbExcitatoryNeurons+(nbInhibitoryNeurons/nbInput), -4);
				
				intraHE += ChangeRate(w1, neurons.w, 0, nbExcitatoryNeurons, i*nbExcitatoryNeurons/(float)nbInput, nbExcitatoryNeurons/(float)nbInput + i*nbExcitatoryNeurons/(float)nbInput, i*(nbInhibitoryNeurons/nbInput)+nbExcitatoryNeurons, i*(nbInhibitoryNeurons/nbInput)+nbExcitatoryNeurons+(nbInhibitoryNeurons/nbInput), 5);
				interHE += ChangeRate(w1, neurons.w, 0, nbExcitatoryNeurons, i*nbExcitatoryNeurons/(float)nbInput, nbExcitatoryNeurons/(float)nbInput + i*nbExcitatoryNeurons/(float)nbInput, i*(nbInhibitoryNeurons/nbInput)+nbExcitatoryNeurons, i*(nbInhibitoryNeurons/nbInput)+nbExcitatoryNeurons+(nbInhibitoryNeurons/nbInput), -5);
			}	
			
			fprintf(fptrRate,"%10.15Lf ", intraEE/(float)nbInput); //register mean weights of intra E-E weights
			fprintf(fptrRate,"%10.15Lf ", interEE/(float)nbInput); //register mean weights of inter E-E weights
			
			fprintf(fptrRate,"%10.15Lf ", intraEI/(float)nbInput); //register mean weights of intra E-I weights
			fprintf(fptrRate,"%10.15Lf ", interEI/(float)nbInput); //register mean weights of inter E-I weights
			
			fprintf(fptrRate,"%10.15Lf ", intraAE/(float)nbInput); //register mean weights of intra A-E weights 
			fprintf(fptrRate,"%10.15Lf ", interAE/(float)nbInput); //register mean weights of inter A-E weights
			
			fprintf(fptrRate,"%10.15Lf ", intraAI/(float)nbInput); //register mean weights of intra A-I weights
			fprintf(fptrRate,"%10.15Lf ", interAI/(float)nbInput); //register mean weights of inter A-I weights
			
			fprintf(fptrRate,"%10.15Lf ", intraHE/(float)nbInput); //register mean weights of intra H-E  weights
			fprintf(fptrRate,"%10.15Lf ", interHE/(float)nbInput); //register mean weights of inter H-E  weights
			
			fprintf(fptrRate,"%10.15Lf ", intraHI/(float)nbInput); //register mean weights of intra H-I weights
			fprintf(fptrRate,"%10.15Lf ", interHI/(float)nbInput); //register mean weights of inter H-I weights
			
			
			fprintf(fptrRate, "\n");
			fflush(fptrRate);
			
			freeMatrix(w1, neurons.n);
			w1 = copyMatrix(neurons.w, neurons.n);	//copy last weights matrix
		}
		
		if( ((t+1)==(durationLearning/3)) || ((t+1)==durationLearning) || ((t+1)==nbIterations) || ((t+1)==(int)round((1800.0+2.0*3600.0)/dt)) || ((t+1)==(int)round((1800.0+4.0*3600.0)/dt)) )
		//if( ((t+1)==(int)round(3600.0/dt)) || ((t+1)==(int)round(7200.0/dt)) || ((t+1)==(int)round(14400.0/dt)) || ((t+1)==(int)round(21600.0/dt)) || ((t+1)==(int)round(43200.0/dt)) || ((t+1)==(int)round(64800.0/dt)) || ((t+1)==nbIterations) )
		{
			saveMatrix(fptrMatrix, neurons.w, neurons.n, neurons.n);
			fflush(fptrMatrix);
			printf("%d \n", t+1);
		}
	}
	
	fclose(fptrMatrix);
	fclose(fptrSpike);
	fclose(fptrRate);
	freeMatrix(w1, neurons.n);
		
	
	/**** Save adjacency matrix ****/
	fptr = fopen("adjacency.txt" ,"w");
	
	if(fptr == NULL)
	{
		printf("Error!");   
		exit(1);             
	}
	
	//for each element of the matrix
	for(i=0; i < neurons.n; i++)
    {
		for(j=0; j < neurons.n; j++)
	    {
			fprintf(fptr,"%d ", neurons.a[i][j]);
		}
		fprintf(fptr, "\n");
	}
	
	fclose(fptr);
	
	
	/**** Save bifurcation parameters ****/
	fptr = fopen("alpha.txt","w");
	
	if(fptr == NULL)
	{
		printf("Error!");   
		exit(1);             
	}
	
	//for each neurons of the network
	for(i=0; i < neurons.n; i++)
    {
		fprintf(fptr,"%f ", neurons.eta[i]);
	}
	
	fclose(fptr);
	
	
	/**** Save type of neurons ****/
	fptr = fopen("inhibitory.txt","w");
	
	if(fptr == NULL)
	{
		printf("Error!");   
		exit(1);             
	}
	
	//for each neurons of the network
	for(i=0; i < neurons.n; i++)
    {
		if(neurons.type_neuron[i]!=0)
		{
			fprintf(fptr,"-1 ");
		}
		else
		{
			fprintf(fptr,"1 ");
		}
	}
	
	fclose(fptr);
	
	
	/**** Free memory ****/
	freeNeurons(&neurons);
    
	
    return 0;
}