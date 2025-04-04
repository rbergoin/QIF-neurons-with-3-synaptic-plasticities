#include "input.h"

/*
* Struct to represent a network of N QIF neurons 
*
*/
struct neurons 
{
    int n;								//number of neurons
	float ratio;						//ratio of excitatory neurons
	int *type_neuron;					//Type of neurons: 0 excitatory, 1 hebbian inhibitory, 2 anti-hebbian inhibitory
	
	long double *v;						//membrane potential of the neurons
	long double *s_e;					//synaptic excitatory inputs
	long double *s_h_i;					//synaptic hebbian inhibitory inputs
	long double *s_a_i;					//synaptic anti-hebbian inhibitory inputs
	
	long double *t_spikes;				//time of the last spike in s of each neuron
	long double *t_refractory;			//time of the last refractory in s of each neuron
	int *state_neuron;					//state of each neuron (-1=refractory period negative 0=accumulation period 1=refractory period positive)
	
	float *eta;							//bifurcation/excitability parameter of the neurons
	double *inputs;						//external input currents of the neurons
	
	float vp;							//peak value 
	float vr;							//reset value 
	
	double g; 							//(chemical) global coupling strength
	
	float epsilon1;						//learning rate for the slow adaptation
	float epsilon2;						//learning rate for the fast adaptation
	int **a;							//adjacency matrix
	long double **w;					//coupling weights matrix
	
	float dt;							//integration time step
	float tau_m;						//membrane time constant
	float tau_d_e;						//time decay excitatory/AMPA
	float tau_d_i;						//time decay inhibitory/GABA
};




/*
* Copy values of a table in a new one
*
* @param	table	table to copy
* @param	n		size of the table
*/
int *copyTable(int *table, int n) 
{ 
	int i;
	int *copy = (int *) malloc(sizeof(int) * n);
	
	for(i=0; i<n; i++)
	{
		copy[i] = table[i];
	}
	
	return copy;
}



/*
* Copy values of a table in a new one
*
* @param	table	table to copy
* @param	n		size of the table
*/
long double *copyTableL(long double *table, int n) 
{ 
	int i;
	long double *copy = (long double *) malloc(sizeof(long double) * n);
	
	for(i=0; i<n; i++)
	{
		copy[i] = table[i];
	}
	
	return copy;
}



/*
* Copy values of a matrix in a new one
*
* @param	matrix	matrix to copy
* @param	n		size of the matrix
*/
long double **copyMatrix(long double **matrix, int n) 
{ 
	int i, j;
	long double **copy = (long double **) malloc(sizeof(long double *) * n);

	for (i=0; i<n; i++) 
	{
		copy[i] = (long double *) malloc(sizeof(long double) * n);
		for (j=0; j<n; j++) 
		{
			copy[i][j] = matrix[i][j];
		}
	}
	
	return copy;
}



/*
* Copy values of a matrix in a new one
*
* @param	matrix	matrix to copy
* @param	n		size of the matrix
*/
void freeMatrix(long double **matrix, int n)  
{ 
	int i;
	
	//Free matrix
	for (i=0; i<n; i++) 
	{
		free(matrix[i]);
	}
	free(matrix);
}


/*
* Initialize a network of QIF neurons
*
* @param	n					number of neurons
* @param	vp					peak value 
* @param	vr					reset value 	
* @param	g 					(chemical) global coupling strength
* @param	epsilon1			learning rate for the slow adaptation
* @param	epsilon2			learning rate for the fast adaptation
* @param	dt 		 			integration time step
* @param	tau_m				membrane time constant
* @param	tau_d_e				time decay excitatory/AMPA
* @param	tau_d_i;			time decay inhibitory/GABA
* @param	inhibitoryPolicy	type of organization policy of inhibitory neurons
* @param	adjacencyPolicy		type of policy for the adjacency matrix initialization
* @param	weightPolicy		type of policy for the coupling weights matrix initialization
* @param	etaPolicy			type of policy for the bifurcation parameter array initialization
* @param	vPolicy			    type of policy for the membrane potential array initialization
* @param	ratio				ratio of excitatory neurons
*/
struct neurons initNeurons(int n, float vp, float vr, double g, float epsilon1, float epsilon2, float dt, float tau_m, float tau_d_e, float tau_d_i, char inhibitoryPolicy, char* adjacencyPolicy, char weightPolicy, char etaPolicy, char vPolicy, float ratio)
{    	
	int i, j, a, w = sqrt(n);
	
	struct neurons neurons;
	neurons.n = n;
	neurons.vp = vp;
	neurons.vr = vr;
	neurons.g = g;
	neurons.epsilon1 = epsilon1;
	neurons.epsilon2 = epsilon2;
	neurons.dt = dt;
	neurons.tau_m = tau_m;
	neurons.tau_d_e = tau_d_e;
	neurons.tau_d_i = tau_d_i;
	neurons.ratio = ratio;
	int nbInhibition = n - n/100.0*ratio;
	float weightMaxValue = 1.0;
	
	//Allocate and initialize type of neuron array
	neurons.type_neuron = (int *) malloc(sizeof(int) * n);
	for(i=0; i<n; i++)
	{
		switch(inhibitoryPolicy)
		{
			case 'o' :	//Excitatory and inhibitory neurons in order
				//The first ratio% are excitatory neurons
				if(i<(n/100.0*ratio))
				{
					neurons.type_neuron[i] = 0;
				}
				else	//The % remaining are inhibitory neurons
				{
					if((i%2)==0)		//one out of two neurons is hebbian inhibitory, the other anti-hebbian inhibitory
					//if((i<85) || ((i>=95) && (i<100)))		//2 groups of anti-hebbian inhibitory and hebbian inhibitory
					{
						neurons.type_neuron[i] = 1;
					}
					else
					{
						neurons.type_neuron[i] = 2;
					}
				}
				break;
			case 'r' :	//Initialize inhibitory array in a uniformly random way according to the ratio of excitatory
				if((rand()%(int)(100.0/(100.0-ratio))) == 0)
				{
					if((i%2)==0)		//one out of two neurons is hebbian inhibitory, the other anti-hebbian inhibitory
					{
						neurons.type_neuron[i] = 1;
					}
					else
					{
						neurons.type_neuron[i] = 2;
					}
				}
				else
				{
					neurons.type_neuron[i] = 0;
				}
				break;
			case 'c' :	//Initialize inhibitory array with pool at the center
				if(((i%w)>=(int)((w/2.0)-sqrt(nbInhibition)/2.0)) && ((i%w)<=(int)((w/2.0)+sqrt(nbInhibition)/2.0)) && ((i/w)>=(int)((w/2.0)-sqrt(nbInhibition)/2.0)) && ((i/w)<=(int)((w/2.0)+sqrt(nbInhibition)/2.0)))
				{
					if((i%2)==0)		//one out of two neurons is hebbian inhibitory, the other anti-hebbian inhibitory
					{
						neurons.type_neuron[i] = 1;
					}
					else
					{
						neurons.type_neuron[i] = 2;
					}
				}
				else
				{
					neurons.type_neuron[i] = 0;
				}
				break;
			case '4' :	//Initialize inhibitory array with 4 pools (each cortical areas)
				if( (((i%w)>=(int)((w/4.0)-sqrt(nbInhibition/4.0)/2.0)) && ((i%w)<=(int)((w/4.0)+sqrt(nbInhibition/4.0)/2.0)) && ((i/w)>=(int)((w/4.0)-sqrt(nbInhibition/4.0)/2.0)) && ((i/w)<=(int)((w/4.0)+sqrt(nbInhibition/4.0)/2.0))) || (((i%w)>=(int)((3.0*w/4.0)-sqrt(nbInhibition/4.0)/2.0)) && ((i%w)<=(int)((3.0*w/4.0)+sqrt(nbInhibition/4.0)/2.0)) && ((i/w)>=(int)((w/4.0)-sqrt(nbInhibition/4.0)/2.0)) && ((i/w)<=(int)((w/4.0)+sqrt(nbInhibition/4.0)/2.0))) || (((i%w)>=(int)((w/4.0)-sqrt(nbInhibition/4.0)/2.0)) && ((i%w)<=(int)((w/4.0)+sqrt(nbInhibition/4.0)/2.0)) && ((i/w)>=(int)((3.0*w/4.0)-sqrt(nbInhibition/4.0)/2.0)) && ((i/w)<=(int)((3.0*w/4.0)+sqrt(nbInhibition/4.0)/2.0))) || (((i%w)>=(int)((3.0*w/4.0)-sqrt(nbInhibition/4.0)/2.0)) && ((i%w)<=(int)((3.0*w/4.0)+sqrt(nbInhibition/4.0)/2.0)) && ((i/w)>=(int)((3.0*w/4.0)-sqrt(nbInhibition/4.0)/2.0)) && ((i/w)<=(int)((3.0*w/4.0)+sqrt(nbInhibition/4.0)/2.0))))
				{
					if((i%2)==0)		//one out of two neurons is hebbian inhibitory, the other anti-hebbian inhibitory
					{
						neurons.type_neuron[i] = 1;
					}
					else
					{
						neurons.type_neuron[i] = 2;
					}
				}
				else
				{
					neurons.type_neuron[i] = 0;
				}
				break;
			default :	//Excitatory and inhibitory neurons in order
				//The first ratio% are excitatory neurons
				if(i<(n/100.0*ratio))
				{
					neurons.type_neuron[i] = 0;
				}
				else	//The % remaining are inhibitory neurons
				{
					if((i%2)==0)		//one out of two neurons is hebbian inhibitory, the other anti-hebbian inhibitory
					{
						neurons.type_neuron[i] = 1;
					}
					else
					{
						neurons.type_neuron[i] = 2;
					}
				} 
				break;
		}
	}
	
	
	//Allocate and initialize adjacency matrix
	neurons.a = (int **) malloc(sizeof(int *) * n);
	
	for (i=0; i<n; i++) 
	{
		neurons.a[i] = (int *) malloc(sizeof(int) * n);
		for (j=0; j<n; j++) 
		{
			neurons.a[i][j] = 0;							//By default no connected
			
			if(strcmp(adjacencyPolicy, "f")==0)					//Fully connected, no recurrent links
			{
				if(i!=j)
				{
					neurons.a[i][j] = 1;
				}
			}
			if(strcmp(adjacencyPolicy, "n")==0)					//Not connected
			{
				neurons.a[i][j] = 0;
			}
			else if(strcmp(adjacencyPolicy, "fr")==0)			//Fully connected with recurrent links (all)
			{
				neurons.a[i][j] = 1;
			}
			else if(strcmp(adjacencyPolicy, "fe")==0)			//Fully connected except one neuron with no connection
			{
				if(i!=0)
				{
					neurons.a[i][j] = 1;
				}
			}
			else if(strcmp(adjacencyPolicy, "ru")==0)			//Randomly uniform connected
			{
				neurons.a[i][j] = rand()%2;
			}
			else if(strcmp(adjacencyPolicy, "r23")==0)			//Randomly 2/3 connected
			{
				if((rand()%3)%2 == 0)
				{
					neurons.a[i][j] = 1;
				}
			}
			else if(strcmp(adjacencyPolicy, "r13")==0)			//Randomly 1/3 connected
			{
				if((rand()%3)%2 == 1)
				{
					neurons.a[i][j] = 1;
				}
			}
			else if(strcmp(adjacencyPolicy, "r14")==0)			//Randomly 1/4 connected
			{
				if((rand()%4) == 0)
				{
					neurons.a[i][j] = 1;
				}
			}
			else if(strcmp(adjacencyPolicy, "r34")==0)			//Randomly 3/4 connected
			{
				if((rand()%4) != 0)
				{
					neurons.a[i][j] = 1;
				}
			}
			else if(strcmp(adjacencyPolicy, "r120")==0)			//Randomly 1/20 connected
			{
				if((rand()%20) == 0)
				{
					neurons.a[i][j] = 1;
				}
			}
			else if(strcmp(adjacencyPolicy, "o")==0)			//One to one connected (in index order)
			{
				if(i == j+1)
				{
					neurons.a[i][j] = 1;
				}
			}
			else if(strcmp(adjacencyPolicy, "t")==0)			//Two by two connected (in index order)
			{
				if((i == j+1) || (i == j-1))
				{
					neurons.a[i][j] = 1;
				}
			}
			else if(strcmp(adjacencyPolicy, "tr")==0)			//Two by two and recurrently connected (in index order)
			{
				if((i == j+1) || (i == j-1) || (i == j))
				{
					neurons.a[i][j] = 1;
				}
			}
			else if(strcmp(adjacencyPolicy, "c")==0)			//Two by two in circle connected (in index order)
			{
				if((i == j+1) || (i == j-1))
				{
					neurons.a[i][j] = 1;
				}
				if(i == 0)
				{
					neurons.a[i][n-1] = 1;
				}
				if(i == (n-1))
				{
					neurons.a[i][0] = 1;
				}
			}
			else if(strcmp(adjacencyPolicy, "oa")==0)			//One(0) to all(1-N) connected (connected in index order)
			{
				if(i!=0)
				{
					neurons.a[i][0] = 1;
				}
				if((i == j+1) || (i == j-1))
				{
					neurons.a[i][j] = 1;
				}
			}
			else if(strcmp(adjacencyPolicy, "oaf")==0)			//One(0) to all(1-N) connected with feedback (connected in index order)
			{
				if(i!=0)
				{
					neurons.a[i][0] = 1;
				}
				if((i == j+1) || (i == j-1))
				{
					neurons.a[i][j] = 1;
					if(j!=0)
					{
						neurons.a[0][j] = 1;
					}
				}
			}
			else if(strcmp(adjacencyPolicy, "ao")==0)			//All(1-N) to one connected (0)
			{
				if((i==0) && (j!=0))
				{
					neurons.a[i][j] = 1;
				}
				else if((j==0) && (i!=0))
				{
					neurons.a[i][j] = 1;
				}
			}
		}
	}
	if(strcmp(adjacencyPolicy, "rre")==0)						
	{
		createRandomRegular(neurons.a, n/2.0, n, ratio);			//Random regular connected, K=n/2 
	}
	else if(strcmp(adjacencyPolicy, "sw")==0)			
	{
		createSmallWorld(neurons.a, n/4.0, 0.1, n);					//Small world connected, K=n/4, p=0.1
	}
	else if(strcmp(adjacencyPolicy, "sf")==0)
	{
		createScaleFree(neurons.a, (0.05*n), 0.05*n, n);			//Scale free connected, K=m0, m0=0.05n
	}
	else if(strcmp(adjacencyPolicy, "swr")==0)			
	{
		createSmallWorldRandom(neurons.a, n/4.0, 0.1, n);			//Small world random connected, K=n/4, p=0.1
	}
	else if(strcmp(adjacencyPolicy, "sfr")==0)
	{
		createScaleFreeRandom(neurons.a, 0.05*n, n);				//Scale free random connected, m0=0.05n
	}
	else if(strcmp(adjacencyPolicy, "mod")==0)
	{
		createModular(neurons.a, n/4.0, n);							//Modular connected, K=n/4
	}
	else if(strcmp(adjacencyPolicy, "ml")==0)
	{
		createMultiLayer(neurons.a, 4, n);							//Multi layer connected, nb layer = 4
	}
	else if(strcmp(adjacencyPolicy, "mlf")==0)
	{
		createMultiLayerFeedback(neurons.a, 4, n);					//Multi layer connected, nb layer = 4
	}
	else if(strcmp(adjacencyPolicy, "af")==0)
	{
		createAllToFull(neurons.a, 10, n);							//All to full connected, nb input = 10
	}
	else if(strcmp(adjacencyPolicy, "res")==0)
	{
		createReservoir(neurons.a, 6, 2, n);						//Reservoirconnected, nb input = 6, nb output = 2
	}
	else if(strcmp(adjacencyPolicy, "map")==0)			
	{
		createMapTopology(neurons.a, 5, n/5);						//Map topology, n = w*h, ratio /5
	}
	else if(strcmp(adjacencyPolicy, "rtp")==0)			
	{
		createPhysicalTopologyDependant(neurons.a, n, sqrt(n), neurons.type_neuron);			//Random topology depending on physical distances
	}
	else if(strcmp(adjacencyPolicy, "rtp2")==0)			
	{
		createPhysicalTopologyDependant_2(neurons.a, n, sqrt(n), neurons.type_neuron);			//Random topology depending on physical distances
	}
	else if(strcmp(adjacencyPolicy, "rtp3")==0)			
	{
		createPhysicalTopologyDependant_3(neurons.a, n, sqrt(n), neurons.type_neuron);			//Random topology depending on physical distances
	}
	
	
	//Allocate and initialize coupling weights array
	neurons.w = (long double **) malloc(sizeof(long double *) * n);
	a = n - (n/100.0*ratio); //number of inhibitory neurons
	for (i=0; i<n; i++) 
	{
		neurons.w[i] = (long double *) malloc(sizeof(long double) * n);
		for (j=0; j<n; j++) 
		{
			if(neurons.a[i][j]==1)  //if the link between node i and j exists
			{
				switch(weightPolicy)
				{
					case 'g' :	//Random value between -1.0 and 1.0 (depending of type of neuron) distributed with a normal/gaussian law mean= +/-0.5, std=0.2
						if(neurons.type_neuron[j]!=0)
						{
							neurons.w[i][j] = get_random_normal(0.0, 0.2, -1.0, 0.0);
						}
						else
						{
							neurons.w[i][j] = get_random_normal(0.0, 0.2, 0.0, 1.0);
						}
						break;
					case 'r' :	//Random value uniform between -1 and 1 (depending of type of neuron)
						if(neurons.type_neuron[j]!=0)
						{
							neurons.w[i][j] = get_random(-0.9999, 0.0001);
						}
						else
						{
							neurons.w[i][j] = get_random(0.0001, 0.9999);
						}
						//neurons.w[i][j] = get_random(-0.9999, 0.9999);
						break;
					case 'd' :	//Double cluster intialisation
						if(neurons.type_neuron[j]==1 && (neurons.type_neuron[i]==1 || neurons.type_neuron[i]==2))
						{
							if((i<(n/100.0*ratio)+a/2 && j<(n/100.0*ratio)+a/2) || (i>=(n/100.0*ratio)+a/2 && j>=(n/100.0*ratio)+a/2))
							{
								neurons.w[i][j] = 0.0;
							}
							else
							{
								neurons.w[i][j] = -weightMaxValue;
							}
							//neurons.w[i][j] = get_random(-weightMaxValue, 0.0);
						}
						else if(neurons.type_neuron[j]==2 && (neurons.type_neuron[i]==1 || neurons.type_neuron[i]==2))
						{
							if((i<(n/100.0*ratio)+a/2 && j<(n/100.0*ratio)+a/2) || (i>=(n/100.0*ratio)+a/2 && j>=(n/100.0*ratio)+a/2))
							{
								neurons.w[i][j] = -weightMaxValue;
							}
							else
							{
								neurons.w[i][j] = 0.0;
							}
							//neurons.w[i][j] = get_random(-weightMaxValue, 0.0);
						}
						else if(neurons.type_neuron[j]==1)
						{
							if((i<(n/100.0*ratio)/2 && j<(n/100.0*ratio)+a/2) || (i>=(n/100.0*ratio)/2 && j>=(n/100.0*ratio)+a/2))
							{
								neurons.w[i][j] = 0.0;
							}
							else
							{
								neurons.w[i][j] = -weightMaxValue;
							}
							//neurons.w[i][j] = get_random(-weightMaxValue, 0.0);
						}
						else if(neurons.type_neuron[j]==2)
						{
							if((i<(n/100.0*ratio)/2 && j<(n/100.0*ratio)+a/2) || (i>=(n/100.0*ratio)/2 && j>=(n/100.0*ratio)+a/2))
							{
								neurons.w[i][j] = -weightMaxValue;
							}
							else
							{
								neurons.w[i][j] = 0.0;
							}
							//neurons.w[i][j] = get_random(-weightMaxValue, 0.0);
						}
						else if(neurons.type_neuron[i]==1 || neurons.type_neuron[i]==2)
						{
							if((i<(n/100.0*ratio)+a/2 && j<(n/100.0*ratio)/2) || (i>=(n/100.0*ratio)+a/2 && j>=(n/100.0*ratio)/2))
							{
								neurons.w[i][j] = weightMaxValue;
							}
							else
							{
								neurons.w[i][j] = 0.0;
							}
							//neurons.w[i][j] = get_random(0.0, weightMaxValue);
						}
						else
						{
							if((i<(n/100.0*ratio)/2 && j<(n/100.0*ratio)/2) || (i>=(n/100.0*ratio)/2 && j>=(n/100.0*ratio)/2))
							{
								neurons.w[i][j] = weightMaxValue;
							}
							else
							{
								neurons.w[i][j] = 0.0;
							}
							//neurons.w[i][j] = get_random(0.0, weightMaxValue);
						}
						break;
					case 'p' :	//Two proto clusters intialisation
						if(neurons.type_neuron[j]==1 && (neurons.type_neuron[i]==1 || neurons.type_neuron[i]==2))
						{
							if((i<(n/100.0*ratio)+a/2 && j<(n/100.0*ratio)+a/2) || (i>=(n/100.0*ratio)+a/2 && j>=(n/100.0*ratio)+a/2))
							{
								neurons.w[i][j] = get_random_normal(0.0, 0.15, -1.0, 0.0);
							}
							else
							{
								neurons.w[i][j] = -0.7;
							}
						}
						else if(neurons.type_neuron[j]==2 && (neurons.type_neuron[i]==1 || neurons.type_neuron[i]==2))
						{
							if((i<(n/100.0*ratio)+a/2 && j<(n/100.0*ratio)+a/2) || (i>=(n/100.0*ratio)+a/2 && j>=(n/100.0*ratio)+a/2))
							{
								neurons.w[i][j] = -0.7;
							}
							else
							{
								neurons.w[i][j] = get_random_normal(0.0, 0.15, -1.0, 0.0);
							}
						}
						else if(neurons.type_neuron[j]==1)
						{
							if((i<(n/100.0*ratio)/2 && j<(n/100.0*ratio)+a/2) || (i>=(n/100.0*ratio)/2 && j>=(n/100.0*ratio)+a/2))
							{
								neurons.w[i][j] = get_random_normal(0.0, 0.15, -1.0, 0.0);
							}
							else
							{
								neurons.w[i][j] = -0.7;
							}
						}
						else if(neurons.type_neuron[j]==2)
						{
							if((i<(n/100.0*ratio)/2 && j<(n/100.0*ratio)+a/2) || (i>=(n/100.0*ratio)/2 && j>=(n/100.0*ratio)+a/2))
							{
								neurons.w[i][j] = -0.7;
							}
							else
							{
								neurons.w[i][j] = get_random_normal(0.0, 0.15, -1.0, 0.0);
							}
						}
						else if(neurons.type_neuron[i]==1 || neurons.type_neuron[i]==2)
						{
							if((i<(n/100.0*ratio)+a/2 && j<(n/100.0*ratio)/2) || (i>=(n/100.0*ratio)+a/2 && j>=(n/100.0*ratio)/2))
							{
								neurons.w[i][j] = 0.7;
							}
							else
							{
								neurons.w[i][j] = get_random_normal(0.0, 0.15, 0.0, 1.0);
							}
						}
						else
						{
							if((i<(n/100.0*ratio)/2 && j<(n/100.0*ratio)/2) || (i>=(n/100.0*ratio)/2 && j>=(n/100.0*ratio)/2))
							{
								neurons.w[i][j] = 0.7;
							}
							else
							{
								neurons.w[i][j] = get_random_normal(0.0, 0.15, 0.0, 1.0);
							}
						}
						break;
					case 'm' :	//Minimal conditions two clusters intialisation
						//Hebbian inhibitory to Anti-Hebbian inhibitory
						if(neurons.type_neuron[j]==1 && neurons.type_neuron[i]==2)
						{
							//To its cluster 
							if((i<(n/100.0*ratio)+a/2 && j<(n/100.0*ratio)+a/2) || (i>=(n/100.0*ratio)+a/2 && j>=(n/100.0*ratio)+a/2))
							{
								neurons.w[i][j] = 0.0;
							}
							//To the rest 
							else
							{
								neurons.w[i][j] = -weightMaxValue;
							}
						}
						//Hebbian inhibitory to Hebbian inhibitory
						else if(neurons.type_neuron[j]==1 && neurons.type_neuron[i]==1)
						{
							//To its cluster 
							if((i<(n/100.0*ratio)+a/2 && j<(n/100.0*ratio)+a/2) || (i>=(n/100.0*ratio)+a/2 && j>=(n/100.0*ratio)+a/2))
							{
								neurons.w[i][j] = 0.0;
							}
							//To the rest 
							else
							{
								neurons.w[i][j] = -weightMaxValue;
							}
						}
						//Anti-hebbian inhibitory to Hebbian inhibitory
						else if(neurons.type_neuron[j]==2 && neurons.type_neuron[i]==1)
						{
							//To its cluster 
							if((i<(n/100.0*ratio)+a/2 && j<(n/100.0*ratio)+a/2) || (i>=(n/100.0*ratio)+a/2 && j>=(n/100.0*ratio)+a/2))
							{
								neurons.w[i][j] = -weightMaxValue;
							}
							//To the rest 
							else
							{
								neurons.w[i][j] = 0.0;
							}
						}
						//Anti-hebbian inhibitory to Anti-Hebbian inhibitory
						else if(neurons.type_neuron[j]==2 && neurons.type_neuron[i]==2)
						{
							//To its cluster 
							if((i<(n/100.0*ratio)+a/2 && j<(n/100.0*ratio)+a/2) || (i>=(n/100.0*ratio)+a/2 && j>=(n/100.0*ratio)+a/2))
							{
								neurons.w[i][j] = -weightMaxValue;
							}
							//To the rest 
							else
							{
								neurons.w[i][j] = 0.0;
							}
						}
						//Hebbian inhibitory to excitatory
						else if(neurons.type_neuron[j]==1)
						{
							//To its cluster 
							if((i<(n/100.0*ratio)/2 && j<(n/100.0*ratio)+a/2) || (i>=(n/100.0*ratio)/2 && j>=(n/100.0*ratio)+a/2))
							{
								neurons.w[i][j] = 0.0;
							}
							//To the rest 
							else
							{
								neurons.w[i][j] = -weightMaxValue;
							}
						}
						//Anti-hebbian inhibitory to excitatory
						else if(neurons.type_neuron[j]==2)
						{
							//To its cluster 
							if((i<(n/100.0*ratio)/2 && j<(n/100.0*ratio)+a/2) || (i>=(n/100.0*ratio)/2 && j>=(n/100.0*ratio)+a/2))
							{
								neurons.w[i][j] = -weightMaxValue;
							}
							//To the rest 
							else
							{
								neurons.w[i][j] = 0.0;
							}
						}
						//Excitatory to Hebbian inhibitory
						else if(neurons.type_neuron[i]==1)
						{
							//To its cluster 
							if((i<(n/100.0*ratio)+a/2 && j<(n/100.0*ratio)/2) || (i>=(n/100.0*ratio)+a/2 && j>=(n/100.0*ratio)/2))
							{
								neurons.w[i][j] = get_random(0.0, 0.99999); //get_random(0.0, 0.99999)
							}
							//To the rest 
							else
							{
								neurons.w[i][j] = get_random(0.0, 0.99999); //get_random(0.0, 0.99999)
							}
						}
						//Excitatory to Anti-hebbian inhibitory
						else if(neurons.type_neuron[i]==2)
						{
							//To its cluster 
							if((i<(n/100.0*ratio)+a/2 && j<(n/100.0*ratio)/2) || (i>=(n/100.0*ratio)+a/2 && j>=(n/100.0*ratio)/2))
							{
								neurons.w[i][j] = get_random(0.0, 0.99999); //get_random(0.0, 0.99999)
							}
							//To the rest 
							else
							{
								neurons.w[i][j] = get_random(0.0, 0.99999); //get_random(0.0, 0.99999)
							}
						}
						//Excitatory to excitatory
						else
						{
							//To its cluster 
							if((i<(n/100.0*ratio)/2 && j<(n/100.0*ratio)/2) || (i>=(n/100.0*ratio)/2 && j>=(n/100.0*ratio)/2))
							{
								neurons.w[i][j] = get_random(0.0, 0.99999);						
							}
							//To the rest 
							else
							{
								neurons.w[i][j] = get_random(0.0, 0.99999);	
							}
						}
						break;
					case 'f' :	//Four cluster intialisation
						if(neurons.type_neuron[j]==1 && (neurons.type_neuron[i]==1 || neurons.type_neuron[i]==2))
						{
							if(i<(n/100.0*ratio)+a/4 && j<(n/100.0*ratio)+a/4)
							{
								neurons.w[i][j] = 0.0;
							}
							else if( (i<(n/100.0*ratio)+2*a/4 && j<(n/100.0*ratio)+2*a/4 ) && (i>=(n/100.0*ratio)+a/4 && j>=(n/100.0*ratio)+a/4) )
							{
								neurons.w[i][j] = 0.0;
							}
							else if( (i<(n/100.0*ratio)+3*a/4 && j<(n/100.0*ratio)+3*a/4 ) && (i>=(n/100.0*ratio)+2*a/4 && j>=(n/100.0*ratio)+2*a/4) )
							{
								neurons.w[i][j] = 0.0;
							}
							else if( (i<(n/100.0*ratio)+4*a/4 && j<(n/100.0*ratio)+4*a/4 ) && (i>=(n/100.0*ratio)+3*a/4 && j>=(n/100.0*ratio)+3*a/4) )
							{
								neurons.w[i][j] = 0.0;
							}
							else
							{
								neurons.w[i][j] = -weightMaxValue;
							}
						}
						else if(neurons.type_neuron[j]==2 && (neurons.type_neuron[i]==1 || neurons.type_neuron[i]==2))
						{
							if(i<(n/100.0*ratio)+a/4 && j<(n/100.0*ratio)+a/4)
							{
								neurons.w[i][j] = -weightMaxValue;
							}
							else if( (i<(n/100.0*ratio)+2*a/4 && j<(n/100.0*ratio)+2*a/4 ) && (i>=(n/100.0*ratio)+a/4 && j>=(n/100.0*ratio)+a/4) )
							{
								neurons.w[i][j] = -weightMaxValue;
							}
							else if( (i<(n/100.0*ratio)+3*a/4 && j<(n/100.0*ratio)+3*a/4 ) && (i>=(n/100.0*ratio)+2*a/4 && j>=(n/100.0*ratio)+2*a/4) )
							{
								neurons.w[i][j] = -weightMaxValue;
							}
							else if( (i<(n/100.0*ratio)+4*a/4 && j<(n/100.0*ratio)+4*a/4 ) && (i>=(n/100.0*ratio)+3*a/4 && j>=(n/100.0*ratio)+3*a/4) )
							{
								neurons.w[i][j] = -weightMaxValue;
							}
							else
							{
								neurons.w[i][j] = 0.0;
							}
						}
						else if(neurons.type_neuron[j]==1)
						{
							if(i<(n/100.0*ratio)/4 && j<(n/100.0*ratio)+a/4)
							{
								neurons.w[i][j] = 0.0;
							}
							else if( (i<2*(n/100.0*ratio)/4 && j<(n/100.0*ratio)+2*a/4 ) && (i>=(n/100.0*ratio)/4 && j>=(n/100.0*ratio)+a/4) )
							{
								neurons.w[i][j] = 0.0;
							}
							else if( (i<3*(n/100.0*ratio)/4 && j<(n/100.0*ratio)+3*a/4 ) && (i>=2*(n/100.0*ratio)/4 && j>=(n/100.0*ratio)+2*a/4) )
							{
								neurons.w[i][j] = 0.0;
							}
							else if( (i<4*(n/100.0*ratio)/4 && j<(n/100.0*ratio)+4*a/4 ) && (i>=3*(n/100.0*ratio)/4 && j>=(n/100.0*ratio)+3*a/4) )
							{
								neurons.w[i][j] = 0.0;
							}
							else
							{
								neurons.w[i][j] = -weightMaxValue;
							}
						}
						else if(neurons.type_neuron[j]==2)
						{
							if(i<(n/100.0*ratio)/4 && j<(n/100.0*ratio)+a/4)
							{
								neurons.w[i][j] = -weightMaxValue;
							}
							else if( (i<2*(n/100.0*ratio)/4 && j<(n/100.0*ratio)+2*a/4 ) && (i>=(n/100.0*ratio)/4 && j>=(n/100.0*ratio)+a/4) )
							{
								neurons.w[i][j] = -weightMaxValue;
							}
							else if( (i<3*(n/100.0*ratio)/4 && j<(n/100.0*ratio)+3*a/4 ) && (i>=2*(n/100.0*ratio)/4 && j>=(n/100.0*ratio)+2*a/4) )
							{
								neurons.w[i][j] = -weightMaxValue;
							}
							else if( (i<4*(n/100.0*ratio)/4 && j<(n/100.0*ratio)+4*a/4 ) && (i>=3*(n/100.0*ratio)/4 && j>=(n/100.0*ratio)+3*a/4) )
							{
								neurons.w[i][j] = -weightMaxValue;
							}
							else
							{
								neurons.w[i][j] = 0.0;
							}
						}
						else if(neurons.type_neuron[i]==1 || neurons.type_neuron[i]==2)
						{
							if(i<(n/100.0*ratio)+a/4 && j<(n/100.0*ratio)/4)
							{
								neurons.w[i][j] = weightMaxValue;
							}
							else if( (i<(n/100.0*ratio)+2*a/4 && j<2*(n/100.0*ratio)/4) && (i>=(n/100.0*ratio)+a/4 && j>=(n/100.0*ratio)/4) )
							{
								neurons.w[i][j] = weightMaxValue;
							}
							else if( (i<(n/100.0*ratio)+3*a/4 && j<3*(n/100.0*ratio)/4) && (i>=(n/100.0*ratio)+2*a/4 && j>=2*(n/100.0*ratio)/4) )
							{
								neurons.w[i][j] = weightMaxValue;
							}
							else if( (i<(n/100.0*ratio)+4*a/4 && j<4*(n/100.0*ratio)/4) && (i>=(n/100.0*ratio)+3*a/4 && j>=3*(n/100.0*ratio)/4) )
							{
								neurons.w[i][j] = weightMaxValue;
							}
							else
							{
								neurons.w[i][j] = 0.0;
							}		
						}
						else
						{
							if(i<(n/100.0*ratio)/4 && j<(n/100.0*ratio)/4)
							{
								neurons.w[i][j] = weightMaxValue;
							}
							else if( (i<2*(n/100.0*ratio)/4 && j<2*(n/100.0*ratio)/4) && (i>=(n/100.0*ratio)/4 && j>=(n/100.0*ratio)/4) )
							{
								neurons.w[i][j] = weightMaxValue;
							}
							else if( (i<3*(n/100.0*ratio)/4 && j<3*(n/100.0*ratio)/4) && (i>=2*(n/100.0*ratio)/4 && j>=2*(n/100.0*ratio)/4) )
							{
								neurons.w[i][j] = weightMaxValue;
							}
							else if( (i<4*(n/100.0*ratio)/4 && j<4*(n/100.0*ratio)/4) && (i>=3*(n/100.0*ratio)/4 && j>=3*(n/100.0*ratio)/4) )
							{
								neurons.w[i][j] = weightMaxValue;
							}
							else
							{
								neurons.w[i][j] = 0.0;
							}
						}
						break;
					case 'n' :	//Constant value null
						neurons.w[i][j] = 0.0;
						break;
			
					default : //Random value uniform between -1 and 1 (depending of type of neuron)
						if(neurons.type_neuron[j]!=0)
						{
							neurons.w[i][j] = get_random(-0.9999, 0.0001);
						}
						else
						{
							neurons.w[i][j] = get_random(0.0001, 0.9999);
						}
						break;
				}
			}
			else
			{
				neurons.w[i][j] = 0.0001;
			}
		}
	}	
	
	
	//Allocate and initialize inputs array
	neurons.inputs =  (double *) calloc(n, sizeof(double));
	
	
	//Allocate and initialize time spikes array
	neurons.t_spikes =  (long double *) calloc(n, sizeof(long double));
	
	
	//Allocate and initialize time refractory array
	neurons.t_refractory =  (long double *) calloc(n, sizeof(long double));
	
	
	//Allocate and initialize state array
	neurons.state_neuron =  (int *) calloc(n, sizeof(int));
	
	
	//Allocate and initialize bifurcation parameter array
	neurons.eta = (float *) malloc(sizeof(float) * n);
	for(i=0; i<n; i++)
	{
		switch(etaPolicy)
		{
			case 'm' :	//Identical negative value
				neurons.eta[i] = -pow(2.0*M_PI*tau_m, 2.0); //-2HZ
				break;
			case 'n' :	//Identical null value
				neurons.eta[i] = 0.0; 
				break;
			case 'p' :	//Identical positive value
				neurons.eta[i] = pow(2.0*M_PI*tau_m, 2.0);	//2 Hz
				break;
			case 't' :	//Two values if excitatory or inhibitory
			    if(neurons.type_neuron[i]==0)
			   	{
			   		neurons.eta[i] = pow(M_PI*tau_m, 2.0); //1HZ
			   	}
				else
				{
					neurons.eta[i] = pow((2.0*M_PI*tau_m), 2.0); //2HZ
				}
				break;
			case 'r' :	//Non-identical uniformaly random value between 0.0 and 1.0 HZ
				neurons.eta[i] = get_random(0.0, pow(M_PI*tau_m, 2.0)); 
				break;
			case 'g' :	//Non-identical normallly random value between 0.0 and 2.0 HZ, mean=1HZ, std=0.5HZ
				//neurons.eta[i] = get_random_normal(pow(M_PI, 2.0), pow((0.5*M_PI), 2.0), 0.0, pow((2.0*M_PI), 2.0)); //mean=1HZ, std=0.5HZ range=[0-1-2]HZ 
				//neurons.eta[i] = get_random_normal(pow((0.2*M_PI), 2.0), pow((0.05*M_PI), 2.0), pow((0.1*M_PI), 2.0), pow((0.3*M_PI), 2.0)); //mean=0.2HZ, std=0.05HZ range=[0.1-0.2-0.3]HZ 
				neurons.eta[i] = get_random_normal(0.0, pow(1.0*M_PI*tau_m, 2.0), -pow(2.0*M_PI*tau_m, 2.0), pow(2.0*M_PI*tau_m, 2.0));
				break;
			default :	//Identical value of 0.0
				neurons.eta[i] = 0.0; 
				break;
		}
	}
	
	//Allocate and initialize synaptic excitatory inputs array
	neurons.s_e =  (long double *) calloc(n, sizeof(long double));
	
	//Allocate and initialize synaptic hebbian inhibitory inputs array
	neurons.s_h_i =  (long double *) calloc(n, sizeof(long double));
	
	//Allocate and initialize synaptic anti-hebbian inhibitory inputs array
	neurons.s_a_i =  (long double *) calloc(n, sizeof(long double));
	
	//Allocate and initialize the membrane potential array
	neurons.v = (long double *) malloc(sizeof(long double) * n);
	for(i=0; i<n; i++)
	{
		switch(vPolicy)
		{
			case 'r' :	//Random value between vr and vp
				neurons.v[i] = get_random(vr, vp); 
				break;
			case 'd' :	
				if((i<40) || ((i>=80) && (i<90)))	
				{
					neurons.v[i] = vp;
				}
				else
				{
					neurons.v[i] = vr;
				}
				break;
			case 'c' :	//Constant value of 0
				neurons.v[i] = 0.0;
				break;
			case '2' :	//Two states (0 and vp)
				neurons.v[i] = (rand()%2)*vp;
				break;
			case '3' :	//Three states (0, vp/2, vp)
				neurons.v[i] = (rand()%3)*vp/2.0;
				break;
			case 'o' :	//n states ordered
				neurons.v[i] = ((double)i/n)*2.0*vp-vr;
				break;
			default :	//Random value between vr and vp
				neurons.v[i] = get_random(vr, vp); 
				break;
		}
	}
	
	return neurons;
}



/*
* Initialize a network of QIF neurons with a save
*
* @param	n					number of neurons
* @param	vp					peak value 
* @param	vr					reset value 	
* @param	g 					(chemical) global coupling strength
* @param	epsilon1			learning rate for the slow adaptation
* @param	epsilon2			learning rate for the fast adaptation
* @param	dt 		 			integration time step
* @param	tau_m				membrane time constant
* @param	tau_d_e				time decay excitatory/AMPA
* @param	tau_d_i;			time decay inhibitory/GABA
* @param	ratio				ratio of excitatory neurons
*/
struct neurons initneuronsSaved(int n, float vp, float vr, double g, float epsilon1, float epsilon2, float dt, float tau_m, float tau_d_e, float tau_d_i, float ratio)
{    	
	FILE *fp;
	int i, j;
	
	struct neurons neurons;
	neurons.n = n;
	neurons.vp = vp;
	neurons.vr = vr;
	neurons.g = g;
	neurons.epsilon1 = epsilon1;
	neurons.epsilon2 = epsilon2;
	neurons.dt = dt;
	neurons.tau_m = tau_m;
	neurons.tau_d_e = tau_d_e;
	neurons.tau_d_i = tau_d_i;
	neurons.ratio = ratio;
	
	//Allocate and initialize adjacency matrix
	fp = fopen("save/adjacency.txt", "r");
	if(fp == NULL)
	{
		printf("Error!");   
		exit(1);             
	}
	neurons.a = (int **) malloc(sizeof(int *) * n);
	for (i=0; i<n; i++) 
	{
		neurons.a[i] = (int *) malloc(sizeof(int) * n);
		for (j=0; j<n; j++) 
		{
			fscanf(fp, "%d ", &neurons.a[i][j]);
		}
	}
	fclose(fp);
	
	//Allocate and initialize coupling weights array
	fp = fopen("save/weights_matrix_3.txt", "r");
	if(fp == NULL)
	{
		printf("Error!");   
		exit(1);             
	}
	neurons.w = (long double **) malloc(sizeof(long double *) * n);
	for (i=0; i<n; i++) 
	{
		neurons.w[i] = (long double *) malloc(sizeof(long double) * n);
		for (j=0; j<n; j++) 
		{
			fscanf(fp, "%Lf ", &neurons.w[i][j]);
		}
	}
	fclose(fp);	
	
	//Allocate and initialize inputs array
	neurons.inputs =  (double *) calloc(n, sizeof(double));
	
	//Allocate and initialize time spikes array
	neurons.t_spikes =  (long double *) calloc(n, sizeof(long double));
	
	//Allocate and initialize time refractory array
	neurons.t_refractory =  (long double *) calloc(n, sizeof(long double));
	
	//Allocate and initialize state array
	neurons.state_neuron =  (int *) calloc(n, sizeof(int));
	
	//Allocate and initialize  bifurcation parameter array
	fp = fopen("save/alpha.txt", "r");
	if(fp == NULL)
	{
		printf("Error!");   
		exit(1);             
	}
	neurons.eta = (float *) malloc(sizeof(float) * n);
	for(i=0; i<n; i++)
	{
		fscanf(fp, "%f ", &neurons.eta[i]);
	}
	fclose(fp);
	
	
	//Allocate and initialize synaptic excitatory inputs array
	neurons.s_e =  (long double *) calloc(n, sizeof(long double));
	
	//Allocate and initialize synaptic hebbian inhibitory inputs array
	neurons.s_h_i =  (long double *) calloc(n, sizeof(long double));
	
	//Allocate and initialize synaptic anti-hebbian inhibitory inputs array
	neurons.s_a_i =  (long double *) calloc(n, sizeof(long double));
	
	
	//Allocate and initialize the membrane potential array
	fp = fopen("save/phases.txt", "r");
	if(fp == NULL)
	{
		printf("Error!");   
		exit(1);             
	}
	fseek(fp, -((18*n)+1), SEEK_END);
	neurons.v = (long double *) malloc(sizeof(long double) * n);
	for(i=0; i<n; i++)
	{
		fscanf(fp, "%Lf ", &neurons.v[i]);
	}
	fclose(fp);
	
	
	//Allocate and initialize inhibitory array
	fp = fopen("save/inhibitory.txt", "r");
	if(fp == NULL)
	{
		printf("Error!");   
		exit(1);             
	}
	neurons.type_neuron = (int *) malloc(sizeof(int) * n);
	for(i=0; i<n; i++)
	{
		fscanf(fp, "%d ", &j);
		//The first ratio% are excitatory neurons
		if(j==1)
		{
			neurons.type_neuron[i] = 0;
		}
		else	//The 20% remaining are inhibitory neurons
		{
			if((i%2)==0)		//one out of two neurons is hebbian inhibitory, the other anti-hebbian inhibitory
			{
				neurons.type_neuron[i] = 1;
			}
			else
			{
				neurons.type_neuron[i] = 2;
			}
		}
	}
	fclose(fp);
	
	return neurons;
}


/*
* Free the memory of the neurons
*
* @param	neurons			pointer on the current network
*/
void freeNeurons(struct neurons *neurons)
{    	
	int i;
	
	//Free coupling weights array
	for (i=0; i<neurons->n; i++) 
	{
		free(neurons->w[i]);
	}
	free(neurons->w);
	
	//Free adjacency matrix
	for (i=0; i<neurons->n; i++) 
	{
		free(neurons->a[i]);
	}
	free(neurons->a);
	
	//Free type of neurons array
	free(neurons->type_neuron);
	
	//Free inputs array
	free(neurons->inputs);
	
	//Free time spikes array
	free(neurons->t_spikes);
	
	//Free time refractory array
	free(neurons->t_refractory);
	
	//Free state array
	free(neurons->state_neuron);
	
	//Free bifurcation parameter array
	free(neurons->eta);
	
	//Free the membrane potential array
	free(neurons->v);
	
	//Free the synaptic excitatory inputs array
	free(neurons->s_e);
	
	//Free the synaptic hebbian inhibitory inputs array
	free(neurons->s_h_i);
	
	//Free the synaptic anti-hebbian inhibitory inputs array
	free(neurons->s_a_i);
}



/*
* Euler method to update the synaptic input
*
* @param	s_i				the synaptic input of the neuron i
* @param	spikes			the spike vector for the neighbor of i
* @param	n				number of neurons
* @param	dt 		 		integration time step
* @param	tau				membrane decay time
* @param	a				the adjacency vector for i and its neighbors
* @param	w				the couping vector for i and its neighbors
* @param	type_neuron		the type of the neighbors neurons
* @param	type_selected	the type selected for the neurons
*/ 
long double euler_s(long double s_i, int *spikes, int n, float dt, float tau, int *a, long double *w, int *type_neuron, int type_selected)
{   
	int j, nrt=1;
	long double rt = 0.0;
	
	//for each neighbors of neuron i
	for(j=0; j < n; j++)
    {
		if(a[j] && (type_neuron[j]==type_selected))	//if the connection exists
		{
			rt += w[j]*spikes[j];
			nrt++;
		}
	}
  	
	return s_i + -dt*(1.0/tau)*s_i + rt/nrt;
} 



/*
* Euler method to update the membrane potential
*
* @param	v_i			the membrane potential of the neuron i
* @param	s_e			synaptic excitatory input of the neuron i
* @param	s_h_i		synaptic hebbian inhibitory input of the neuron i
* @param	s_a_i		synaptic anti-hebbian inhibitory input of the neuron i
* @param	g 			(chemical) global coupling strength
* @param	dt 		 	integration time step
* @param	tau			membrane time constant
* @param	eta			the bifurcation/excitability parameter of the neuron i
* @param	input		input added in the neuron i
*/ 
long double euler_v(long double v_i, long double s_e, long double s_h_i, long double s_a_i, double g, float dt, float tau, float eta, double input, int type_neuron)
{   
	if(type_neuron==0)
	{
		return v_i + (dt/tau)*(v_i*v_i + eta + 1.0*g*s_e + 2.0*g*s_h_i + 4.0*g*s_a_i +input) + sqrt(dt/tau)*get_random_normal(0.0, pow(4.0*M_PI*tau, 2.0), -pow(5.0*M_PI*tau, 2.0), pow(5.0*M_PI*tau, 2.0));
		//return v_i + (dt/tau)*(v_i*v_i + eta + 1.0*g*s_e + 2.0*g*s_h_i + 4.0*g*s_a_i +input); //No noise 
	}
	else
	{
		return v_i + (dt/tau)*(v_i*v_i + eta + 1.0*g*s_e + 2.0*g*s_h_i + 4.0*g*s_a_i +input) + sqrt(dt/tau)*get_random_normal(0.0, pow(4.0*M_PI*tau, 2.0), -pow(5.0*M_PI*tau, 2.0), pow(5.0*M_PI*tau, 2.0));
		//return v_i + (dt/tau)*(v_i*v_i + eta + 1.0*g*s_e + 2.0*g*s_h_i + 4.0*g*s_a_i +input); //No noise 
	}
} 



/*
* Update neurons' membrane potential 
*
* @param	neurons			pointer on the current network
* @param	spikes			if neurons spike
* @param	t				iteration time
* @param	fptr			file to register
* @param	save			if we register the spikes
*/   
void update_states(struct neurons *neurons, int* spikes, int t, FILE *fptr, int save)
{
    int i;
	
	int *oldSpikes = copyTable(spikes, neurons->n);				//copy spikes array at time t-1
		
	//for each neurons of the network
	for(i=0; i < neurons->n; i++)
    {   
		spikes[i] = 0;
		
		//(-1=refractory period negative 0=accumulation period 1=refractory period positive)
		if(neurons->state_neuron[i] == 0)	//if in accumulation period
		{
			neurons->s_e[i] = euler_s(neurons->s_e[i], oldSpikes, neurons->n, neurons->dt, neurons->tau_d_e, neurons->a[i], neurons->w[i], neurons->type_neuron, 0);

			neurons->s_h_i[i] = euler_s(neurons->s_h_i[i], oldSpikes, neurons->n, neurons->dt, neurons->tau_d_i, neurons->a[i], neurons->w[i],  neurons->type_neuron, 1);
			
			neurons->s_a_i[i] = euler_s(neurons->s_a_i[i], oldSpikes, neurons->n, neurons->dt, neurons->tau_d_i, neurons->a[i], neurons->w[i],  neurons->type_neuron, 2);
			
			neurons->v[i] = euler_v(neurons->v[i], neurons->s_e[i], neurons->s_h_i[i], neurons->s_a_i[i], neurons->g, neurons->dt, neurons->tau_m, neurons->eta[i], neurons->inputs[i], neurons->type_neuron[i]);
			
			if(neurons->v[i] >= neurons->vp)
			{
				neurons->state_neuron[i] = 1;
				neurons->t_refractory[i] = t*neurons->dt;
			}
			else if(neurons->v[i] < neurons->vr)
			{
				neurons->v[i] = neurons->vr;
			}
		}
	   	
		if((neurons->state_neuron[i] == 1) && (t*neurons->dt >= (neurons->t_refractory[i] + (1.0/neurons->v[i])*neurons->tau_m ))) 	//if the neuron spikes
		{
			neurons->state_neuron[i] = -1;
			spikes[i] = 1;
			neurons->t_spikes[i] = t*neurons->dt;		//conversion iteration time in second
			
			if(save)
			{
				saveSpike(fptr, i, neurons->t_spikes[i]);
			}
		}
		
		if((neurons->state_neuron[i] == -1) && (t*neurons->dt >= (neurons->t_refractory[i] + (2.0/neurons->v[i])*neurons->tau_m )))	//if the neuron had time to refract
		{
			neurons->state_neuron[i] = 0;
			neurons->v[i] = neurons->vr;
		}
    }
	
	free(oldSpikes); 	//free memory 
}


/*
* Plasticity function for Hebbian STDP asymmetric (causal)
*
* @param	t_i				time last spike of neuron i (in s)
* @param	t_j				time last spike of neuron j (in s)
* @param	delta_pot		potentiation brought by the plasticity function	
* @param	delta_dep		depression brought by the plasticity function	
* @param	forgetting		forgetting constant function added to the plasticity
*/ 
void hebbian_STDP_asymmetric(long double t_i, long double t_j, long double *delta_pot, long double *delta_dep, float forgetting)
{
	long double delta_s, delta_t = t_i-t_j;
	float t_pot = 0.02; 		//time window potentiation in s (~0 to 20 ms)
	float t_dep = 0.05; 		//time window depression in s (~0 to 20–100 ms)
	float a_pot = 3.0;			//amplitude potentiation
	float a_dep = 1.0;			//amplitude depression
	
	if(delta_t>=0)
	{
		delta_s = a_pot*exp2l(-delta_t/t_pot);
	}
	else
	{
		delta_s = -a_dep*exp2l(delta_t/t_dep);
	}
	
	delta_s -= forgetting;
	
	if(delta_s>=0)
	{
		*delta_pot = delta_s;
		*delta_dep = 0.0;
	}
	else
	{
		*delta_pot = 0.0;
		*delta_dep = delta_s;
	}
}


/*
* Plasticity function for Hebbian STDP asymmetric (causal)
*
* @param	t_i				time last spike of neuron i (in s)
* @param	t_j				time last spike of neuron j (in s)
* @param	delta_pot		potentiation brought by the plasticity function	
* @param	delta_dep		depression brought by the plasticity function	
* @param	forgetting		forgetting constant function added to the plasticity
*/ 
void hebbian_STDP_asymmetric_2(long double t_i, long double t_j, long double *delta_pot, long double *delta_dep, float forgetting)
{
	long double delta_s, delta_t = t_i-t_j;
	float t_pot = 0.02; 			//time window potentiation in s (~0 to 20 ms)
	float t_dep = 0.05; 			//time window depression in s (~0 to 20–100 ms)
	float a_pot = 5.296;			//amplitude potentiation
	float a_dep = 2.949;			//amplitude depression
	
	if(delta_t>=0)
	{
		delta_s = a_pot*exp2l(-delta_t/t_pot) - a_dep*exp2l(-4.0*delta_t/t_pot);
	}
	else
	{
		delta_s =  a_pot*exp2l(4.0*delta_t/t_dep)-a_dep*exp2l(delta_t/t_dep);
	}
	
	delta_s -= forgetting;
	
	if(delta_s>=0)
	{
		*delta_pot = delta_s;
		*delta_dep = 0.0;
	}
	else
	{
		*delta_pot = 0.0;
		*delta_dep = delta_s;
	}
}


/*
* Plasticity function for anti-Hebbian STDP asymmetric (acausal)
*
* @param	t_i				time last spike of neuron i (in s)
* @param	t_j				time last spike of neuron j (in s)
* @param	delta_pot		potentiation brought by the plasticity function	
* @param	delta_dep		depression brought by the plasticity function	
* @param	forgetting		forgetting constant function added to the plasticity
*/ 
void anti_hebbian_STDP_asymmetric(long double t_i, long double t_j, long double *delta_pot, long double *delta_dep, float forgetting)
{
	long double delta_s, delta_t = t_i-t_j;
	float t_pot = 0.05; 		//time window potentiation in s (~0 to 20 ms)
	float t_dep = 0.02; 		//time window depression in s (~0 to 20–100 ms)
	float a_pot = 1.0;			//amplitude potentiation
	float a_dep = 3.0;			//amplitude depression
	
	if(delta_t>=0)
	{
		delta_s = -a_dep*exp2l(-delta_t/t_dep);
	}
	else
	{
		delta_s = a_pot*exp2l(delta_t/t_pot);
	}
	
	delta_s += forgetting;
	
	if(delta_s>=0)
	{
		*delta_pot = delta_s;
		*delta_dep = 0.0;
	}
	else
	{
		*delta_pot = 0.0;
		*delta_dep = delta_s;
	}
}


/*
* Plasticity function for Hebbian STDP symmetric (correlated)(Mexican hat/Ricker wavelet function)
*
* @param	t_i				time last spike of neuron i (in s)
* @param	t_j				time last spike of neuron j (in s)
* @param	delta_pot		potentiation brought by the plasticity function	
* @param	delta_dep		depression brought by the plasticity function
* @param	forgetting		forgetting constant function added to the plasticity	
*/ 
void hebbian_STDP_symmetric(long double t_i, long double t_j, long double *delta_pot, long double *delta_dep, float forgetting)
{
	long double delta_s, delta_t = t_i-t_j;	
	float t_scale = 0.1; 		//time window potentiation and depression in s (-40,40 ms potentiation -100,100 ms depression) 
	float a = 3.0;				//amplitude		//1.371--> a=0.5
	
	delta_s = a * (1.0 - pow(delta_t/t_scale, 2.0)) * exp(-pow(delta_t, 2.0) / (2.0*pow(t_scale, 2.0)));
	
	delta_s -= forgetting;
	
	if(delta_s>=0)
	{
		*delta_pot = delta_s;
		*delta_dep = 0.0;
	}
	else
	{
		*delta_pot = 0.0;
		*delta_dep = delta_s;
	}
}


/*
* Plasticity function for anti-Hebbian STDP symmetric (uncorrelated)(reverse Mexican hat/Ricker wavelet function)
*
* @param	t_i				time last spike of neuron i (in s)
* @param	t_j				time last spike of neuron j (in s)
* @param	delta_pot		potentiation brought by the plasticity function	
* @param	delta_dep		depression brought by the plasticity function
* @param	forgetting		forgetting constant function added to the plasticity	
*/ 
void anti_hebbian_STDP_symmetric(long double t_i, long double t_j, long double *delta_pot, long double *delta_dep, float forgetting)
{	
	long double delta_s, delta_t = t_i-t_j;	
	float t_scale = 0.1; 		//time window potentiation and depression in s (-40,40 ms potentiation -100,100 ms depression) 
	float a = 3.0;				//amplitude //1.371--> a=0.5
	
	delta_s = -a * (1.0 - pow(delta_t/t_scale, 2.0)) * exp(-pow(delta_t, 2.0) / (2.0*pow(t_scale, 2.0)));
	
	delta_s += forgetting;
	
	if(delta_s>=0)
	{
		*delta_pot = delta_s;
		*delta_dep = 0.0;
	}
	else
	{
		*delta_pot = 0.0;
		*delta_dep = delta_s;
	}
}


/*
* Runge-kutta method 4th order for the weight (w)
*
* @param	w_i_j			the couping value for link from node j and to i
* @param	t_i				time last spike of neuron i (in s)
* @param	t_j				time last spike of neuron j (in s)
* @param	dt				the stepsize of the RK method
* @param	epsilon1		the dynamic of the neurons for slow adaptation
* @param	epsilon2		the dynamic of the neurons for fast adaptation
* @param	type_neuron_j	type of the neuron j
*/ 
long double RK_w(long double w_i_j, long double t_i, long double t_j, float dt, float epsilon1, float epsilon2, int type_neuron_j, int type_neuron_i)
{   
	long double w1, w2, w3, w4;
	long double delta_pot = 0.0;
	long double delta_dep = 0.0;
	float forgetting = 0.1;	//0.1 0.05 0.04 0.025 0.02 0.0125 0.01
	
	if(type_neuron_j==0) 			//If the presynaptic neuron j is excitatory
	{
		hebbian_STDP_asymmetric_2(t_i, t_j, &delta_pot, &delta_dep, forgetting);
		
		if(type_neuron_i==0) 			//If the postsynaptic neuron i is excitatory
		{
			w1 = epsilon2 * (-tanhl(100.0*(w_i_j-1.0)) * delta_pot + tanhl(100.0*(w_i_j)) * delta_dep);
	
			w2 = epsilon2 * (-tanhl(100.0*((w_i_j+w1*dt/2.0)-1.0)) * delta_pot + tanhl(100.0*((w_i_j+w1*dt/2.0))) * delta_dep);
	
			w3 = epsilon2 * (-tanhl(100.0*((w_i_j+w2*dt/2.0)-1.0)) * delta_pot + tanhl(100.0*((w_i_j+w2*dt/2.0))) * delta_dep);
	
			w4 = epsilon2 * (-tanhl(100.0*((w_i_j+w3*dt)-1.0)) * delta_pot + tanhl(100.0*((w_i_j+w3*dt))) * delta_dep);
		}
		else
		{
			w1 = epsilon1 * (-tanhl(100.0*(w_i_j-1.0)) * delta_pot + tanhl(100.0*(w_i_j)) * delta_dep);
	
			w2 = epsilon1 * (-tanhl(100.0*((w_i_j+w1*dt/2.0)-1.0)) * delta_pot + tanhl(100.0*((w_i_j+w1*dt/2.0))) * delta_dep);
	
			w3 = epsilon1 * (-tanhl(100.0*((w_i_j+w2*dt/2.0)-1.0)) * delta_pot + tanhl(100.0*((w_i_j+w2*dt/2.0))) * delta_dep);
	
			w4 = epsilon1 * (-tanhl(100.0*((w_i_j+w3*dt)-1.0)) * delta_pot + tanhl(100.0*((w_i_j+w3*dt))) * delta_dep);
		}	
	}
	else if(type_neuron_j==1) 		//If the presynaptic neuron j is hebbian inhibitory
	{
		hebbian_STDP_symmetric(t_i, t_j, &delta_pot, &delta_dep, forgetting);
		
		w1 = epsilon1 * (-tanhl(100.0*(w_i_j)) * delta_pot + tanhl(100.0*(w_i_j+1.0)) * delta_dep);
	
		w2 = epsilon1 * (-tanhl(100.0*((w_i_j+w1*dt/2.0))) * delta_pot + tanhl(100.0*((w_i_j+w1*dt/2.0)+1.0)) * delta_dep);
	
		w3 = epsilon1 * (-tanhl(100.0*((w_i_j+w2*dt/2.0))) * delta_pot + tanhl(100.0*((w_i_j+w2*dt/2.0)+1.0)) * delta_dep);
	
		w4 = epsilon1 * (-tanhl(100.0*((w_i_j+w3*dt))) * delta_pot + tanhl(100.0*((w_i_j+w3*dt)+1.0)) * delta_dep);
	} 
	else							//If the presynaptic neuron j is anti-hebbian inhibitory
	{
		anti_hebbian_STDP_symmetric(t_i, t_j, &delta_pot, &delta_dep, forgetting);	
		
		w1 = epsilon1 * (-tanhl(100.0*(w_i_j)) * delta_pot + tanhl(100.0*(w_i_j+1.0)) * delta_dep);
	
		w2 = epsilon1 * (-tanhl(100.0*((w_i_j+w1*dt/2.0))) * delta_pot + tanhl(100.0*((w_i_j+w1*dt/2.0)+1.0)) * delta_dep);
	
		w3 = epsilon1 * (-tanhl(100.0*((w_i_j+w2*dt/2.0))) * delta_pot + tanhl(100.0*((w_i_j+w2*dt/2.0)+1.0)) * delta_dep);
	
		w4 = epsilon1 * (-tanhl(100.0*((w_i_j+w3*dt))) * delta_pot + tanhl(100.0*((w_i_j+w3*dt)+1.0)) * delta_dep);
	}
	
	
    return (w_i_j + dt*((w1 + 2.0*w2 + 2.0*w3 + w4)/6.0));
}


/*
* Update coupling weights of the network
*
* @param	neurons			pointer on the current network
* @param	spikes			if neurons spike
*/  
void update_weights(struct neurons *neurons, int *spikes) 
{
    int i, j;
	//for each neurons of the network
	for(i=0; i < neurons->n; i++)
    {
		for(j=0; j < neurons->n; j++)
	    {
			if((neurons->a[i][j]==1) && (spikes[i] || spikes[j])) //if the link between neurons j to i exists and one of the neuron spikes
			{
				neurons->w[i][j] = RK_w(neurons->w[i][j], neurons->t_spikes[i], neurons->t_spikes[j], neurons->dt, neurons->epsilon1, neurons->epsilon2, neurons->type_neuron[j], neurons->type_neuron[i]);		
			}
		}
	}
}



/*
* Change the inputs of the network
*
* @param	neurons			pointer on the current network
* @param	inputs			new inputs
* @param	n					number of inputs to change
*/   
void set_inputs(struct neurons *neurons, double *inputs, int n)
{
    int i;
	//for each neurons of the network
	for(i=0; i < n; i++)
    {
		neurons->inputs[i] = inputs[i];
	}
}



/*
* Reset exitatory weights with random values
*
* @param	neurons			pointer on the current network
*/ 
void reset_excitatory_weights_random(struct neurons *neurons)
{
    int i, j;
	//for each neurons of the network
	for(i=0; i < neurons->n; i++)
    {
		for(j=0; j < neurons->n; j++)
	    {
			if((neurons->type_neuron[j]==0) && (neurons->type_neuron[i]==0))
			{
				neurons->w[i][j] = get_random(0.0001, 0.9999);
			}
		}
	}
}



/*
* Reset exitatory weights with null values
*
* @param	neurons			pointer on the current network
*/ 
void reset_excitatory_weights_null(struct neurons *neurons)
{
    int i, j;
	//for each neurons of the network
	for(i=0; i < neurons->n; i++)
    {
		for(j=0; j < neurons->n; j++)
	    {
			neurons->w[i][j] = 0.0;
		}
	}
}