#include "graph.h"



/********************** Additional processing ************************/


/*
* Function to swap elements of array
*
* @param	xp		array 1
* @param	yp		array 2
*/
void swap(double* xp, double* yp) 
{ 
    double temp = *xp; 
    *xp = *yp; 
    *yp = temp; 
} 


/*
* Function to save a spike of neuron
*
* @param	i			index of the neuron
* @param	t_spike		time of the spike in second
*/
void saveSpike(int i, long double t_spike)
{
	/**** Save spikes  through the time ****/
	FILE *fptr = fopen("spikes.txt","a");
	
	if(fptr == NULL)
	{
		printf("Error!");   
		exit(1);             
	}
	
	fprintf(fptr,"%d %3.5Lf\n", i, t_spike);
	
	fclose(fptr);
}
 
 
/*
* Function to perform Selection Sort 
*
* @param	arr		array to sort
* @param	n		number of element
*/
void selectionSort(double arr[], int n) 
{ 
	int i, j, min_idx; 
  
    //One by one move boundary of unsorted subarray 
    for (i = 0; i < n - 1; i++) 
	{ 
		//Find the minimum element in unsorted array 
		min_idx = i; 
		for (j = i + 1; j < n; j++)
		{
			if (arr[j] < arr[min_idx])
			{ 
				min_idx = j;
			} 
  		} 
        //Swap the found minimum element 
        //with the first element 
        swap(&arr[min_idx], &arr[i]); 
    } 
}




/********************** Filled input vector ************************/


/*
* Change the input vector by null values
*
* @param	inputs			the input vector
* @param	n				the number of input in the vector
*/
void addNullInputs(double *inputs, int n)
{
	int i;
	
	//For all inputs
	for(i=0; i < n; i++)
	{
		inputs[i] = 0.0;
	}
}


/*
* Change the input vector by binary values on a specific cluster, stimulation brain 50-200hz
*
* @param	inputs			the input vector
* @param	n				the number of input in the vector
* @param	index			the index of the center position the cluster
* @param	nbInput			the number of input to activatee
* @param	tau_m			the membrane time constant of the target neurons
* @param	inputFrequency	the frequency of the external input applied
*/
void addBinaryLocalized(double *inputs, int n, int index, int nbInput, float tau_m, float inputFrequency)
{
	int i;
	
	//For all inputs
	for(i=0; i < n; i++)
	{
		//If we are on the cluster
		if((i>=(index-nbInput/2)) && (i<(index+nbInput/2)))
		{
			//inputFrequency = get_random(0, 100.0);
			
			inputs[i] = pow(inputFrequency*M_PI*tau_m, 2.0);	//inputFrequency Hz;
		}
		else
		{
			inputs[i] = 0.0;
		}
	}
}


/*
* Change the input vector by binary values on a specific cluster with random activation
*
* @param	inputs			the input vector
* @param	n				the number of input in the vector
* @param	index			the index of the center position the cluster
* @param	nbInput			the number of input to activate
* @param	tau_m			the membrane time constant of the target neurons
* @param	inputFrequency	the frequency of the external input applied
*/
void addBinaryLocalizedRandom(double *inputs, int n, int index, int nbInput, float tau_m, float inputFrequency)
{
	int i;
	//int ratio = (rand()%2) +1;
	
	//For all inputs
	for(i=0; i < n; i++)
	{
		//If we are on the cluster
		if((i>=(index-nbInput/2)) && (i<(index+nbInput/2)))
		{
			if((rand()%2)!=0)
			//if((rand()%ratio)==0)
			{
				inputs[i] = pow(inputFrequency*M_PI*tau_m, 2.0);
			}
			else
			{
				inputs[i] = 0.0;
			}
		}
		else
		{
			inputs[i] = 0.0;
		}
	}
}



