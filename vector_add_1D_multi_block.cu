#include<stdio.h>
#include<stdlib.h>

__global__ void arradd(int* md, int* nd, int* pd, int size)
{
	//Get unique identification number for a given thread
	int myid = blockIdx.x*blockDim.x + threadIdx.x;
	
	pd[myid] = md[myid] + nd[myid];
}


int main()
{
	int size = 2000 * sizeof(int);
	int m[2000], n[2000], p[2000],*md, *nd,*pd;
	int i=0;

	//Initialize the arrays
	for(i=0; i<2000; i++ )
	{
		m[i] = i;
		n[i] = i;
		p[i] = 0;
	}

	// Allocate memory on GPU and transfer the data
	cudaMalloc(&md, size);
	cudaMemcpy(md, m, size, cudaMemcpyHostToDevice);

	cudaMalloc(&nd, size);
	cudaMemcpy(nd, n, size, cudaMemcpyHostToDevice);

	cudaMalloc(&pd, size);

	// Define number of threads and blocks
	dim3   DimGrid(10, 1);     
	dim3   DimBlock(200, 1);   

	// Launch the GPU kernel function
	arradd<<< DimGrid,DimBlock >>>(md,nd,pd,size);

	// Transfer the results back to CPU memory
	cudaMemcpy(p, pd, size, cudaMemcpyDeviceToHost);
	
	// Free GPU arrays
	cudaFree(md); 
	cudaFree(nd);
	cudaFree (pd);

	// Print the results
	for(i=0; i<2000; i++ )
	{
		printf("\t%d",p[i]);
	}
}
