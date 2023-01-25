#include<stdio.h>
#include<stdlib.h>

/*
	CUDA - 'Hello world' Program. 
	This program adds two Vectors m, n 
	m + n = p 

	Learning objective: Introduction 
	* Creating Blocks
	* Creating Threads
	* Kernel Function
	* cudaMalloc
	* CudaMemcpy
	* Launch Kernel
	* threadIdx
	* cudaFree
*/


__global__ void arradd(int* md, int* nd, int* pd, int size)
{
	int myid = threadIdx.x;
	
	pd[myid] = md[myid] + nd[myid];
}


int main()
{
	int size = 200 * sizeof(int);
	int m[200], n[200], p[200],*md, *nd,*pd;
	int i=0;

	
	for(i=0; i<200; i++ )
	{
		m[i] = i;
		n[i] = i;
		p[i] = 0;
	}

	cudaMalloc(&md, size);
	cudaMemcpy(md, m, size, cudaMemcpyHostToDevice);

	cudaMalloc(&nd, size);
	cudaMemcpy(nd, n, size, cudaMemcpyHostToDevice);

	cudaMalloc(&pd, size);

	dim3   DimGrid(1, 1);     
	dim3   DimBlock(200, 1);   


	arradd<<< DimGrid,DimBlock >>>(md,nd,pd,size);

	cudaMemcpy(p, pd, size, cudaMemcpyDeviceToHost);
	cudaFree(md); 
	cudaFree(nd);
	cudaFree(pd);

	for(i=0; i<200; i++ )
	{
		printf("\t%d",p[i]);
	}	
}




