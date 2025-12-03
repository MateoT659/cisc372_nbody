#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "vector.h"
#include "config.h"

vector3* d_values;
vector3** d_accels;

dim3 blockSizeGrid(16, 16);
dim3 nBlocksGrid(
	(NUMENTITIES + blockSizeGrid.x - 1) / blockSizeGrid.x,
	(NUMENTITIES + blockSizeGrid.y - 1) / blockSizeGrid.y
);

int blockSize = 256;
int nBlocks = (NUMENTITIES + blockSize - 1) / blockSize;

#define EC(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ void initAccels(vector3** accels, vector3* values) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < NUMENTITIES) {
		accels[i] = &values[i * NUMENTITIES];
	}
}

__global__ void pairwiseAccels(vector3** accels, vector3* hPos, double* mass) {
	int i, j, k;
	
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i >= NUMENTITIES || j >= NUMENTITIES) return;

	if (i == j) {
		FILL_VECTOR(accels[i][j], 0, 0, 0);
	}
	else {
		vector3 distance;
		for (k = 0; k < 3; k++) distance[k] = hPos[i][k] - hPos[j][k];
		double magnitude_sq = distance[0] * distance[0] + distance[1] * distance[1] + distance[2] * distance[2];
		double magnitude = sqrt(magnitude_sq);
		double accelmag = -1 * GRAV_CONSTANT * mass[j] / magnitude_sq;
		FILL_VECTOR(accels[i][j], accelmag * distance[0] / magnitude, accelmag * distance[1] / magnitude, accelmag * distance[2] / magnitude);
	}
}

__device__ void TreeSum(vector3* values, int thx) {
	int stride, k;

	for(stride = 1; stride<blockDim.x; stride <<=1) {
		__syncthreads();
		if ((thx % (stride<<1)) == 0 && thx + stride < NUMENTITIES) {
			for (k = 0; k < 3; k++) {
				values[thx][k] += values[thx + stride][k];
			}
		}
	}
}

__global__ void accelSums(vector3** accels, vector3* hPos, vector3* hVel) {
	int i,j, k;

	i = blockIdx.y;
	j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= NUMENTITIES || j >= NUMENTITIES) return;

	//tree sum
	TreeSum(accels[i], j);

	vector3 accelSum = { 0,0,0 };
	/*if (j % blockDim.x == 0) {
		for (k = 0; k < 3; k++) {
			atomicAdd(&accelSum[k], accels[i][j][k]);
		}
	}*/
	
	//compute the new velocity based on the acceleration and time interval
	if (j == 0) {
		for (k = 0; k < 3; k++) {
			hVel[i][k] += accelSum[k] * INTERVAL;
			hPos[i][k] += hVel[i][k] * INTERVAL;
		}
	}
}

extern "C" void compute() {
	//make an acceleration matrix which is NUMENTITIES squared in size;
	pairwiseAccels<<<nBlocksGrid, blockSizeGrid>>>(d_accels, d_hPos, d_mass);
	EC(cudaDeviceSynchronize());

	int accelSumsBlockSize = 256;
	dim3 accelSumsNBlocks(
		(NUMENTITIES + accelSumsBlockSize - 1) / accelSumsBlockSize,
		NUMENTITIES
	);

	accelSums<<<accelSumsNBlocks, accelSumsBlockSize>>>(d_accels, d_hPos, d_hVel);
	EC(cudaDeviceSynchronize());
}

extern "C" void initDeviceMemory(int numEntities) {
	EC(cudaMalloc(&d_hPos, sizeof(vector3) * NUMENTITIES));
	EC(cudaMalloc(&d_hVel, sizeof(vector3) * NUMENTITIES));
	EC(cudaMalloc(&d_mass, sizeof(double) * NUMENTITIES));

	EC(cudaMemcpy(d_hPos, hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice));
	EC(cudaMemcpy(d_hVel, hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice));
	EC(cudaMemcpy(d_mass, mass, sizeof(double) * NUMENTITIES, cudaMemcpyHostToDevice));

	EC(cudaMalloc(&d_values, sizeof(vector3) * NUMENTITIES * NUMENTITIES));
	EC(cudaMalloc(&d_accels, sizeof(vector3*) * NUMENTITIES));

	initAccels<<<nBlocksGrid, blockSizeGrid>>>(d_accels, d_values);
	EC(cudaDeviceSynchronize());
}

extern "C" void freeDeviceMemory(int numEntities) {
	EC(cudaFree(d_values));
	EC(cudaFree(d_accels));

	EC(cudaMemcpy(hPos, d_hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost));
	EC(cudaMemcpy(hVel, d_hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost));
	EC(cudaMemcpy(mass, d_mass, sizeof(double) * NUMENTITIES, cudaMemcpyDeviceToHost));

	EC(cudaFree(d_hPos));
	EC(cudaFree(d_hVel));
	EC(cudaFree(d_mass));
}