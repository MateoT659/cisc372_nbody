#include <stdlib.h>
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

//for accel sums, each block is entirely in one row to do a local tree sum
int accelSumsBlockSize = 256;
dim3 accelSumsNBlocks(
	(NUMENTITIES + accelSumsBlockSize - 1) / accelSumsBlockSize,
	NUMENTITIES
);

int blockSize = 256;
int nBlocks = (NUMENTITIES + blockSize - 1) / blockSize;

__global__ void initAccels(vector3** accels, vector3* values) {
	//thread controls initializing one row of accels
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < NUMENTITIES) {
		accels[i] = &values[i * NUMENTITIES];
	}
}

//update diags each time because they are used as places to store the sum to save a bit of memory
__global__ void pairwiseAccelsDiag(vector3** accels, vector3* hPos, double* mass) {
	//thread i sets the ith diagonal to 0
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= NUMENTITIES) return;

	FILL_VECTOR(accels[i][i], 0, 0, 0);
}

__global__ void pairwiseAccels(vector3** accels, vector3* hPos, double* mass) {
	//mostly unchanged code, basic parallelization using grid. 
	//i took out the diagonal conditional and put it in another kernel so warps perform better
	int i, j, k;
	
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i >= NUMENTITIES || j >= NUMENTITIES || i == j) return;

	vector3 distance;
	for (k = 0; k < 3; k++) distance[k] = hPos[i][k] - hPos[j][k];
	double magnitude_sq = distance[0] * distance[0] + distance[1] * distance[1] + distance[2] * distance[2];
	double magnitude = sqrt(magnitude_sq);
	double accelmag = -1 * GRAV_CONSTANT * mass[j] / magnitude_sq;
	FILL_VECTOR(accels[i][j], accelmag * distance[0] / magnitude, accelmag * distance[1] / magnitude, accelmag * distance[2] / magnitude);
}

__device__ void TreeSum(vector3* values, int sharedIndex) {
	//tree sum reduction for the device
	int stride, k;

	for(stride = 1; stride < blockDim.x; stride <<= 1) {
		if ((sharedIndex % (stride<<1)) == 0) {
			for (k = 0; k < 3; k++) {
				values[sharedIndex][k] += values[sharedIndex + stride][k];
			}
		}
		__syncthreads();
	}
}


__global__ void accelSums(vector3** accels, vector3* hPos, vector3* hVel) {
	//each block is in one row, tree sum every index the block controls then add to the diagonal of that row
	int row, col, k, sharedIndex;

	row = blockIdx.y;
	col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row >= NUMENTITIES) return;

	__shared__ vector3 sharedAccels[256];
	sharedIndex = threadIdx.x;

	for (k = 0; k < 3; k++) {
		//row == col check is important - if a block atomic adds to accels[row][row] before the block containing
		//accels[row][row] does the tree sum, it will be twice added. fix: remove from shared array entirely
		sharedAccels[sharedIndex][k] = (col >= NUMENTITIES || row == col) ? 0.0 : accels[row][col][k];
	}
	__syncthreads();

	TreeSum(sharedAccels, sharedIndex);

	if (sharedIndex == 0) {
		for (k = 0; k < 3; k++) {
			atomicAdd(&accels[row][row][k], sharedAccels[0][k]); //note diag is 0
		}
	}
}

__global__ void updateVelPos(vector3 * *accels, vector3 * hPos, vector3 * hVel) {
	//each thread updates vel and pos using a diagonal (the sum) 
	int i, k; 
	
	i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= NUMENTITIES) return;

	for (k = 0; k < 3; k++) {
		hVel[i][k] += accels[i][i][k] * INTERVAL;
		hPos[i][k] += hVel[i][k] * INTERVAL;
	}
}

extern "C" void compute() {
	//initialize the diagonals of the accels matrix to 0 (they're used for summing later)
	pairwiseAccelsDiag<<<nBlocks, blockSize>>>(d_accels, d_hPos, d_mass);

	//compute the accelerations on the non-diagonals of accels
	pairwiseAccels<<<nBlocksGrid, blockSizeGrid>>>(d_accels, d_hPos, d_mass);

	//sum up using reduction and store each row's sum in accels[i][i]
	accelSums<<<accelSumsNBlocks, accelSumsBlockSize>>>(d_accels, d_hPos, d_hVel);

	//update hvel and hpos to avoid a race condition
	updateVelPos<<<nBlocks, blockSize>>>(d_accels, d_hPos, d_hVel);
}

extern "C" void initDeviceMemory(int numEntities) {
	//init device pos, vel, mass once
	cudaMalloc(&d_hPos, sizeof(vector3) * NUMENTITIES);
	cudaMalloc(&d_hVel, sizeof(vector3) * NUMENTITIES);
	cudaMalloc(&d_mass, sizeof(double) * NUMENTITIES);

	//copy random values
	cudaMemcpy(d_hPos, hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_hVel, hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_mass, mass, sizeof(double) * NUMENTITIES, cudaMemcpyHostToDevice);

	//init accels and values here so they don't have to be re-malloced every time
	cudaMalloc(&d_values, sizeof(vector3) * NUMENTITIES * NUMENTITIES);
	cudaMalloc(&d_accels, sizeof(vector3*) * NUMENTITIES);

	//initializing 2d array
	initAccels<<<nBlocksGrid, blockSizeGrid>>>(d_accels, d_values);
}

extern "C" void freeDeviceMemory(int numEntities) {
	cudaFree(d_values);
	cudaFree(d_accels);

	cudaMemcpy(hPos, d_hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
	cudaMemcpy(hVel, d_hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
	cudaMemcpy(mass, d_mass, sizeof(double) * NUMENTITIES, cudaMemcpyDeviceToHost);

	cudaFree(d_hPos);
	cudaFree(d_hVel);
	cudaFree(d_mass);
}