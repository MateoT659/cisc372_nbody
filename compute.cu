#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"

//serial version
__global__
void initAccels(vector3** accels, vector3* values) {
	int i = threadIdx.x;
	accels[i] = &values[i * NUMENTITIES];
}

__global__
void pairwiseAccels(vector3** accels, vector3* hPos, double* mass) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if (i == j) {
		FILL_VECTOR(accels[i][j], 0, 0, 0);
	}
	else {
		vector3 distance;
		for (int k = 0; k < 3; k++) distance[k] = hPos[i][k] - hPos[j][k];
		double magnitude_sq = distance[0] * distance[0] + distance[1] * distance[1] + distance[2] * distance[2];
		double magnitude = sqrt(magnitude_sq);
		double accelmag = -1 * GRAV_CONSTANT * mass[j] / magnitude_sq;
		FILL_VECTOR(accels[i][j], accelmag * distance[0] / magnitude, accelmag * distance[1] / magnitude, accelmag * distance[2] / magnitude);
	}
}

extern "C" void compute() {
	//make an acceleration matrix which is NUMENTITIES squared in size;
	int i, j, k;
	
	// --- OLD CODE ---
	vector3* values = (vector3*)malloc(sizeof(vector3) * NUMENTITIES * NUMENTITIES);
	vector3** accels = (vector3**)malloc(sizeof(vector3*) * NUMENTITIES);

	for (i = 0; i < NUMENTITIES; i++)
		accels[i] = &values[i * NUMENTITIES];

	// --- NEW CODE ---
	vector3* d_values;
	vector3** d_accels;

	cudaMalloc(&d_values, sizeof(vector3) * NUMENTITIES * NUMENTITIES);
	cudaMalloc(&d_accels, sizeof(vector3) * NUMENTITIES);

	int threads_per_block_init = NUMENTITIES;

	initAccels << <1, threads_per_block_init >> > (d_accels, d_values);
	
	
	cudaMemcpy(d_hPos, hPos, NUMENTITIES * sizeof(vector3), cudaMemcpyHostToDevice);
	cudaMemcpy(d_hVel, hVel, NUMENTITIES * sizeof(vector3), cudaMemcpyHostToDevice);
	cudaMemcpy(d_mass, mass, NUMENTITIES * sizeof(double), cudaMemcpyHostToDevice);


	dim3 threads_per_block_PWA(16, 16);
	dim3 n_blocks_PWA((NUMENTITIES + threads_per_block_PWA.x - 1) / threads_per_block_PWA.x, (NUMENTITIES + threads_per_block_PWA.y - 1) / threads_per_block_PWA.y);

	pairwiseAccels<<<n_blocks_PWA, threads_per_block_PWA >>>(d_accels, d_hPos, d_mass);
	
	
	
	//sum up the rows of our matrix to get effect on each entity, then update velocity and position.
	for (i = 0; i < NUMENTITIES; i++) {
		vector3 accel_sum = { 0,0,0 };
		for (j = 0; j < NUMENTITIES; j++) {
			for (k = 0; k < 3; k++)
				accel_sum[k] += accels[i][j][k];
		}
		//compute the new velocity based on the acceleration and time interval
		//compute the new position based on the velocity and time interval
		for (k = 0; k < 3; k++) {
			hVel[i][k] += accel_sum[k] * INTERVAL;
			hPos[i][k] += hVel[i][k] * INTERVAL;
		}
	}
	
	cudaFree(d_values);
	cudaFree(d_accels);
	cudaFree(d_hPos);
	cudaFree(d_hVel);
	cudaFree(d_mass);
	
	free(accels);
	free(values);
}