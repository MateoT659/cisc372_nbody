#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "vector.h"
#include "config.h"

//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL

__global__ void initAccels(vector3** accels, vector3* values) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < NUMENTITIES) {
		accels[i] = &values[i * NUMENTITIES];
	}
}

__global__ void pairwiseAccels(vector3** accels, vector3* hPos, double* mass) {
	int i, j, k;
	for (i = 0; i < NUMENTITIES; i++) {
		for (j = 0; j < NUMENTITIES; j++) {
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
	}
}

__global__ void accelSums(vector3** accels, vector3* hPos, vector3* hVel) {
	int i, j, k;
	for (i = 0; i < NUMENTITIES; i++) {
		vector3 accel_sum = { 0,0,0 };
		for (j = 0; j < NUMENTITIES; j++) {
			for (k = 0; k < 3; k++)
				accel_sum[k] += accels[i][j][k];
		}
		//compute the new velocity based on the acceleration and time interval
		for (k = 0; k < 3; k++) {
			hVel[i][k] += accel_sum[k] * INTERVAL;
		}
	}
}

extern "C" void compute() {
	//make an acceleration matrix which is NUMENTITIES squared in size;
	vector3* d_values;
	cudaMalloc(&d_values, sizeof(vector3) * NUMENTITIES * NUMENTITIES);

	vector3** d_accels;
	cudaMalloc(&d_accels, sizeof(vector3*) * NUMENTITIES);

	cudaMalloc(&d_hPos, sizeof(vector3) * NUMENTITIES);
	cudaMemcpy(d_hPos, hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);

	cudaMalloc(&d_hVel, sizeof(vector3) * NUMENTITIES);
	cudaMemcpy(d_hVel, hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);
	
	cudaMalloc(&d_mass, sizeof(double) * NUMENTITIES);
	cudaMemcpy(d_mass, mass, sizeof(double) * NUMENTITIES, cudaMemcpyHostToDevice);

	int blockSize = 256;
	int nBlocks = (NUMENTITIES + blockSize - 1) / blockSize;
	initAccels<<<nBlocks, blockSize>>>(d_accels, d_values);

	pairwiseAccels<<<1, 1>>>(d_accels, d_hPos, d_mass);

	accelSums<<<1, 1>>>(d_accels, d_hPos, d_hVel);

	cudaFree(d_values);
	cudaFree(d_accels);

	cudaMemcpy(hPos, d_hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
	cudaMemcpy(hVel, d_hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
	cudaMemcpy(mass, d_mass, sizeof(double) * NUMENTITIES, cudaMemcpyDeviceToHost);

	cudaFree(d_hPos);
	cudaFree(d_hVel);
	cudaFree(d_mass);
}