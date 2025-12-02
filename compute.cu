#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"

__global__
void initAccels(vector3** accels, vector3* values) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < NUMENTITIES) {
		accels[i] = &values[i * NUMENTITIES];
	}
}

__global__
void pairwiseAccels(vector3** accels, vector3* hPos, double* mass) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j;
	if (i < NUMENTITIES) {
		for (j = 0; j < NUMENTITIES; j++) {
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
	}
}
//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
extern "C" void compute() {
	//make an acceleration matrix which is NUMENTITIES squared in size;
	int i, j, k;

	vector3* d_values;
	vector3** d_accels;

	cudaMalloc(&d_values, sizeof(vector3)* NUMENTITIES* NUMENTITIES);
	cudaMalloc(&d_accels, sizeof(vector3*) * NUMENTITIES);

	int blockSize = 256;
	int numBlocks = (NUMENTITIES + blockSize - 1) / blockSize;
	initAccels<<<numBlocks, blockSize >> > (d_accels, d_values);


	cudaMalloc(&d_hPos, sizeof(vector3*) * NUMENTITIES);
	cudaMalloc(&d_hVel, sizeof(vector3*) * NUMENTITIES);
	cudaMalloc(&d_mass, sizeof(double) * NUMENTITIES);

	cudaMemcpy(&hPos, d_hPos, sizeof(vector3*) * NUMENTITIES, cudaMemcpyDeviceToHost);
	cudaMemcpy(&hVel, d_hVel, sizeof(vector3*) * NUMENTITIES, cudaMemcpyDeviceToHost);
	cudaMemcpy(&mass, d_mass, sizeof(double) * NUMENTITIES, cudaMemcpyDeviceToHost);

	vector3* values = (vector3*)malloc(sizeof(vector3) * NUMENTITIES * NUMENTITIES);
	vector3** accels = (vector3**)malloc(sizeof(vector3*) * NUMENTITIES);
	
	for (i = 0; i < NUMENTITIES; i++)
		accels[i] = &values[i * NUMENTITIES];



	blockSize = 256;
	numBlocks = (NUMENTITIES + blockSize - 1) / blockSize;

	pairwiseAccels<<<numBlocks, blockSize>>>(d_accels, d_hPos, d_mass);

	//first compute the pairwise accelerations.  Effect is on the first argument.
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

	free(accels);
	free(values);

	cudaFree(d_accels);
	cudaFree(d_values);

	cudaMemcpy(d_hPos, &hPos, sizeof(vector3*) * NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_hVel, &hVel, sizeof(vector3*) * NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_mass, &mass, sizeof(double) * NUMENTITIES, cudaMemcpyHostToDevice);

	cudaFree(d_hPos);
	cudaFree(d_hVel);
	cudaFree(d_mass);
}