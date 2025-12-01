#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "vector.h"
#include "config.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stdout, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

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
	if (i < NUMENTITIES && j < NUMENTITIES) {
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

__global__
void sumMatrices(vector3** accels, vector3* hVel, vector3* hPos) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < NUMENTITIES) {
		vector3 accel_sum = { 0,0,0 };
		for (int j = 0; j < NUMENTITIES; j++) {
			for (int k = 0; k < 3; k++)
				accel_sum[k] += accels[i][j][k];
		}
		//compute the new velocity based on the acceleration and time interval
		//compute the new position based on the velocity and time interval
		for (int k = 0; k < 3; k++) {
			hVel[i][k] += accel_sum[k] * INTERVAL;
			hPos[i][k] += hVel[i][k] * INTERVAL;
		}
	}
}

extern "C" void compute() {
	//make an acceleration matrix which is NUMENTITIES squared in size;
	
	vector3* d_values;
	vector3** d_accels;

	gpuErrchk(cudaMalloc(&d_values, sizeof(vector3) * NUMENTITIES * NUMENTITIES));
	gpuErrchk(cudaMalloc(&d_accels, sizeof(vector3) * NUMENTITIES));

	int threads_per_block_init = NUMENTITIES;

	initAccels << <1, threads_per_block_init >> > (d_accels, d_values);
	gpuErrchk(cudaDeviceSynchronize())
	
	gpuErrchk(cudaMemcpy(d_hPos, hPos, NUMENTITIES * sizeof(vector3), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_hVel, hVel, NUMENTITIES * sizeof(vector3), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_mass, mass, NUMENTITIES * sizeof(double), cudaMemcpyHostToDevice));


	dim3 threads_per_block_PWA(16, 16);
	dim3 n_blocks_PWA((NUMENTITIES + threads_per_block_PWA.x - 1) / threads_per_block_PWA.x, (NUMENTITIES + threads_per_block_PWA.y - 1) / threads_per_block_PWA.y);

	pairwiseAccels<<<n_blocks_PWA, threads_per_block_PWA >>>(d_accels, d_hPos, d_mass);
	gpuErrchk(cudaDeviceSynchronize());
	
	
	//sum up the rows of our matrix to get effect on each entity, then update velocity and position.
	int threads_per_block_update = 256;
	int n_blocks_update = (NUMENTITIES + threads_per_block_update - 1) / threads_per_block_update;

	sumMatrices<<<n_blocks_update, threads_per_block_update>>>(d_accels, d_hVel, d_hPos);
	gpuErrchk(cudaDeviceSynchronize());

	gpuErrchk(cudaMemcpy(hPos, d_hPos, NUMENTITIES * sizeof(vector3), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(hVel, d_hVel , NUMENTITIES * sizeof(vector3), cudaMemcpyDeviceToHost));
	
	cudaFree(d_values);
	cudaFree(d_accels);
	cudaFree(d_hPos);
	cudaFree(d_hVel);
	cudaFree(d_mass);
}