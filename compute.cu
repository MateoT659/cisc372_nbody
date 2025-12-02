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
		fprintf(stderr, "GPUassert: %s (%s) %d\n", cudaGetErrorName(code), cudaGetErrorString(code), line);
		fflush(stderr);
		if (abort) exit((int)code);
	}
}
#define gpuCheckLastError() gpuErrchk(cudaGetLastError())

// serial version: set up device pointer array, safe bounds
__global__
void initAccels(vector3** accels, vector3* values, int n) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < n) {
		accels[i] = &values[(size_t)i * (size_t)n];
	}
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
			const double eps = 1e-12;
			if (magnitude_sq <= eps) {
				FILL_VECTOR(accels[i][j], 0, 0, 0);
			} else {
				double magnitude = sqrt(magnitude_sq);
				double accelmag = -1.0 * GRAV_CONSTANT * mass[j] / magnitude_sq;
				FILL_VECTOR(accels[i][j], accelmag * distance[0] / magnitude, accelmag * distance[1] / magnitude, accelmag * distance[2] / magnitude);
			}
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
		for (int k = 0; k < 3; k++) {
			hVel[i][k] += accel_sum[k] * INTERVAL;
			hPos[i][k] += hVel[i][k] * INTERVAL;
		}
	}
}

extern "C" void compute() {
	// allocate local device storage for this compute invocation
	vector3* d_values = NULL;
	vector3** d_accels = NULL;

	// allocate NxN vector storage (contiguous)
	gpuErrchk(cudaMalloc(&d_values, sizeof(vector3) * (size_t)NUMENTITIES * (size_t)NUMENTITIES));
	// allocate array of device pointers (vector3*)
	gpuErrchk(cudaMalloc(&d_accels, sizeof(vector3*) * (size_t)NUMENTITIES));

	// init pointer array on device from host (safe pattern)
	int threads_per_block_init = 256;
	int n_blocks_init = (NUMENTITIES + threads_per_block_init - 1) / threads_per_block_init;
	initAccels << <n_blocks_init, threads_per_block_init >> > (d_accels, d_values, NUMENTITIES);
	gpuCheckLastError();
	gpuErrchk(cudaDeviceSynchronize());

	// copy inputs (ensure these globals were allocated previously by initDeviceMemory)
	gpuErrchk(cudaMemcpy(d_hPos, hPos, NUMENTITIES * sizeof(vector3), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_hVel, hVel, NUMENTITIES * sizeof(vector3), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_mass, mass, NUMENTITIES * sizeof(double), cudaMemcpyHostToDevice));

	dim3 threads_per_block_PWA(16, 16);
	dim3 n_blocks_PWA((NUMENTITIES + threads_per_block_PWA.x - 1) / threads_per_block_PWA.x,
		(NUMENTITIES + threads_per_block_PWA.y - 1) / threads_per_block_PWA.y);

	pairwiseAccels<<<n_blocks_PWA, threads_per_block_PWA>>>(d_accels, d_hPos, d_mass);
	gpuCheckLastError();
	gpuErrchk(cudaDeviceSynchronize());

	int threads_per_block_update = 256;
	int n_blocks_update = (NUMENTITIES + threads_per_block_update - 1) / threads_per_block_update;

	sumMatrices<<<n_blocks_update, threads_per_block_update>>>(d_accels, d_hVel, d_hPos);
	gpuCheckLastError();
	gpuErrchk(cudaDeviceSynchronize());

	gpuErrchk(cudaMemcpy(hPos, d_hPos, NUMENTITIES * sizeof(vector3), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(hVel, d_hVel, NUMENTITIES * sizeof(vector3), cudaMemcpyDeviceToHost));

	gpuErrchk(cudaFree(d_values));
	gpuErrchk(cudaFree(d_accels));
}

extern "C" void initDeviceMemory(int numObjects) {
	gpuErrchk(cudaMalloc(&d_hPos, numObjects * sizeof(vector3)));
	gpuErrchk(cudaMalloc(&d_hVel, numObjects * sizeof(vector3)));
	gpuErrchk(cudaMalloc(&d_mass, numObjects * sizeof(double)));

	gpuErrchk(cudaMemcpy(d_hPos, hPos, numObjects * sizeof(vector3), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_hVel, hVel, numObjects * sizeof(vector3), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_mass, mass, numObjects * sizeof(double), cudaMemcpyHostToDevice));
}

extern "C" void freeDeviceMemory(int numObjects) {

	gpuErrchk(cudaMemcpy(hPos, d_hPos, numObjects * sizeof(vector3), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(hVel, d_hVel, numObjects * sizeof(vector3), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(mass, d_mass, numObjects * sizeof(double), cudaMemcpyDeviceToHost));

	gpuErrchk(cudaFree(d_hPos));
	gpuErrchk(cudaFree(d_hVel));
	gpuErrchk(cudaFree(d_mass));
}