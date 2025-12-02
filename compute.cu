#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "vector.h"
#include "config.h"

vector3 *d_values;
vector3 **d_accels;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s (%d) %s %d\n", cudaGetErrorString(code), (int)code, file, line);
		if (abort) exit((int)code);
	}
}
#define gpuCheckLastError() gpuErrchk(cudaGetLastError())

// NOTE: initAccels kernel is no longer needed for correctness.
// Keeping it commented out for reference.
/*
__global__
void initAccels(vector3** accels, vector3* values, int n) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < n) {
		accels[i] = &values[i * n];
	}
}
*/

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
			// guard against zero-distance to avoid div-by-zero / NaNs
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
	dim3 threads_per_block_PWA(16, 16);
	dim3 n_blocks_PWA((NUMENTITIES + threads_per_block_PWA.x - 1) / threads_per_block_PWA.x, (NUMENTITIES + threads_per_block_PWA.y - 1) / threads_per_block_PWA.y);

	pairwiseAccels<<<n_blocks_PWA, threads_per_block_PWA>>>(d_accels, d_hPos, d_mass);
	{
		cudaError_t e = cudaGetLastError();
		fprintf(stderr, "pairwiseAccels launch: %s (%s)\n", cudaGetErrorName(e), cudaGetErrorString(e));
	}
	gpuErrchk(cudaDeviceSynchronize());

	//sum up the rows of our matrix to get effect on each entity, then update velocity and position.
	int threads_per_block_update = 256;
	int n_blocks_update = (NUMENTITIES + threads_per_block_update - 1) / threads_per_block_update;

	sumMatrices<<<n_blocks_update, threads_per_block_update>>>(d_accels, d_hVel, d_hPos);
	{
		cudaError_t e = cudaGetLastError();
		fprintf(stderr, "sumMatrices launch: %s (%s)\n", cudaGetErrorName(e), cudaGetErrorString(e));
	}
	gpuErrchk(cudaDeviceSynchronize());
	
}

extern "C" void initDeviceMemory(int numObjects)
{
	//allocate memory on device
	gpuErrchk(cudaMalloc(&d_hVel, sizeof(vector3) * numObjects));
	gpuErrchk(cudaMalloc(&d_hPos, sizeof(vector3) * numObjects));
	gpuErrchk(cudaMalloc(&d_mass, sizeof(double) * numObjects));

	// flat storage for N x N acceleration vectors
	gpuErrchk(cudaMalloc(&d_values, sizeof(vector3) * (size_t)numObjects * (size_t)numObjects));
	// allocate array of device pointers (vector3*)
	gpuErrchk(cudaMalloc(&d_accels, sizeof(vector3*) * numObjects));

	// Build host-side pointer array that points into the contiguous d_values block,
	// then copy that pointer-array to device. This is safer than writing device pointers
	// from a kernel and avoids subtle bugs.
	vector3 **h_accels = (vector3 **)malloc(sizeof(vector3*) * numObjects);
	if (!h_accels) {
		fprintf(stderr, "Failed to allocate host pointer array\n");
		exit(1);
	}
	for (int i = 0; i < numObjects; ++i) {
		// pointer arithmetic on device pointer is fine on the host as an offset value:
		// host pointer value holds device pointer (address) returned by cudaMalloc.
		h_accels[i] = d_values + (size_t)i * (size_t)numObjects;
	}
	// copy the array of device pointers into device memory
	gpuErrchk(cudaMemcpy(d_accels, h_accels, sizeof(vector3*) * numObjects, cudaMemcpyHostToDevice));
	free(h_accels);

	//transfer generated values to device
	gpuErrchk(cudaMemcpy(d_hPos, hPos, numObjects * sizeof(vector3), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_hVel, hVel, numObjects * sizeof(vector3), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_mass, mass, numObjects * sizeof(double), cudaMemcpyHostToDevice));

}

//freeHostMemory: Free storage allocated by a previous call to initHostMemory
//Parameters: None
//Returns: None
//Side Effects: Frees the memory allocated to global variables hVel, hPos, and mass.
extern "C" void freeDeviceMemory()
{
	//transfer memory back to host
	gpuErrchk(cudaMemcpy(hPos, d_hPos, NUMENTITIES * sizeof(vector3), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(hVel, d_hVel, NUMENTITIES * sizeof(vector3), cudaMemcpyDeviceToHost));

	//free memory on device
	gpuErrchk(cudaFree(d_hVel));
	gpuErrchk(cudaFree(d_hPos));
	gpuErrchk(cudaFree(d_mass));
	gpuErrchk(cudaFree(d_values));
	gpuErrchk(cudaFree(d_accels));

}