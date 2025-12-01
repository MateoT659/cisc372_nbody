//reduction fo practice
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void sum(int* array, int N) {
	int thx = threadIdx.x;
	int stride = blockDim.x;
	int blockx = blockIdx.x;
	int threadIndex = 2*(stride*blockx + thx);
	int start = 2*(stride*blockx);

	int end = (blockx+1)*(stride*2);

	int gap = 1;

	while((threadIndex-start)%(gap<<1) == 0 && gap<stride*2){
		if(threadIndex+gap < end && threadIndex+gap < N){
			array[threadIndex] += array[threadIndex+gap];
		}
		__syncthreads();
		gap <<= 1;
	}
}

int main() {
	int N = 1 << 20;

	int* arr = (int*)malloc(N * sizeof(int));

	for (int i = 0; i < N; i++) {
		arr[i] = 1;
	}
	
	int* d_array;
	cudaMalloc(&d_array, N * sizeof(int));
	cudaMemcpy(d_array, arr, N * sizeof(int), cudaMemcpyHostToDevice);

	clock_t start = clock();

	int block_size = 256;
	int n_blocks = ((N/2 + 1)+block_size-1)/(block_size);

	sum<<<n_blocks, block_size>>>(d_array, N);
	cudaDeviceSynchronize();
	
	clock_t end = clock();

	cudaMemcpy(arr, d_array, sizeof(int), cudaMemcpyDeviceToHost);
	
	printf("N = %d\nSUM = %d\nTime Taken = %f seconds\n", N, arr[0], (double)(end - start) / CLOCKS_PER_SEC);

	cudaFree(d_array);

	free(arr);
	return 0;
}