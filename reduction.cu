//reduction fo practice
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void sum(int* array, int N) {
	int thx = threadIdx.x;
	int stride = blockDim.x;
	for (int i = thx; i < N; i+= stride) {
		array[0] += array[i];
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

	sum<<<1, block_size>>>(d_array, N);
	cudaDeviceSynchronize();

	cudaMemcpy(arr, d_array, sizeof(int), cudaMemcpyDeviceToHost);

	clock_t end = clock();
	printf("N = %d\nSUM = %d\nTime Taken = %f seconds\n", N, arr[0], (double)(clock() - start) / CLOCKS_PER_SEC);

	cudaFree(d_array);

	free(arr);
	return 0;
}