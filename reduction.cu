//reduction fo practice
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void sum(int* array, int N) {
	for (int i = 1; i < N; i++) {
		array[0] += array[i];
	}
}

int main() {
	int N = 1 << 30;

	int* arr = (int*)malloc(N * sizeof(int));

	for (int i = 0; i < N; i++) {
		arr[i] = 1;
	}
	
	int* d_array;
	cudaMalloc(&d_array, N * sizeof(int));
	cudaMemcpy(d_array, arr, N * sizeof(int), cudaMemcpyHostToDevice);

	clock_t start = clock();


	sum<<<1, 1>>>(d_array, N);
	cudaDeviceSynchronize();

	cudaMemcpy(arr, d_array, sizeof(int), cudaMemcpyDeviceToHost);

	clock_t end = clock();
	printf("start: %d, end: %d, end-start: %d, cps: %d, time: %f", start, end, end-start, CLOCKS_PER_SEC, (double)(end-start)/CLOCKS_PER_SEC);
	printf("N = %d\nSUM = %d\nTime Taken = %f seconds\n", N, arr[0], (double)(clock() - start) / CLOCKS_PER_SEC);

	cudaFree(d_array);

	free(arr);
	return 0;
}