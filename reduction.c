//reduction fo practice
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void reduce(int* array, const int N) {
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
	
	

	clock_t start = clock();

	reduce << 1, 1 >> (arr, N);
	cudaDeviceSynchronize();

	printf("N = %d\nSUM = %d\nTime Taken = %f seconds\n", N, sum, (double)(clock()-start)/CLOCKS_PER_SEC);


	free(arr);
	return 0;
}