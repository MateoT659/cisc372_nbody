//reduction fo practice
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
	int N = 1 << 30;

	int* arr = (int*)malloc(N * sizeof(int));

	for (int i = 0; i < N; i++) {
		arr[i] = 1;
	}

	clock_t start = clock();

	
	int sum = 0;
	for (int i = 0; i < N; i++) {
		sum += arr[i];
	}

	printf("\tN = %d\n\tSUM = %d\n\tTime Taken = %f seconds\n", N, sum, (double)(clock()-start)/CLOCKS_PER_SEC);


	free(arr);
	return 0;
}