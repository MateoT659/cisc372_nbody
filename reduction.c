//reduction fo practice
#include <stdio.h>
#include <stdlib.h>

int sum_one_to_n(int n) {
	return n * (n + 1) / 2;
}

int main() {
	int N = 1 << 20;

	int* arr = (int*)malloc(N * sizeof(int));

	for (int i = 0; i < N; i++) {
		arr[i] = i;
	}



	int sum = 0;
	for (int i = 0; i < N; i++) {
		sum += arr[i];
	}

	printf("N = %d\n SUM = %d\n EXPECTED_SUM = %d\n", N, sum, sum_one_to_n(N));

	return 0;
}