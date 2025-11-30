//reduction fo practice
#include <stdio.h>

int main() {
	int N = 1 << 20;

	int* arr = (int*) malloc(N*sizeof(int))

	for (int i = 0; i < N; i++) {
		arr[i] = 1;
	}

	int sum = 0;
	for (int i = 0; i < N; i++) {
		sum += arr[i];
	}

	printf("N = %d\n, SUM = %d\n", N, sum);

	return 0;
}