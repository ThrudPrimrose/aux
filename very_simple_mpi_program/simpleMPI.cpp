#include <iostream>
#include <mpi.h>

void foo()
{
	// just check omp directives compile
	volatile int a = 4;
#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < 1000; i++)
	{
		a++;
	}
}

int main(int argc, char *argv[])
{
	MPI_Init(&argc, &argv);

	foo();
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (!rank)
	{
		std::cout << "Help simple MPI program works!" << std::endl;
	}

	MPI_Finalize();
	return 0;
}
