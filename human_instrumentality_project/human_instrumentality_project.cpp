#include <iostream>
#include <mpi.h>

void foo()
{
	volatile int a = 4;
	int b = 8;
	#pragma omp parallel for schedule(dynamic, b)
	for (int i =0; i<1000; i++)
	{
		a++;
	}
	std::cout << a << std::endl;
}

int main(int argc, char *argv[])
{
	MPI_Init(&argc, &argv);
	for (int i=0;i<100;i++)
	{
		foo();
	}
	MPI_Finalize();
	return 0;
}
