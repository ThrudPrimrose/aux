    #include <omp.h>
    #include <iostream>
    #define YES 0
    #define NO -1
    int main(int argc, char* argv[]) {
        int canOffload = NO;
        #pragma omp target map(tofrom: canOffload)
        {
            if (!omp_is_initial_device()) {
                canOffload = YES;
            }
        }
	std::cout << canOffload << std::endl;
        return canOffload;
    }
