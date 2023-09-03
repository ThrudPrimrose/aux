#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>

int cuda_hello_launcher();
int cuda_hello_launcher2();
int ex();

int main() {
    std::cout << ex() << std::endl;
    cuda_hello_launcher();
    cuda_hello_launcher2();
    return 0;
}