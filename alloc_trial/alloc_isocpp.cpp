#include <omp.h>
#include <iostream>
#include <ranges>
#include <algorithm>
#include <execution>
#include <signal.h>
#include <exception>
#include <stdlib.h>
#include <thrust/system_error.h>
#include <chrono>

void SignalHandler(int signal)
{
    printf("Signal %d",signal);
    throw "!Access Violation!";
}

size_t allocableSize(size_t base, int deviceId){
    typedef void (*SignalHandlerPointer)(int);
    SignalHandlerPointer previousHandlerSIGSEGV;
    //SignalHandlerPointer previousHandlerCUDA_EXCEPTION_14;
    previousHandlerSIGSEGV = signal(SIGSEGV , SignalHandler);
    //previousHandlerCUDA_EXCEPTION_14 = signal(CUDA_EXCEPTION_14 , SignalHandler);

    bool allocSuccess = true;

    // Base is the alignment size, any modern gpu will have ~1 gb of memory
    // Then we will keep trying to allocate more and more each try +1 ~1gb
    // The minimum increment will be ~250 mb which a multiple of base
    size_t min = static_cast<size_t>(2.5 * 1e8);
    while (base < min){
        base += base;
    }
    size_t sizeToAlloc = base;

    do {
        char* allocatedMemory = nullptr;
        std::cout << "Try to alloc: " << static_cast<double>(sizeToAlloc * 1e-9) << " GB" << std::endl;
        try{
            char* allocatedMemory = new char[sizeToAlloc];

            sizeToAlloc += base;
            // = to make a copy and not to use unified memory through vector
            //std::cout << "enter" << std::endl;
            auto start = std::chrono::high_resolution_clock::now();
            // Unified memory
            std::for_each(
                std::execution::par_unseq,
                allocatedMemory, allocatedMemory + sizeToAlloc, // loop range
                    [=] (char& v) {
                    v = 0;
                }
            );
            // Not unified memory?
            std::for_each_n(
                std::execution::par_unseq,
                std::views::iota(0).begin(), sizeToAlloc, // loop range
                    [=] (int i) {
                    allocatedMemory[i] = 0;
                }
            );
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = duration_cast<std::chrono::nanoseconds>(stop - start);
            std::cout << "Time per item: " << static_cast<double>(duration.count()) / static_cast<double>(sizeToAlloc) << std::endl;
            delete[] allocatedMemory;
            //std::cout << "exit" << std::endl;
        }catch (thrust::system::system_error const &exc) 
        {
            delete[] allocatedMemory;
            allocatedMemory = nullptr;
            allocSuccess = false;
            std::cout << "exception: " << exc.what() << std::endl;
        }   catch (std::exception const &exc) 
        {
            delete[] allocatedMemory;
            allocatedMemory = nullptr;
            allocSuccess = false;
            std::cout << "exception: " << exc.what() << std::endl;
        }
        catch (...)
        {
            std::cerr << "Unknown exception caught\n";
        }
    } while (allocSuccess);
    
    std::cout << static_cast<double>((sizeToAlloc - base) * 1e-9) << " GB can be allocated" << std::endl;
    return sizeToAlloc - base;
}


void handler(void){
    std::cout << ":(" << std::endl;
    //throw std::runtime_error("owo-new-handler-called");
}

int main(){
    std::unexpected_handler currentHandler = std::get_unexpected();
    std::unexpected_handler newHandler = handler;
    std::set_unexpected(newHandler);
    std::terminate_handler currentTermianteHandler = std::get_terminate();
    std::set_terminate(handler);
    int i = atexit(handler);
    if (i != 0) {
        fprintf(stderr, "cannot set exit function\n");
        exit(EXIT_FAILURE);
    }
    allocableSize(1024, 0);
}