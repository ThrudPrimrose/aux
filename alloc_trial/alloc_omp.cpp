#include <omp.h>
#include <iostream>

size_t allocableSize(size_t base, int deviceId){
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
            void* u = omp_target_alloc(sizeToAlloc, deviceId);
            if (u == NULL){
                throw std::runtime_error("No memory (omp_target_allocs)");
            } else {
                omp_target_free(u, deviceId);
                u = NULL;
            }
            {
                #pragma omp target enter data map(to: allocatedMemory[0:sizeToAlloc]) device(deviceId)
                {}
            }
            {
                #pragma omp target update from(allocatedMemory[0:sizeToAlloc]) device(deviceId)
                {}
            }

            sizeToAlloc += base;
            delete[] allocatedMemory;
        } catch (int errCode) {
            delete[] allocatedMemory;
            allocatedMemory = nullptr;
            allocSuccess = false;
        }
    } while (allocSuccess);
    
    std::cout << static_cast<double>((sizeToAlloc - base) * 1e-9) << " GB can be allocated" << std::endl;
    return sizeToAlloc - base;
}


int main(){
    allocableSize(1024, 0);
}