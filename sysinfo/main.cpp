#include <unistd.h>
#include <iostream>
#include <atomic>
#include <new>
struct A
{
    int a;
};

size_t getL1CacheLineSize()
{
    return sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
    // return std::hardware_destructive_interference_size;
}

struct B
{
    alignas(std::hardware_destructive_interference_size) std::atomic<int> a;
};

int main()
{
    long l1_cache_line_size = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
    std::cout << l1_cache_line_size << std::endl;
    std::cout << sizeof(std::atomic<int>) << std::endl;
    std::cout << sizeof(A) << std::endl;
    std::cout << sizeof(B) << std::endl;
}