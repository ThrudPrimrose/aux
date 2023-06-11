#include "sycl.h"
#include "cuda.h"
#include <iostream>

template <typename T>
class DeviceQuery
{
public:
    DeviceQuery(){};
};

class DeviceQueryFactory
{
public:
    /*
    static DeviceQuery<cuda::getDeviceInfo()> *init()
    {
        //return new DeviceQuery<cuda::getDeviceInfo()>();
        return nullptr;
    }
    */
};

int main()
{
    std::cout << cuda::getDeviceInfo() << std::endl;
    // auto a = DeviceQuery::init();
}