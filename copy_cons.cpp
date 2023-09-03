#include <iostream>

class A
{
public:
    int a;
    A() : a(0) { std::cout << "Calling default constructory" << std::endl; }
    A(const A &other) : a(other.a) { std::cout << "Calling copy constructor" << std::endl; }
    A &operator=(const A &other)
    {
        a = other.a;
        std::cout << "Calling assignment operator" << std::endl;
        return *this;
    }
};

int main()
{
    A a1;
    A a2(a1);
    A a3;
    a3 = a1;
}