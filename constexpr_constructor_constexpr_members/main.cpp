#include <cstdint>
#include <iostream>

struct FLOPsPerStep {
public:
  const uint64_t a;
  const uint64_t b;
  const uint64_t c;
  constexpr FLOPsPerStep(): a{1}, b{2}, c{0} {}
  constexpr FLOPsPerStep(const uint64_t a, const uint64_t b): a{a}, b{b}, c{0} {}
  /* This is not possible
  constexpr FLOPsPerStep& operator=(FLOPsPerStep& other) {
    a = other.a;
    b = other.b;
    c = other.c;
  }
  */
  ~FLOPsPerStep() = default; //constexpr destructor is c++20 only
};

/*
FLOPsPerStep fps;

constexpr FLOPsPerStep create(const uint64_t a, const uint64_t b) {
    fps = FLOPsPerStep{a, b};
}
*/

int main(){
    constexpr FLOPsPerStep a;
    constexpr FLOPsPerStep b{3, 4};

    constexpr uint64_t a1 = a.a;
    constexpr uint64_t a2 = b.a;

    if constexpr (a1) {
        std::cout << a1 << std::endl;
    }
    if constexpr (a2) {
        std::cout << a2 << std::endl;
    }
    if constexpr (a.b) {
        std::cout << a.b << std::endl;
    }
    if constexpr (b.b) {
        std::cout << b.b << std::endl;
    }
}