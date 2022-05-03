#include <iostream>
#include <upcxx/upcxx.hpp>

int main(int argc, char *argv[])
{
	upcxx::init();
	if (upcxx::rank_me() == 0)
	{
		std::cout << "Hello from very simple UPC++ program!" << std::endl;
	}
	upcxx::finalize();
}
