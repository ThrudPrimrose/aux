import os
import sys
import subprocess

cmakeLists = """
set(CMAKE_BUILD_TYPE Release CACHE STRING "Set the CMAKE_BUILD_TYPE to the build mode of the generated application" FORCE)

add_executable(b_${CMAKE_BUILD_TYPE} b.cpp)
target_link_libraries(b_${CMAKE_BUILD_TYPE} PUBLIC a_${CMAKE_BUILD_TYPE})

# Add reconfigure step as a dependency
add_dependencies(b_${CMAKE_BUILD_TYPE} reconf)
"""

generated_sub_dir="""
add_subdirectory(${CMAKE_BINARY_DIR}/src/b)
"""


bcpp = """
#include <iostream>

void get_greeting_from_a();

int main(){
    get_greeting_from_a();
    std::cout << "Hello from b" << std::endl;
}
"""

cur_dir = os.getcwd()
print(cur_dir)
with open( os.path.join(cur_dir, "CMakeLists.txt"), "w") as clists:
    with  open( cur_dir + "/../../GeneratedSubdirectory.cmake", "w") as gensub:
        with open( os.path.join(cur_dir, "b.cpp"), "w") as b:
            clists.write(cmakeLists)
            gensub.write(generated_sub_dir)
            b.write(bcpp)
            pass

subprocess.run([f'cd {cur_dir}/../.. && cmake .. && make b_Release'], shell=True, cwd=os.getcwd())