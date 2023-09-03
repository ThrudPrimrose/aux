libtoolize && aclocal && autoconf && autoheader && automake --add-missing
CC=nvc CXX="nvc++" ./configure --with-cuda=/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/cuda --with-arch=sm_86
#VERBOSE=1 make --debug
#cd src/exec && VERBOSE=1 make --debug