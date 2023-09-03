AC_INIT([cuda_automake], [1.0], [])
AM_INIT_AUTOMAKE([foreign])

AC_PROG_CXX
AC_CONFIG_HEADERS([src/config.h])
AC_CONFIG_FILES([
 Makefile
 src/Makefile
])

AC_PROG_RANLIB

AC_LANG_DEFINE([CUDA], [cuda], [CUDA], [CUDACXX],
[])

AC_ARG_WITH([cuda],
   [AS_HELP_STRING([--with-cuda=PATH], [specify CUDA installation path])],
   [with_cuda="$withval"],
   [with_cuda=""])

AC_ARG_WITH([arch],
   [AS_HELP_STRING([--with-arch=ARCH], [specify GPU architecture])],
   [AC_MSG_CHECKING(for GPU arch)
   case "$withval" in
         sm_*)
         arch="$withval" ;;
         *)
         AC_MSG_ERROR([Invalid architecture. Option should start with 'sm_'.]) ;;
   esac
   ],
   [
   AC_MSG_ERROR([Architecture is required (2). Use --arch=ARCH, where ARCH starts with 'sm_' given $withval.])
   ])

if test -n "$with_cuda"
then
   if test "$with_cuda" = yes
   then
      AC_MSG_ERROR([With-cuda argument requires the path to CUDA, --with-cuda=<CUDA_PATH>])
   fi

   if test -z "$with_cuda"
   then
      AC_MSG_ERROR([If you are building with CUDA GPU support, please input the CUDA path with --with-cuda=<CUDA_PATH>])
   fi

   # Extract the path from the "--with-cuda" argument
   cuda_path=$(echo "$with_cuda" | sed 's/--with-cuda=//')

   # Check if the path is valid
   if test ! -d "$cuda_path"; then
      AC_MSG_ERROR([The input for cuda-path=$cuda_path is not a valid directory])
   fi

   CUDA_INCLUDE_FLAGS="-I$with_cuda/include"
   CUDA_LIBS="-L$with_cuda/lib64 -lcudart" #-L$with_cuda/lib 
   NVCC="$with_cuda/bin/nvcc"
   CUDA_PATH="$with_cuda"
else
   AC_MSG_ERROR([B: $with_cuda])
fi


AC_SUBST(CUDA_INCLUDE_FLAGS)
AC_SUBST(CUDA_LIBS)
AC_SUBST(NVCC)
AC_SUBST(CUDA_PATH)

AC_OUTPUT