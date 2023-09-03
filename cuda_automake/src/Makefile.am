AUTOMAKE_OPTIONS = subdir-objects

ARFLAGS = crU

lib_LIBRARIES = libcuda_automake_example_release.a libcuda_automake_example_debug.a

libcuda_automake_example_release_a_SOURCES = aux/hmm.cpp 
#aux/aux.cu aux/aux2.cu
libcuda_automake_example_release_a_CXXFLAGS = $(CUDA_INCLUDE_FLAGS)

libcuda_automake_example_debug_a_SOURCES = aux/hmm.cpp 
#aux/aux.cu aux/aux2.cu
libcuda_automake_example_debug_a_CXXFLAGS = $(CUDA_INCLUDE_FLAGS)

CUDA_OBJS_RELEASE = $(abs_top_srcdir)/src/aux/libcuda_automake_example_release_a-aux.o $(abs_top_srcdir)/src/aux/libcuda_automake_example_release_a-aux2.o
CUDA_OBJS_DEBUG = $(abs_top_srcdir)/src/aux/libcuda_automake_example_debug_a-aux.o $(abs_top_srcdir)/src/aux/libcuda_automake_example_debug_a-aux2.o

#libcuda_automake_example_release_a_BUILT_SOURCES = $(CUDA_OBJS_RELEASE)
#libcuda_automake_example_debug_a_BUILT_SOURCES = $(CUDA_OBJS_DEBUG)

#libcuda_automake_example_release_a_SOURCES += $(libcuda_automake_example_release_a_BUILT_SOURCES)
#libcuda_automake_example_debug_a_SOURCES += $(libcuda_automake_example_debug_a_BUILT_SOURCES)

#libcuda_automake_example_debug_a_OBJECTS = $(libcuda_automake_example_debug_a_SOURCES:.cpp=.o) $(CUDA_OBJS_DEBUG)
#libcuda_automake_example_release_a_OBJECTS = $(libcuda_automake_example_release_a_SOURCES:.cpp=.o) $(CUDA_OBJS_RELEASE)

NVCCFLAGS_RELEASE = -arch=sm_86 -O3
NVCCFLAGS_DEBUG = -arch=sm_86 -O0 -g

$(CUDA_OBJS_RELEASE): $(abs_top_srcdir)/src/aux/libcuda_automake_example_release_a-%.o: $(abs_top_srcdir)/src/aux/%.cu
	@echo "Compiling $< to $@"
	$(NVCC) $(NVCCFLAGS_RELEASE) -G -Xcompiler="$(CXXFLAGS_PEANO_2D_RELEASE)" -ccbin=$(CXX) -c $< -o $@ 

$(CUDA_OBJS_DEBUG): $(abs_top_srcdir)/src/aux/libcuda_automake_example_debug_a-%.o: $(abs_top_srcdir)/src/aux/%.cu
	@echo "Compiling $< to $@"
	$(NVCC) $(NVCCFLAGS_DEBUG) --dlink-time-opt -lineinfo -Xcompiler="$(CXXFLAGS_PEANO_2D_DEBUG)" -ccbin=$(CXX) -c $< -o $@ 


libcuda_automake_example_release_a_LIBADD = $(CUDA_OBJS_RELEASE)
libcuda_automake_example_debug_a_LIBADD = $(CUDA_OBJS_DEBUG)

#libcuda_automake_example_release_a_OBJECTS = $(libcuda_automake_example_release_a_SOURCES:.cpp=.o) $(CUDA_OBJS_RELEASE)
#libcuda_automake_example_debug_a_OBJECTS = $(libcuda_automake_example_debug_a_SOURCES:.cpp=.o) $(CUDA_OBJS_DEBUG)

#.PHONY: libcuda_automake_example_release.a
#libcuda_automake_example_release.a: $(libcuda_automake_example_release_a_OBJECTS)
#	$(AR) $(ARFLAGS) $@ $^

#.PHONY: libcuda_automake_debug_release.a
#libcuda_automake_example_debug.a: $(libcuda_automake_example_debug_a_OBJECTS)
#	$(AR) $(ARFLAGS) $@ $^

#noinst_EXTRA_libcuda_automake_example_release_a_SOURCES = $(CUDA_OBJS_RELEASE)
#noinst_EXTRA_libcuda_automake_example_debug_a_SOURCES = $(CUDA_OBJS_DEBUG)

# Rule for linking the library
#libcuda_automake_example_release_a_LINK = $(NVCC) -shared -o $@ $(libmylibrary_a_OBJECTS) $(CUDA_LDFLAGS)
#libcuda_automake_example_debug_a_LINK = $(NVCC) -shared -o $@ $(libmylibrary_a_OBJECTS) $(CUDA_LDFLAGS)

# Phony target to build the library
#.PHONY: libcuda_automake_example_release.a
#libcuda_automake_example_release.a: $(libcuda_automake_example_release_a_OBJECTS)
#	$(libcuda_automake_example_release_a_LINK)

#.PHONY: libcuda_automake_example_release.a
#libcuda_automake_example_debug.a: $(libcuda_automake_example_debug_a_OBJECTS)
#	$(libcuda_automake_example_debug_a_LINK)