objects = main1.o main2.o

all: main1.o main2.o
	nvcc -arch=sm_86 main1.o -o main1
	nvcc -arch=sm_86 main2.o -o main2

%.o: %.cu
	nvcc -arch=sm_86 -I. $< -o $@

clean:
	rm -f *.o main1 main2