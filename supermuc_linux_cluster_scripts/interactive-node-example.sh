module load netcdf
module load metis

export GASNET_PHYSMEM_MAX='41 GB'

upcxx-run -n 64 -N 1 -shared-heap 128MB ./pond-plain -x 8000 -y 8000 -p 500 -c 10 --scenario 3 -o ./out -e 50.0
