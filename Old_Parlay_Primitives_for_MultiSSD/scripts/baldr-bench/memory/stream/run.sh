# Usage: ./run.sh NUM_THREADS ARRAY_SIZE
# compile stream with array size
# mcmodel is for pointer size
# use openmp for parallel work loads
source /opt/AMD/aocc-compiler-4.1.0/setenv_AOCC.sh
clang -O -DSTREAM_ARRAY_SIZE=$2 -mcmodel=medium -fopenmp -D_OPENMP stream.c -o stream
export OMP_NUM_THREADS=$1
./stream
