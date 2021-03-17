#!/bin/bash

# using a trick from https://medium.com/@jeffrey_91423/binding-to-the-right-gpu-in-mpi-cuda-programs-263ac753d232

export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK

$@
