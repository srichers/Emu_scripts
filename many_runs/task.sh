#!/bin/bash

cd $1
../../../main3d.gnu.haswell.TPROF.ex inputs >& emu_output.txt
#python3 ../../../reduce_data.py >& reduce_data_output.txt
#python3 ../../../combine_files.py reduced_data.h5 >& combine.txt
#python3 ../../../combine_files.py reduced_data_fft_power.h5 >& combine_fft.txt
python3 ../../../convertToHDF5.py
