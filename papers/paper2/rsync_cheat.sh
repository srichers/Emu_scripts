#!/bin/bash

rsync --exclude="Emu/" --exclude="*plt*" --include="*.h5" --include="*/" --exclude="*.*" --relative --recursive --verbose --prune-empty-dirs srichers@cori.nersc.gov:/global/project/projectdirs/m3761 .

rsync --exclude="*plt*" --include="*.h5" --include="*/" --exclude="*.*" --relative --recursive --verbose --prune-empty-dirs srichers@cori.nersc.gov:/global/project/projectdirs/m3018/Emu .

rsync --exclude="*plt*" --include="*.h5" --include="*/" --exclude="*.*" --relative --recursive --verbose --prune-empty-dirs saricher@bridges2.psc.edu:/ocean/projects/phy200048p/shared .

# note: using -vv will tell you which rule included/excluded each file/folder

#rsync -vaz srichers@cori.nersc.gov:/global/project/projectdirs/m3761/*/**/*.h5 . --relative
#rsync -vaz srichers@cori.nersc.gov:/global/project/projectdirs/m3018/Emu/*/**/*.h5 . --relative
#rsync -vaz saricher@bridges2.psc.edu:/ocean/projects/phy200048p/shared/*/**/*.h5 . --relative
