import numpy as np
import gellmann as gm

# Save data to hdf5 dataset
# f is the hdf5 file we are writing to
def append_to_hdf5(f, datasetname, data):
        chunkshape = tuple([1]   +list(data.shape))
        maxshape   = tuple([None]+list(data.shape))

        if np.isrealobj(data):
                # grab dataset from file. Create the dataset if it doesn't already exist
                if datasetname in f:
                        dset = f[datasetname]
                        dset.resize(np.shape(dset)[0] + 1, axis=0)
                else:
                        zeros = np.zeros( chunkshape )
                        dset = f.create_dataset(datasetname, data=zeros, maxshape=maxshape, chunks=chunkshape)

                # grow the dataset by one and add the data
                dset[-1] = data
                
        else:
                # grab dataset from file. Create the dataset if it doesn't already exist
                if datasetname+"R" in f:
                        dsetR = f[datasetname+"R"]
                        dsetR.resize(np.shape(dsetR)[0] + 1, axis=0)
                        dsetI = f[datasetname+"I"]
                        dsetI.resize(np.shape(dsetI)[0] + 1, axis=0)
                else:
                        zeros = np.zeros( chunkshape )
                        dsetR = f.create_dataset(datasetname+"R", data=zeros, maxshape=maxshape, chunks=chunkshape)
                        dsetI = f.create_dataset(datasetname+"I", data=zeros, maxshape=maxshape, chunks=chunkshape)

                # grow the dataset by one and add the data
                dsetR[-1] = np.real(data)
                dsetI[-1] = np.imag(data)

def append_to_hdf5_1D_scalar(outputfile, datasetname, data):
    n0 = np.shape(data)[0]
    nz = np.shape(data)[-1]
    scalar = np.zeros((n0,nz))
    for i0 in range(n0):
            scalar[i0] = gm.scalarfunc(data[i0])
    append_to_hdf5(outputfile, datasetname, scalar)


def append_to_hdf5_scalar(outputfile, datasetname, data):
    nz = np.shape(data)[-1]
    scalar = gm.scalarfunc(data)
    append_to_hdf5(outputfile, datasetname, scalar)
