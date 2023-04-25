import numpy as np
import h5py
from spin_flip_tools import four_current

# Calculate the gradient of the four-current in the tetrad frame
def store_gradients(merger_data_filename, emu_data_loc, output_filename, xmin, xmax, ymin, ymax, zmin,zmax, tindex):
    print("Reading data")
    merger_data = h5py.File(merger_data_filename, 'r')
    christoffel = np.array(merger_data['christoffel'])[:,:,:, xmin:xmax+1, ymin:ymax+1, zmin:zmax+1, np.newaxis,np.newaxis,np.newaxis] # [up,lo,lo,           x,y,z, f1,f2,nzsim]
    tetrad      = np.array(merger_data['tetrad'     ])[:,:,   xmin:xmax+1, ymin:ymax+1, zmin:zmax+1, np.newaxis,np.newaxis,np.newaxis] # [(tet_lo), coord_up, x,y,z, f1,f2,nzsim]
    tetrad_low  = np.array(merger_data['tetrad_low' ])[:,:,   xmin:xmax+1, ymin:ymax+1, zmin:zmax+1, np.newaxis,np.newaxis,np.newaxis] # [(tet_up), coord_lo, x,y,z, f1,f2,nzsim]
    x = np.array(merger_data['x(cm)'])[xmin:xmax+1, 0,           0          ]
    y = np.array(merger_data['y(cm)'])[0,           ymin:ymax+1, 0          ]
    z = np.array(merger_data['z(cm)'])[0,           0,           zmin:zmax+1]
    merger_data.close()

    nx=len(x)
    ny=len(y)
    nz=len(z)

    # get the number of cells in the 1d simulations
    emu_filename = emu_data_loc + "i{:03d}".format(xmin)+"_j{:03d}".format(ymin)+"_k{:03d}".format(zmin)+"/allData.h5"
    Jshape = np.shape(four_current(emu_filename))
    nzsim = Jshape[4]
    nF = Jshape[2]
    print("nzsim = ", nzsim)
    print("nF = ", nF)

    # fill the tetrad four-current values
    print("Filling Jtet")
    Jtet = 0j * np.zeros((4, nx, ny, nz, nF, nF, nzsim)) # [(tetrad_up), x, y, z, f1, f2, zsim]
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                emu_filename = emu_data_loc + "i{:03d}".format(i+xmin)+"_j{:03d}".format(j+ymin)+"_k{:03d}".format(k+zmin)+"/allData.h5"
                Jtet[:,i,j,k,:,:,:] = four_current(emu_filename)[tindex] # [txyz, f1, f2, z]
    
    # transform the tetrad four-current to the coordinate four-current
    # J^l = e^l_(:) Jtet^(:)
    print ("Transforming Jtet to Jcoord")
    Jcoord = 0j * np.zeros((4, nx, ny, nz, nF, nF, nzsim)) # [coord_up, x, y, z, f1, f2, zsim]
    for l in range(nF):
        Jcoord[l] = np.sum(tetrad[:,l] * Jtet, axis=0)
    
    # Take the gradient of the four-current
    # [txyz(grad,lo), txyz_up, x, y, z, f1, f2, zsim]
    print("Calculating gradJcoord")
    gradJcoord = 0j * np.zeros((4, 4, nx, ny, nz, nF, nF, nzsim))
    gradJcoord[1:] = np.gradient(Jcoord, x, y, z, axis=(1,2,3))

    # change index order to [txyz_up, txyz(grad_lo), x, y, z, f1, f2, zsim]
    gradJcoord = np.swapaxes(gradJcoord, 0, 1)

    # calculate the covariant derivative of the four-current
    # [txyz_up, xyz(grad,lo), x, y, z, f1, f2, zsim]
    # J^l_;m = J^l_,m + \Gamma^l_mn J^n
    print("Calculating covarJcoord")
    covarJcoord = 0j * np.zeros((4, 4, nx, ny, nz, nF, nF, nzsim))
    for l in range(4):
        for m in range(4):
            covarJcoord[l,m] = gradJcoord[l,m] + np.sum(christoffel[l,m] * Jcoord, axis=0)

    # map back into tetrad frame
    # [txyz_up, txyz(grad_lo), x, y, z, f1, f2, zsim]
    print("Calculating covarJtet")
    covarJtet = 0j * np.zeros((4, 4, nx, ny, nz, nF, nF, nzsim))
    for l in range(4):
        for m in range(4):
            covarJtet[l,m] = np.sum(tetrad_low[l,:,np.newaxis] * tetrad[m,np.newaxis,:] * covarJcoord, axis=(0,1))

    print(np.shape(covarJtet))

    # write covarJtet to file
    output = h5py.File(output_filename, 'w')
    output["covarJtet(eV^3)"] = covarJtet
    output["x(cm)"] = x
    output["y(cm)"] = y
    output["z(cm)"] = z
    output.close()

# read the covariant derivative of the four-current from file
def read_gradient(filename):
    data = h5py.File(filename, 'r')
    covarJtet = np.array(data["covarJtet(eV^3)"])
    x = np.array(data["x(cm)"])
    y = np.array(data["y(cm)"])
    z = np.array(data["z(cm)"])
    data.close()
    return covarJtet, x, y, z