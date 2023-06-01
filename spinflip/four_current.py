import numpy as np
import h5py
from constants import c, hbar, M_p

#returns J, Jbar like the above function but this one works on h5 files (takes in the dictionary outputed by extract)
#keys must be of the form ['Fx00_Re', 'Fx00_Rebar', 'Fx01_Imbar', ... 'N00_Re', 'N00_Rebar', ... 'dz(cm)', 'it', 't(s)']>
#where the numbers denote flavor components       
#number of flavors is variable. 
#Returns (nt, 4, nF, nF, nz) in units of eV^3
def four_current(infilename):
    h5_dict = h5py.File(infilename, "r")
    
    num_flavors=max([int(key[2]) for key in list(h5_dict.keys()) if key[0]=='F'])+1
    component_shape=np.shape(h5_dict['N00_Re(1|ccm)'])
    components=['N', 'Fx', 'Fy', 'Fz']

    # get coordinages, times
    #dz = np.array(h5_dict['dz(cm)'])
    #nz = np.shape(h5_dict['N00_Re(1|ccm)'])[1]
    #z = np.arange(dz/2., nz*dz, dz)
    #t = h5_dict['t(s)']

    # component shape: [nt,nz]
    # result: [4, nF, nF, nt, nz]
    J_Re=np.array([[[h5_dict[components[n] + str(min(i,j)) + str(max(i,j)) +'_Re(1|ccm)'] 
                     for i in range(0,num_flavors)] 
                     for j in range (0, num_flavors)]
                     for n in range(0,4)]) 

    J_Im=np.array([[[np.zeros(component_shape) if i==j else h5_dict[components[n] + str(min(i,j)) + str(max(i,j)) +'_Im(1|ccm)']
                     for i in range(0,num_flavors)]
                     for j in range (0, num_flavors)]
                     for n in range(0,4)])   

    J_Re_bar=np.array([[[h5_dict[components[n] + str(min(i,j)) + str(max(i,j)) +'_Rebar(1|ccm)']
                         for i in range(0,num_flavors)]
                         for j in range (0, num_flavors)]
                         for n in range (0,4)]) 

    J_Im_bar=np.array([[[np.zeros(component_shape) if i==j else h5_dict[components[n] + str(min(i,j)) + str(max(i,j)) +'_Imbar(1|ccm)']
                         for i in range(0,num_flavors)]
                         for j in range (0, num_flavors)]
                         for n in range(0,4)])
    
    h5_dict.close()
    
    #make sure J_Im is antisymmetric so J is Hermitian
    for i in range(0,num_flavors):
        for j in range(0,i):
            J_Im_bar[:,i,j,:] = -J_Im_bar[:,i,j,:]
            J_Im[    :,i,j,:] = -J_Im[    :,i,j,:]

    #store complex numbers
    J    = (1+0*1j)*J_Re     + 1j*J_Im
    Jbar = (1+0*1j)*J_Re_bar + 1j*J_Im_bar

    # rearrange the order of indices so that the time index is first
    # indices: [time, component, flavor1, flavor2, z]
    J    = np.transpose(J,    (3,0,1,2,4))
    Jbar = np.transpose(Jbar, (3,0,1,2,4))

    return (c**3*hbar**3) * (J-np.conj(Jbar))


# Calculate the gradient of the four-current in the tetrad frame
def store_gradients(merger_data_filename, emu_data_loc, output_filename, xmin, xmax, ymin, ymax, zmin,zmax, tindex):
    print("Reading data")
    merger_data = h5py.File(merger_data_filename, 'r')
    christoffel = np.array(merger_data['christoffel'])[:,:,:, xmin:xmax+1, ymin:ymax+1, zmin:zmax+1, np.newaxis,np.newaxis,np.newaxis] * (hbar*c) # [up,lo,lo,           x,y,z, f1,f2,nzsim] eV
    tetrad      = np.array(merger_data['tetrad'     ])[:,:,   xmin:xmax+1, ymin:ymax+1, zmin:zmax+1, np.newaxis,np.newaxis,np.newaxis] # [(tet_lo), coord_up, x,y,z, f1,f2,nzsim]
    tetrad_low  = np.array(merger_data['tetrad_low' ])[:,:,   xmin:xmax+1, ymin:ymax+1, zmin:zmax+1, np.newaxis,np.newaxis,np.newaxis] # [(tet_up), coord_lo, x,y,z, f1,f2,nzsim]
    x = np.array(merger_data['x(cm)'])[xmin:xmax+1, 0,           0          ] / (hbar*c) # 1/eV
    y = np.array(merger_data['y(cm)'])[0,           ymin:ymax+1, 0          ] / (hbar*c) # 1/eV
    z = np.array(merger_data['z(cm)'])[0,           0,           zmin:zmax+1] / (hbar*c) # 1/eV
    ne = np.array(merger_data['rho(g|ccm)']*merger_data['Ye']/M_p) * (hbar*c)**3 # [x,y,z] eV^3
    merger_data.close()

    nx=len(x)
    ny=len(y)
    nz=len(z)

    # get the number of cells in the 1d simulations
    emu_filename = emu_data_loc + "i{:03d}".format(xmin)+"_j{:03d}".format(ymin)+"_k{:03d}".format(zmin)+"/allData.h5"
    Jshape = np.shape(four_current(emu_filename))
    nzsim = Jshape[4]
    nF = Jshape[2]

    # fill the tetrad four-current values
    print("Filling Jtet")
    Jtet  = 0j * np.zeros((4, nx, ny, nz, nF, nF, nzsim)) # [(tetrad_up), x, y, z, f1, f2, zsim]
    Jetet = 0j * np.zeros((4, nx, ny, nz, nF, nF, nzsim)) # [(tetrad_up), x, y, z]
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                emu_filename = emu_data_loc + "i{:03d}".format(i+xmin)+"_j{:03d}".format(j+ymin)+"_k{:03d}".format(k+zmin)+"/allData.h5"
                Jtet[:,i,j,k,:,:,:] = four_current(emu_filename)[tindex] # [txyz, f1, f2, z]
                Jetet[0,i,j,k] = ne[i,j,k]
    
    # transform the tetrad four-current to the coordinate four-current
    # J^l = e^l_(:) Jtet^(:)
    print ("Transforming Jtet to Jcoord")
    Jcoord  = 0j * np.zeros((4, nx, ny, nz, nF, nF, nzsim)) # [coord_up, x, y, z, f1, f2, zsim]
    Jecoord = 0j * np.zeros((4, nx, ny, nz, nF, nF, nzsim)) # [coord_up, x, y, z, f1, f2, zsim]
    for l in range(nF):
        Jcoord[l]  = np.sum(tetrad[:,l] * Jtet , axis=0)
        Jecoord[l] = np.sum(tetrad[:,l] * Jetet, axis=0)
    
    # Take the gradient of the four-current
    # [txyz(grad,lo), txyz_up, x, y, z, f1, f2, zsim]
    print("Calculating gradJcoord")
    gradJcoord  = 0j * np.zeros((4, 4, nx, ny, nz, nF, nF, nzsim))
    gradJecoord = 0j * np.zeros((4, 4, nx, ny, nz))
    gradJcoord[1:]  = np.gradient(Jcoord , x, y, z, axis=(1,2,3))
    gradJecoord[1:] = np.gradient(Jecoord, x, y, z, axis=(1,2,3))

    # change index order to [txyz_up, txyz(grad_lo), x, y, z, f1, f2, zsim]
    gradJcoord  = np.swapaxes(gradJcoord , 0, 1)
    gradJecoord = np.swapaxes(gradJecoord, 0, 1)

    # calculate the covariant derivative of the four-current
    # [txyz_up, xyz(grad,lo), x, y, z, f1, f2, zsim]
    # J^l_;m = J^l_,m + \Gamma^l_mn J^n
    print("Calculating covarJcoord")
    covarJcoord  = 0j * np.zeros((4, 4, nx, ny, nz, nF, nF, nzsim))
    covarJecoord = 0j * np.zeros((4, 4, nx, ny, nz))
    for l in range(4):
        for m in range(4):
            covarJcoord[ l,m] = gradJcoord[ l,m] + np.sum(christoffel[l,m] * Jcoord , axis=0)
            covarJecoord[l,m] = gradJecoord[l,m] + np.sum(christoffel[l,m] * Jecoord, axis=0)

    # map back into tetrad frame
    # [txyz_up, txyz(grad_lo), x, y, z, f1, f2, zsim]
    print("Calculating covarJtet")
    covarJtet  = 0j * np.zeros((4, 4, nx, ny, nz, nF, nF, nzsim))
    covarJetet = 0j * np.zeros((4, 4, nx, ny, nz))
    for l in range(4):
        for m in range(4):
            covarJtet[ l,m] = np.sum(tetrad_low[l,:,np.newaxis] * tetrad[m,np.newaxis,:] * covarJcoord , axis=(0,1))
            covarJetet[l,m] = np.sum(tetrad_low[l,:,np.newaxis] * tetrad[m,np.newaxis,:] * covarJecoord, axis=(0,1))

    print(np.shape(covarJtet))
    print(np.shape(covarJetet))

    # write covarJtet to file
    output = h5py.File(output_filename, 'w')
    output["covarJtet(eV^4)"] = covarJtet
    output["covarJetet(eV^4)"] = covarJetet
    output["x(1|eV)"] = x
    output["y(1|eV)"] = y
    output["z(1|eV)"] = z
    output["it"] = tindex
    output["limits"] = np.array([[xmin,xmax],[ymin,ymax],[zmin,zmax]])
    output.close()

# read the covariant derivative of the four-current from file
def read_gradients(filename):
    data = h5py.File(filename, 'r')
    covarJtet = np.array(data["covarJtet(eV^4)"])
    covarJetet = np.array(data["covarJetet(eV^4)"])
    x = np.array(data["x(cm)"])
    y = np.array(data["y(cm)"])
    z = np.array(data["z(cm)"])
    it = np.array(data["it"])
    limits = np.array(data["limits"])
    data.close()
    return covarJtet, covarJetet, x, y, z, it, limits