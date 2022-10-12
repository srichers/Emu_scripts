#!/usr/bin/env python
# coding: utf-8

#CHDIR COMMAND ONLY FOR JUPYTER
import os
import sys
import yt
import numpy as np
import matplotlib.pyplot as plt
import h5py
import amrex_plot_tools as amrex
import emu_yt_module as emu
import reduce_data as rd
from numpy import sin, cos, exp

###### Parameters ######
########################

#fermi coupling constant: G/(c*hbar)^3=1.166 378 7(6)×10−5 GeV−2 --> G=1.166 378 7×10−23 eV^−2 (natural units:c=hbar=1)
pi=np.pi
G=1.1663787*10**(-23) # eV^-2
c=29979245800         # cm/s
hbar=6.582119569e-16  # erg s
#mixing angles (rad): (values from NuFit)
a12=33.4*np.pi*2/360
a13=49.2*np.pi*2/360
a23=8.57*np.pi*2/360

#CP phase:
delta=194*np.pi*2/360
#majorana angles are all 0 and dont influence the matrix

#masses (eV) (Negative mass? 0 mass?)
m_1=0.608596511
m_2=0.608
m_3=0.649487372

#test neutrino momentum:
p_abs=10**7#eV

# set of orthonormal vectors, where n_vector is pointed along theta,phi
class Basis:
        def __init__(self, theta, phi): #theta is polar, phi is azimuthal
            self.phi=phi
            self.n_vector=np.array([1,cos(phi)*sin(theta),sin(phi)*sin(theta),cos(theta)])
            self.x1=np.array([0,cos(phi)*cos(theta),sin(phi)*cos(theta),(-1)*sin(theta)])
            self.x2=np.array([0,-sin(phi),cos(phi),0])
            
            
#### Functions ####
###################

#unitary trace matrix
def trace_matrix(data):#takes in (nF,nF,nz)
        nz = data.shape[2]
        nF = data.shape[0]
        # calculate trace
        diagonals = np.array([data[n,n,:] for n in range(nF)])
        trace = np.sum(diagonals, axis=0) # length of nz
        
        # create an identity matrix multiplied by the trace
        matrix=np.zeros_like(data)
        for n in range(nF):
                matrix[n,n,:]=trace
        
        return matrix
	
#conjugate a matrix
def conj(matrix):
	conjugate=np.transpose(np.conjugate(matrix))
	return conjugate
	
#z-derivative of (nF,nF,nz) matrix (returns nF,nF,nz)
#TODO - needs to implement periodic boundary conditions.
def der(data,ad):
	dq=ad['index','dz'].d
	shape=np.shape(data)
	der=1j*np.zeros(shape)
	for n in range(0,shape[2]):
		der[:,:,n]=(data[:,:,n]-(1+0j)*data[:,:,n-1])/dq[n]
	return der


#Gell-Mann matrices (for scalarfunc)
GM=np.array([[[0, 1, 0],
              [1, 0, 0],
              [0, 0, 0]],
             
             [[ 0, -1j, 0],
              [1j,  0 , 0],
              [ 0,  0 , 0]],
             
             [[1,  0, 0],
              [0, -1, 0],
              [0,  0, 0]],
             
             [[0, 0, 1],
              [0, 0, 0],
              [1, 0, 0]],
             
             [[0 , 0, -1j],
              [0 , 0,  0 ],
              [1j, 0,  0 ]],
             
             [[0, 0, 0],
              [0, 0, 1],
              [0, 1, 0]],
             
             [[0, 0 ,  0 ],
              [0, 0 , -1j],
              [0, 1j,  0 ]],
             
             [[3**(-1/2), 0        , 0           ],
              [0        , 3**(-1/2), 0           ],
              [0        , 0        , -2*3**(-1/2)]]
])

#scalarfunc: averages square magnitude of components for every location in nz and returns a list of these
# input: array of shape (3,3,nz)
# ouptut: array of length nz
def scalarfunc(array):
    nz = np.shape(array)[2]

    # get the coefficients of each Gell-Mann matrix
    # GM matrices are normalized so that Tr(G_a G_b) = 2 delta_ab
    # result has shape (nz,8)
    components = 0.5 * np.array([[
            np.trace( np.matmul( GM[k], array[:,:,n] ) )
            for k in range(8)] for n in range(nz)] )

    # Construct a scalar out of the GM coefficients as sqrt(G.G) where G is the vector of GM matrix coefficients
    # even though for a Hermitian matrix coefficients must be real, the input could be multiplied by an overall phase
    # Return a real number by using a*conj(a)
    # result has length nz
    scalars = np.sqrt( np.sum( components*np.conj(components), axis=1) )

    return np.real(scalars)

#like scalarfunc, but takes in time dependent arrays (nt, nF, nF, nz) and applies scalarfunc at every time point
def scalarfunc_time(array):
    return np.array([scalarfunc(array[t]) for t in range(0,np.shape(array)[0])])
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
            scalar[i0] = scalarfunc(data[i0])
    append_to_hdf5(outputfile, datasetname, scalar)


def append_to_hdf5_scalar(outputfile, datasetname, data):
    nz = np.shape(data)[-1]
    scalar = scalarfunc(data)
    append_to_hdf5(outputfile, datasetname, scalar)

    #total takes in read-only h5 dataset and the array you want to compute and just adds the real and imaginary parts for the array
def total(d, array):
    return (1+0*1j)*np.array(d[array+'R'])+1j*np.array(d[array+'I'])
def J(d):
    return total(d,'J(eV^3)')

#GM gets the 8 GM components of the flux matrix
def GM_components(d):
    flux=J(d)
    return np.array([[[[np.trace( np.matmul( GM[k], flux[t,x,:,:,n] ) ) 
                        for k in range (0,8)] for n in range (0,np.shape(flux)[-1])]
                        for x in range (0,4)] for t in range (0,np.shape(flux)[0])])

def datasaver(data,filename): #data is the array/var to be saved, filename is a string. saves to a directory on my computer so it wont work on another unless you change the path
    current_directory=os.getcwd()
    os.chdir('/home/henryrpg/Desktop/N3AS/savedarrays')
    opendata=open(filename, 'wb')
    pickle.dump(data,opendata)
    opendata.close()
    os.chdir(current_directory)
    return

def dataloader(filename): 
    current_directory=os.getcwd()
    os.chdir('/home/henryrpg/Desktop/N3AS/savedarrays')
    opendata=open(filename, 'rb')
    os.chdir(current_directory)
    return pickle.load(opendata)

###################
###################

## Chiral Potentials ##
# input: flux[spacetime, f1, f2, z]
# output: Sigma_R[spacetime, f1, f2, z]
def sigma(flux):
        Sigma_R=0j*np.zeros(np.shape(flux)) 
        Sigma_L=0j*np.zeros(np.shape(flux))
        for n in range(0,4):
                Sigma_R[n]=2**(1./2.)*G*(flux[n]+trace_matrix(flux[n]))
        Sigma_L=(-1)*np.transpose(Sigma_R, axes=(0,2,1,3)) #for majorana 
        return Sigma_R, Sigma_L

#potential projected onto the basis
def dot(potential,vector):
	projection=-vector[0]*potential[0]
	for k in range(1,4):
		projection=projection+vector[k]*potential[k]
	return -projection

def plus(potential, basis): #(3,3,nz)
	vector=0.5*(basis.x1+1j*basis.x2)
	plus=np.exp(1j*basis.phi)*dot(potential,vector)
	return plus

def minus(potential, basis): #(3,3,nz)
	vector=0.5*(basis.x1-1j*basis.x2)
	minus=np.exp(-1j*basis.phi)*dot(potential,vector)
	return minus
	
def kappa(potential, basis):
	return dot(potential,basis.n_vector)
	
## Mass Matrix ##	
m23=np.array([[1,0*1j,0],
              [0,cos(a23),sin(a23)],
              [0,-sin(a23),cos(a23)]])

m13=np.array([[cos(a13),0,sin(a13)*exp(-1j*delta)],
              [0,1,0],
              [-sin(a13)*exp(1j*delta),0,cos(a13)]])

m12=np.array([[cos(a12),sin(a12),0],
              [-sin(a12),cos(a12),0],
              [0,0*1j,1]])

m=m23 @ m13 @ m12
#m is the mass mixing (MNS) matrix -- m*(1,2,3)=(e,mu,tau)
M_mass_basis=([[m_1,0*1j,0],[0,m_2,0],[0,0,m_3]])
M_3flavor = m @ M_mass_basis @ conj(m) #(3,3)
M=M_3flavor

m2 = np.array([[cos(a12),sin(a12)],[-sin(a12),cos(a12)]])
M_mass_basis_2 =([[m_1,0*1j],[0,m_2]])
M_2flavor = m2 @ M_mass_basis_2 @ conj(m2)
                                    
                

        

#### PLOTS ####
def zplot(funcs,scale):
	fig,ax=plt.subplots()
	for n in np.arange(np.shape(funcs)[0]):
                ax.plot(np.arange(np.shape(funcs)[1]),funcs[n],label='n')
	plt.yscale(scale)
	ax.set_xlabel('z-position (cell number)')
	ax.set_ylabel('Energy (eV)')
	ax.set_title('Variation of functions with Position')
	ax.legend()
	plt.show()
	return
	
def plot(funcs,scale,xlabel,ylabel,name): #funcs is a list of tuples with legend name in [1] position
	fig,ax=plt.subplots()
	for n in np.arange(len(funcs)):
                ax.plot(np.arange(np.shape(funcs[n][0])[0]),funcs[n][0],label=funcs[n][1])
	plt.yscale(scale)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.set_title(name)
	ax.legend()
	plt.show()
	return


# Four currents for particles and antiparticles, indexed by [spacetime component, f1, f2, z]
#total current (defined in the interact function as J) is the particle current minus the conjugate of the antiparticle current
def four_current_eds(eds):
    ad = eds.ds.all_data()

    # get array of number densities and number flux densities
    N,     NI     = rd.get_matrix(ad,"N","")
    Fx,    FxI    = rd.get_matrix(ad,"Fx","")
    Fy,    FyI    = rd.get_matrix(ad,"Fy","")
    Fz,    FzI    = rd.get_matrix(ad,"Fz","")
    Nbar,  NIbar  = rd.get_matrix(ad,"N","bar")
    Fxbar, FxIbar = rd.get_matrix(ad,"Fx","bar") 
    Fybar, FyIbar = rd.get_matrix(ad,"Fy","bar") 
    Fzbar, FzIbar = rd.get_matrix(ad,"Fz","bar")

    # J^mu 4-vectors (number density and fluxes)
    JR    = np.array([N    , Fx    , Fy    , Fz    ])
    JI    = np.array([NI   , FxI   , FyI   , FzI   ])
    JRbar = np.array([Nbar , Fxbar , Fybar , Fzbar ])
    JIbar = np.array([NIbar, FxIbar, FyIbar, FzIbar])
    J    = JR    + 1j*JI
    Jbar = JRbar + 1j*JIbar

    return J, Jbar


#### FUNCS FOR h5py DATA PROCESSING ###

#data extractor, takes in h5py file directory and returns dictionary with data as items and the file keys as ... keys
def extract(h5_directory):
    
    data = h5py.File(h5_directory, "r")
    
    #extract data into dictionary
    d={key:np.array(data[key]) for key in list(data.keys())}
    
    return d

#data extractor but it takes the data out at a specific time with timestep t (won't work if there is time independent data other than dz)
def extract_time(h5_directory, t):
    
    data = h5py.File(h5_directory, "r")
    nt = np.shape(np.array(data['N00_Re(1|ccm)']))[0]
    #extract data into dictionary, selects the timestep t for all time dependent data
    d = dict((key, np.array(data[key])[t]) if key != "dz(cm)" else (key, np.array(data[key])) for key in list(data.keys()))
    
    return d

#returns J, Jbar like the above function but this one works on h5 files (takes in the dictionary outputed by extract)
#keys must be of the form ['Fx00_Re', 'Fx00_Rebar', 'Fx01_Imbar', ... 'N00_Re', 'N00_Rebar', ... 'dz(cm)', 'it', 't(s)']>
#where the numbers denote flavor components       
#number of flavors is variable. 
#Returns (4, nF, nF, nz)
def four_current_h5(h5_dict):
    
     #this is kind of janky but I'm going to figure out how many flavors there are by finding the maximum value of the third letter in the keys, which is always a flavor number
        
    num_flavors=max([int(key[2]) for key in list(h5_dict.keys()) if key[0]=='F'])+1
    component_shape=np.shape(h5_dict['N00_Re(1|ccm)'])
    components=['N', 'Fx', 'Fy', 'Fz']

    J_Re=np.array([[[h5_dict[components[n] + str(min(i,j)) + str(max(i,j)) +'_Re(1|ccm)'] for i in range(0,num_flavors)] 
                            for j in range (0, num_flavors)] for n in range(0,4)]) 
    J_Im=np.array([[[np.zeros(component_shape) if i==j else h5_dict[components[n] + str(min(i,j)) + str(max(i,j)) +'_Im(1|ccm)'] for i in range(0,num_flavors)] for j in range (0, num_flavors)] for n in range(0,4)])   
        
    J_Re_bar=np.array([[[h5_dict[components[n] + str(min(i,j)) + str(max(i,j)) +'_Rebar(1|ccm)'] for i in range(0,num_flavors)] for j in range (0, num_flavors)] 
   for n in range (0,4)]) 
    J_Im_bar=np.array([[[np.zeros(component_shape) if i==j else h5_dict[components[n] + str(min(i,j)) + str(max(i,j)) +'_Imbar(1|ccm)'] for i in range(0,num_flavors)] for j in range (0, num_flavors)] for n in range(0,4)])             
    for i in range(0,num_flavors):
        for j in range(0,i):
            J_Im_bar[:,i,j,:]=-J_Im_bar[:,i,j,:]
            J_Im[:,i,j,:]=-J_Im[:,i,j,:]
            
    J = (1+0*1j)*J_Re + 1j*J_Im
    Jbar = (1+0*1j)*J_Re_bar + 1j*J_Im_bar
    
    return (c**3*hbar**3)*J, (c**3*hbar**3)*Jbar
                    


## Non-Interacting Term ## [f1, f2]

def H_R_free(M):
    return 0.5*(1/p_abs)*np.matmul(M,conj(M))
def H_L_free(M):
    return 0.5*(1/p_abs)*np.matmul(M,conj(M))

H_R_free_3flavor = H_R_free(M_3flavor)
H_L_free_3flavor = H_L_free(M_3flavor)

# Input: what folder do we want to process?
def interact_scalar(d, outputfilename, basis_theta, basis_phi):
    # Read in the data
    eds = emu.EmuDataset(d)
    nz = eds.Nz

    # open the hdf5 file
    outputfile = h5py.File(outputfilename, "a")

    t = eds.ds.current_time
    append_to_hdf5(outputfile,"t(s)",t)

    # write the free Hamiltonians
    if "H_R_free(eV)" not in outputfile:
            outputfile["H_R_free(eV)"] = H_R_free
            outputfile["H_L_free(eV)"] = H_L_free

    # write the z grid
    if "z(cm)" not in outputfile:
            outputfile["z(cm)"] = np.arange(eds.dz/2., nz*eds.dz, eds.dz)
    
    # [spacetime component, f1, f2, z]
    #particle, antiparticle and total neutrino four currents respectively
    J_p = four_current_eds(eds)[0]
    J_a = four_current_eds(eds)[1] 
    J = J_p-np.conj(J_a)
    
    append_to_hdf5_1D_scalar(outputfile, "J_p(eV^3)", J_p)
    append_to_hdf5_1D_scalar(outputfile, "J_a(eV^3)", J_a)
    append_to_hdf5_1D_scalar(outputfile, "J(eV^3)", J)

    # [spacetime, f1, f2, z]
    S_R,S_L=sigma(J)
    append_to_hdf5_1D_scalar(outputfile, "S_R(eV)", S_R)
    append_to_hdf5_1D_scalar(outputfile, "S_L(eV)", S_L)

    # define the basis as along z
    basis = Basis(basis_theta,basis_phi)
    
    # precompute Sigma [f1, f2, z]
    S_R_plus = plus(S_R, basis)
    S_L_plus = plus(S_L, basis)
    S_R_minus = minus(S_R, basis)
    S_L_minus = minus(S_L, basis)
    S_R_kappa = kappa(S_R, basis)
    S_L_kappa = kappa(S_L, basis)
    append_to_hdf5_scalar(outputfile, "S_R_plus(eV)", S_R_plus)
    append_to_hdf5_scalar(outputfile, "S_L_plus(eV)", S_L_plus)
    append_to_hdf5_scalar(outputfile, "S_R_minus(eV)", S_R_minus)
    append_to_hdf5_scalar(outputfile, "S_L_minus(eV)", S_L_minus)
    append_to_hdf5_scalar(outputfile, "S_R_kappa(eV)", S_R_kappa)
    append_to_hdf5_scalar(outputfile, "S_L_kappa(eV)", S_L_kappa)
    
    ## Helicity-Flip Hamiltonian ## [f1, f2, z]
    MSl = np.array([ np.matmul(conj(M),S_L_minus[:,:,n]) for n in range(nz) ])
    SrM = np.array([ np.matmul(S_R_plus[:,:,n],conj(M))  for n in range(nz) ])
    H_LR = (-1/p_abs)*(SrM-MSl)
    H_LR = H_LR.transpose((1,2,0))
    append_to_hdf5_scalar(outputfile, "H_LR(eV)", H_LR)    
    
    # plusminus term [f1, f2, z]
    H_R_plusminus = 2./p_abs * np.array([
            np.matmul(S_R_plus[:,:,z], S_R_minus[:,:,z])
            for z in range(nz)]).transpose((1,2,0))
    H_L_minusplus = 2./p_abs * np.array([
            np.matmul(S_L_minus[:,:,z], S_L_plus[:,:,z])
            for z in range(nz)]).transpose((1,2,0))
    append_to_hdf5_scalar(outputfile, "H_R_plusminus(eV)", H_R_plusminus)
    append_to_hdf5_scalar(outputfile, "H_L_minusplus(eV)", H_L_minusplus)
    
    ##H_R/H_L in the (0,0,10**7) basis (derivatives along x1 and x2 are 0 for 1d setup)
    H_Rz = S_R_kappa + H_R_free[:,:,np.newaxis] + H_R_plusminus
    H_Lz = S_L_kappa + H_L_free[:,:,np.newaxis] + H_L_minusplus
    append_to_hdf5_scalar(outputfile, "H_Rz(eV)", H_Rz)
    append_to_hdf5_scalar(outputfile, "H_Lz(eV)", H_Lz)

    # close the output file
    outputfile.close()
    return
         
#<KeysViewHDF5 ['Fx00_Re', 'Fx00_Rebar', 'Fx01_Imbar', 'Fx01_Re', 'Fx01_Rebar', 'Fx11_Re', 'Fx11_Rebar', 'Fy00_Re', 'Fy00_Rebar', 'Fy01_Imbar', 'Fy01_Re', 'Fy01_Rebar', 'Fy11_Re', 'Fy11_Rebar', 'Fz00_Re', 'Fz00_Rebar', 'Fz01_Imbar', 'Fz01_Re', 'Fz01_Rebar', 'Fz11_Re', 'Fz11_Rebar', 'N00_Re', 'N00_Rebar', 'N01_Imbar', 'N01_Re', 'N01_Rebar', 'N11_Re', 'N11_Rebar', 'dz(cm)', 'it', 't(s)']>
         
def interact_old(d, outputfilename, basis_theta, basis_phi):
    # Read in the data
    
         
    eds = emu.EmuDataset(d)
    nz = eds.Nz

    # open the hdf5 file
    outputfile = h5py.File(outputfilename, "a")

    t = eds.ds.current_time
    append_to_hdf5(outputfile,"t(s)",t)
    
    z = np.arange(eds.dz/2., nz*eds.dz, eds.dz)
   
    # write the free Hamiltonians
    if "H_R_free(eV)" not in outputfile:
            outputfile["H_R_free(eV)"] = H_R_free
            outputfile["H_L_free(eV)"] = H_L_free

    # write the z grid
    if "z(cm)" not in outputfile:
            outputfile["z(cm)"] = z
    
    # [spacetime component, f1, f2, z]
    #particle, antiparticle and total neutrino four currents respectively
    J_p = four_current_eds(eds)[0]
    J_a = four_current_eds(eds)[1] 
    J = J_p-np.conj(J_a)
    
    append_to_hdf5(outputfile, "J_p(eV^3)", J_p)
    append_to_hdf5(outputfile, "J_a(eV^3)", J_a)
    append_to_hdf5(outputfile, "J(eV^3)", J)
    
    # [spacetime, f1, f2, z]
    S_R,S_L=sigma(J)
    append_to_hdf5(outputfile, "S_R(eV)", S_R)
    append_to_hdf5(outputfile, "S_L(eV)", S_L)

    # define the basis as along z
    basis = Basis(basis_theta,basis_phi)
    
    # precompute Sigma [f1, f2, z]
    S_R_plus = plus(S_R, basis)
    S_L_plus = plus(S_L, basis)
    S_R_minus = minus(S_R, basis)
    S_L_minus = minus(S_L, basis)
    S_R_kappa = kappa(S_R, basis)
    S_L_kappa = kappa(S_L, basis)
    append_to_hdf5(outputfile, "S_R_plus(eV)", S_R_plus)
    append_to_hdf5(outputfile, "S_L_plus(eV)", S_L_plus)
    append_to_hdf5(outputfile, "S_R_minus(eV)", S_R_minus)
    append_to_hdf5(outputfile, "S_L_minus(eV)", S_L_minus)
    append_to_hdf5(outputfile, "S_R_kappa(eV)", S_R_kappa)
    append_to_hdf5(outputfile, "S_L_kappa(eV)", S_L_kappa)
    
    ## Helicity-Flip Hamiltonian! ## [f1, f2, z]
    MSl = np.array([ np.matmul(conj(M),S_L_minus[:,:,n]) for n in range(nz) ])
    SrM = np.array([ np.matmul(S_R_plus[:,:,n],conj(M))  for n in range(nz) ])
    H_LR = (-1/p_abs)*(SrM-MSl)
    H_LR = H_LR.transpose((1,2,0))
    append_to_hdf5(outputfile, "H_LR(eV)", H_LR)    
    
    # plusminus term [f1, f2, z]
    H_R_plusminus = 2./p_abs * np.array([
            np.matmul(S_R_plus[:,:,z], S_R_minus[:,:,z])
            for z in range(nz)]).transpose((1,2,0))
    H_L_minusplus = 2./p_abs * np.array([
            np.matmul(S_L_minus[:,:,z], S_L_plus[:,:,z])
            for z in range(nz)]).transpose((1,2,0))
    append_to_hdf5(outputfile, "H_R_plusminus(eV)", H_R_plusminus)
    append_to_hdf5(outputfile, "H_L_minusplus(eV)", H_L_minusplus)
    
    ##H_R/H_L in the (0,0,10**7) basis (derivatives along x1 and x2 are 0 for 1d setup)
    H_Rz = S_R_kappa + H_R_free[:,:,np.newaxis] + H_R_plusminus
    H_Lz = S_L_kappa + H_L_free[:,:,np.newaxis] + H_L_minusplus
    append_to_hdf5(outputfile, "H_Rz(eV)", H_Rz)
    append_to_hdf5(outputfile, "H_Lz(eV)", H_Lz)

    # close the output file
    outputfile.close()
    return


 #different operation for if the dataset d is an h5py file, which has the time dependence already encoded in the objects (d is the location of the h5 file, time is the timestep (integer) at which we process the data)
    
def calculator_h5(d, outputfilename, basis_theta, basis_phi, time):
    
    # Read in the data
    data = extract_time(d, time)
    
    nz = np.shape(data['N00_Re(1|ccm)'])[0]
    
    t = data['t(s)']
    
    z = np.arange(data['dz(cm)']/2., nz*data['dz(cm)'], data['dz(cm)'])
    
    # [spacetime component, f1, f2, z]
    #particle, antiparticle and total neutrino four currents respectively
    J_p = four_current_h5(data)[0]
    J_a = four_current_h5(data)[1] 
    J = J_p-np.conj(J_a)
    
    nF=np.shape(J)[1]
    
    return nz, t, z, J_p, J_a, J, nF
    

def calculator_eds(d, outputfilename, basis_theta, basis_phi):
    
    # Read in the data
    eds = emu.EmuDataset(d)
   
    nz = eds.Nz

    t = eds.ds.current_time
    
    z = np.arange(eds.dz/2., nz*eds.dz, eds.dz)
    
    # [spacetime component, f1, f2, z]
    #particle, antiparticle and total neutrino four currents respectively
    J_p = four_current_eds(eds)[0]
    J_a = four_current_eds(eds)[1] 
    J = J_p-np.conj(J_a)   
    
    nF = np.shape(J)[1]
    
    return nz, t, z, J_p, J_a, J, nF
    

def interact(d, outputfilename, basis_theta, basis_phi, time=0):
    #extract the data, depending on type
    if d[-3:]=='.h5':
        nz, t, z, J_p, J_a, J, nF = calculator_h5(d, outputfilename, basis_theta, basis_phi, time)
    else:
        nz, t, z, J_p, J_a, J, nF = calculator_eds(d, outputfilename, basis_theta, basis_phi)
        
    # open the hdf5 destination file
    outputfile = h5py.File(outputfilename, "a")
    
    #write the time
    append_to_hdf5(outputfile,"t(s)",t)
    
    #define the size of the mass matrices
    if nF == 2:
        M=M_2flavor
    elif nF == 3:
        M=M_3flavor
    else:
        print("unsupported flavor number.")
        return "unsupported flavor number."
        
    
    # write the free Hamiltonians
    if "H_R_free(eV)" not in outputfile:
            outputfile["H_R_free(eV)"] = H_R_free(M)
            outputfile["H_L_free(eV)"] = H_L_free(M)
            
    # write the z grid
    if "z(cm)" not in outputfile:
            outputfile["z(cm)"] = z
            
    #write the fluxes
    append_to_hdf5(outputfile, "J_p(eV^3)", J_p)
    append_to_hdf5(outputfile, "J_a(eV^3)", J_a)
    append_to_hdf5(outputfile, "J(eV^3)", J)
    
    # [spacetime, f1, f2, z]
    S_R,S_L=sigma(J)
    append_to_hdf5(outputfile, "S_R(eV)", S_R)
    append_to_hdf5(outputfile, "S_L(eV)", S_L)

    # define the basis as along z
    basis = Basis(basis_theta,basis_phi)
    
    # precompute Sigma [f1, f2, z]
    S_R_plus = plus(S_R, basis)
    S_L_plus = plus(S_L, basis)
    S_R_minus = minus(S_R, basis)
    S_L_minus = minus(S_L, basis)
    S_R_kappa = kappa(S_R, basis)
    S_L_kappa = kappa(S_L, basis)
    append_to_hdf5(outputfile, "S_R_plus(eV)", S_R_plus)
    append_to_hdf5(outputfile, "S_L_plus(eV)", S_L_plus)
    append_to_hdf5(outputfile, "S_R_minus(eV)", S_R_minus)
    append_to_hdf5(outputfile, "S_L_minus(eV)", S_L_minus)
    append_to_hdf5(outputfile, "S_R_kappa(eV)", S_R_kappa)
    append_to_hdf5(outputfile, "S_L_kappa(eV)", S_L_kappa)
    
    ## Helicity-Flip Hamiltonian! ## [f1, f2, z]
    MSl = np.array([ np.matmul(conj(M),S_L_minus[:,:,n]) for n in range(nz) ])
    SrM = np.array([ np.matmul(S_R_plus[:,:,n],conj(M))  for n in range(nz) ])
    H_LR = (-1/p_abs)*(SrM-MSl)
    H_LR = H_LR.transpose((1,2,0))
    append_to_hdf5(outputfile, "H_LR(eV)", H_LR)    
    
    # plusminus term [f1, f2, z]
    H_R_plusminus = 2./p_abs * np.array([
            np.matmul(S_R_plus[:,:,z], S_R_minus[:,:,z])
            for z in range(nz)]).transpose((1,2,0))
    H_L_minusplus = 2./p_abs * np.array([
            np.matmul(S_L_minus[:,:,z], S_L_plus[:,:,z])
            for z in range(nz)]).transpose((1,2,0))
    append_to_hdf5(outputfile, "H_R_plusminus(eV)", H_R_plusminus)
    append_to_hdf5(outputfile, "H_L_minusplus(eV)", H_L_minusplus)
    
    ##H_R/H_L in the (0,0,10**7) basis (derivatives along x1 and x2 are 0 for 1d setup)
    H_Rz = S_R_kappa + H_R_free(M)[:,:,np.newaxis] + H_R_plusminus
    H_Lz = S_L_kappa + H_L_free(M)[:,:,np.newaxis] + H_L_minusplus
    append_to_hdf5(outputfile, "H_Rz(eV)", H_Rz)
    append_to_hdf5(outputfile, "H_Lz(eV)", H_Lz)

    # close the output file
    outputfile.close()

    return
            
           

         
