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


###### Parameters ######
########################

#fermi coupling constant: G/(c*hbar)^3=1.166 378 7(6)×10−5 GeV−2 --> G=1.166 378 7×10−23 eV^−2 (natural units:c=hbar=1)
pi=np.pi
G=1.1663787*10**(-23)
c=29979245800
hbar=6.582119569e-16
#mixing angles (rad): (different values on wikipedia?)
a12=1e-6*np.pi*2/360
a13=48.3*np.pi*2/360
a23=8.61*np.pi*2/360

#CP phase:
delta=222
#majorana angles are all 0 and dont influence the matrix

#masses (eV) (Negative mass? 0 mass?)
m_1=0.608596511
m_2=0.608
m_3=0.649487372

#basis with momentum along +z (c=1?)


def basis(theta,phi): #theta is polar, phi is azimuthal
	global n_vector
	global x1
	global x2	
	n_vector=np.array([1,np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)])	
	x1=np.array([0,np.cos(phi)*np.cos(theta),np.sin(phi)*np.cos(theta),(-1)*np.sin(theta)])
	x2=np.array([0,-np.sin(phi),np.cos(phi),0])
	return n_vector,x1,x2
basis(0,0)
#running this for some pair of angles changes the lightlike(kappa) component of the basis.

#test neutrino momentum:
p_abs=10**7#eV

nz=1024 #length of box in z direction (nx=ny=1)

########################
########################


#??????
#### Variable definer: running this changes all quantities to the var's you want
def redefine(theta, phi, p_abs):
	global H_LR
	basis(theta,phi)	
	H_LR=1j*np.zeros(np.shape(plus(S_R))) #(3,3,nz)
	for n in range(0,nz):
		MSl=np.matmul(conj(M),plus(S_L)[:,:,n])
		SrM=np.matmul(plus(S_R)[:,:,n],conj(M))
		H_LR[:,:,n]=(-1/p_abs)*(SrM-MSl)
	
	
	return
		






#### Functions ####
###################

#unitary trace matrix
def trace_matrix(data):#takes in (3,3,nz)
	matrix=np.zeros((data.shape[1],data.shape[1],data.shape[2]))
	for n in range(0,data.shape[1]):
		matrix[n,n,:]=np.ones((data.shape[2]))
	for k in range(0,data.shape[2]):
		trace=0
		for n in range(0,data.shape[1]):
			trace=trace+data[n,n,k]
		matrix[:,:,k]=matrix[:,:,k]*trace
	return matrix
	
#scalar trace 	
def trace(data):
	trace=np.zeros((data.shape[2]))
	for n in range(0,data.shape[0]):
		trace=trace+data[n,n,:]
	return trace
	
#conjugate a matrix
def conj(matrix):
	conjugate=np.transpose(np.conjugate(matrix))
	return conjugate
	
#z-derivative of (3,3,nz) matrix (returns 3,3,nz)
def der(data,ad):
	dq=ad['index','dz'].d
	shape=np.shape(data)
	der=1j*np.zeros(shape)
	for n in range(0,shape[2]):
		der[:,:,n]=(data[:,:,n]-(1+0j)*data[:,:,n-1])/dq[n]
	return der
		
def cos(x):
	return np.cos(x)	
def sin(x):
	return np.sin(x)
def e(x):
	return 2.718281828**(x*1j)	

#Gell-Mann matrices (for scalarfunc)
GM=np.array([[[0,1,0],[1,0,0],[0,0,0*1j]],
             [[0,-1j,0],[1j,0,0],[0,0,0*1j]],
             [[1,0,0],[0,-1,0],[0,0,0*1j]],
             [[0,0,1],[0,0,0],[1,0,0*1j]],
             [[0,0,-1j],[0,0,0],[1j,0,0*1j]],
             [[0,0,0],[0,0,1],[0,1,0*1j]],
             [[0,0,0],[0,0,-1j],[0,1j,0*1j]],
             [[3**(-1/2),0,0],[0,3**(-1/2),0],[0*1j,0,-2*3**(-1/2)]]])

#scalarfunc: averages square magnitude of components for every location in nz and returns a list of these
def scalarfunc(array): #3,3
    scalars=1j*np.zeros(nz)
    for n in range(nz):
        components=1j*np.zeros(8)
        for k in range(0,8):
            components[k]=np.trace(np.matmul(GM[k],array[:,:,n]))
        scalars[n]=(1/(2**(1/2))+0*1j)*(sum([(x*conj(x)) for x in components]))**(1/2)
    return scalars

#scalar: works on a single matrix (k,k), outputs positive number 	
def scalar(matrix): #3,3
    components=1j*np.zeros(8)
    for k in range(0,8):
        components[k]=1/2*np.trace(np.matmul(GM[k],matrix))
    return (1/9)*sum([(x*conj(x))**(1/2) for x in components])

#average: for one timestep, takes in #3,3,nz and outputs average value of scalarfunc over space
def scalar_avg(array):
    return sum(scalarfunc(array))/nz
#same purpose as scalarfunc but just squares every elt in the matrix and averages
def old_scalarfunc(array): #takes in (k,k,nz), outputs real (nz)
	nz=array.shape[2]
	scalar=np.zeros(nz)*1j
	for k in range(0,nz):
		for n in np.nditer(array[:,:,k]):
			scalar[k]=n*(conj(n))+scalar[k]
		scalar[k]=scalar[k]**(1/2)
	scalar=np.real(scalar)
	return scalar
def old_scalar(array): 
	scalar=0
	for n in np.nditer(H_free[:,:]):
		scalar=n*conj(n)+scalar
	scalar=np.real(scalar**(1/2))
	return scalar

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
def sigma(flux):
	Sigma_R=np.zeros(np.shape(flux))
	Sigma_L=np.zeros(np.shape(flux))
	for n in range(0,4):
		Sigma_R[n]=2**(1./2.)*G*(flux[n]+trace_matrix(flux[n]))
		for k in range(0,nz): 
			Sigma_L[n,:,:,k]=(-1)*np.transpose(Sigma_R[n,:,:,k]) #for majorana 
	return Sigma_R, Sigma_L

#potential projected onto the basis
def dot(potential,vector):
	projection=np.zeros(np.shape(potential[0]))
	for k in range(0,4):
		projection=projection+vector[k]*potential[k]
	return projection

def plus(potential): #(3,3,nz)
	vector=0.5*(x1+1j*x2)
	plus=dot(potential,vector)
	return plus

def minus(potential): #(3,3,nz)
	vector=0.5*(x1-1j*x2)
	minus=dot(potential,vector)
	return minus
	
def kappa(potential):
	return dot(potential,n_vector)
	
## Mass Matrix ##	
m23=np.array([[1,0*1j,0],[0,cos(a23),sin(a23)],[0,-sin(a23),cos(a23)]])
m13=np.array([[cos(a13),0,sin(a13)*e(-delta)],[0,1,0],[-sin(a13)*e(delta),0,cos(a13)]])
m12=np.array([[cos(a12),sin(a12),0],[-sin(a12),cos(a12),0],[0,0*1j,1]])
m=np.matmul(m23,m13,m12)
#m is the mass mixing (MNS) matrix--I think what the paper wants is a matrix M that evaluates the mass of the particle
M_mass_basis=([[m_1,0*1j,0],[0,m_2,0],[0,0,m_3]])
M1=np.matmul(m,M_mass_basis)
M=np.matmul(M1,conj(m)) #(3,3)
	
    

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



# USAGE: python3 interact_with_data.py
# must be run from within the folder that contains the folder d defined below
# 92 particles correspond to 92 possible directions. They're always moving at the same speed
#physical neutrinos don't change direction
#cells in velocity space

# Four current, indexed by [time, nu/antinu, spacetime component, f1, f2, z]
def four_current(d):
    # get data
    eds = emu.EmuDataset(d)
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
    J     = np.array([N,    Fx,    Fy,    Fz    ])
    JI    = np.array([NI,   FxI,   FyI,   FzI   ])
    Jbar  = np.array([Nbar, Fxbar, Fybar, Fzbar ])
    JIbar = np.array([NIbar,FxIbar,FyIbar,FzIbar])

    Jeverything = (c*hbar)**3 * np.array([J + 1j*JI, Jbar + 1j*JIbar])

    return Jeverything

# Input: what folder do we want to process?
def interact(d):
    Jeverything = four_current(d)
    J     = np.real(J[0])
    JI    = np.imag(J[0])
    Jbar  = np.real(J[1])
    JIbar = np.imag(J[1])

    SigmaR,SigmaL=sigma(J)
    SigmaRI,SigmaLI=sigma(JI)
    SigmaRbar,SigmaLbar=sigma(Jbar)
    SigmaRIbar,SigmaLIbar=sigma(JIbar)

    S_R=SigmaR+1j*SigmaRI #(4,3,3,nz)
    S_L=SigmaL+1j*SigmaLI
    S_Rbar=SigmaRbar+1j*SigmaRIbar
    S_Lbar=SigmaLbar+1j*SigmaLIbar

    ## Helicity-Flip Hamiltonian! ##
    H_LR=1j*np.zeros(np.shape(plus(S_R))) #(3,3,nz)
    for n in range(0,nz):
        MSl=np.matmul(conj(M),plus(S_L)[:,:,n])
        SrM=np.matmul(plus(S_R)[:,:,n],conj(M))
        H_LR[:,:,n]=(-1/p_abs)*(SrM-MSl)

    ## Non-Interacting Term ##
    H_free=0.5*(1/p_abs)*np.matmul(conj(M),M) #For H_R; H_L has the m and m^dagger flipped
    H_free_func=scalar(H_free)*np.ones((1024))

    ##H_R/H_L in the (0,0,10**7) basis (derivatives along x1 and x2 are 0 for 1d setup)

    cross_term_z=1j*np.zeros(np.shape(plus(S_R)))
    for n in range(0,nz):
        cross_term_z[:,:,n]=np.matmul(plus(S_R)[:,:,n],minus(S_R)[:,:,n])

    H_Rz=1j*np.zeros(np.shape(plus(S_R))) #(3,3,nz)
    for n in range(0,np.shape(S_R)[3]):
        H_Rz[:,:,n]=kappa(S_R)[:,:,n]+0.5*(1/p_abs)*(np.matmul(conj(M),M)+4*np.matmul(plus(S_R)[:,:,n],minus(S_R)[:,:,n]))

    H_Lz=1j*np.zeros(np.shape(plus(S_R))) #(3,3,nz)
    for n in range(0,np.shape(S_R)[3]):
        H_Lz[:,:,n]=kappa(S_L)[:,:,n]+0.5*(1/p_abs)*(np.matmul(conj(M),M)+4*np.matmul(plus(S_L)[:,:,n],minus(S_L)[:,:,n]))

        
    return H_LR, cross_term_z, H_Rz



