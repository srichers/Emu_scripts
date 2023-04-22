import numpy as np
import h5py
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../data_reduction")
import reduce_data as rd
import emu_yt_module as emu
from constants import c, hbar

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
                    

 #different operation for if the dataset d is an h5py file, which has the time dependence already encoded in the objects (d is the location of the h5 file, time is the timestep (integer) at which we process the data)
    
def calculator_h5(d, outputfilename, time):
    
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
    

def calculator_eds(d, outputfilename):
    
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
