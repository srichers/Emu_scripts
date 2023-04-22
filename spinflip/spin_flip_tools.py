#!/usr/bin/env python
# coding: utf-8

#CHDIR COMMAND ONLY FOR JUPYTER
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../data_reduction")
import numpy as np
import h5py
import glob
from multiprocessing import Pool
from constants import p_abs, M_3flavor, M_2flavor
from diagonalizer import Diagonalizer
from matrix import visualizer
from constants import c, hbar
from merger_grid import Merger_Grid
from spin_params import SpinParams


# i,j,k are the coordinate to generate plots from. xmin,xmax,ymin,ymax are the limits of the array of points.
#append is the end of the filenames
class Multipoint:
    def __init__(self, i, j, k, sfm_file,
                xmin, xmax, ymin, ymax,
                merger_data_loc,
                append = '_sfmJ', savefig=False):
        self.sfm_file = sfm_file
        self.merger_data_loc = merger_data_loc
        self.filelist = glob.glob(self.sfm_file + "/i*j*k*.h5")
        
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        
        self.i = i
        self.j = j
        self.k = k
        self.chosenfile = self.sfm_file + '/i' + f"{i:03d}" + "_j" + f"{j:03d}" + "_k" + f"{k:03d}" + append + ".h5"
        
        self.MG = Merger_Grid(self.k, self.merger_data_loc)
        
    def angularPlot(self, t, savefig=False):
        SP = SpinParams(t_sim = t, data_loc = self.chosenfile, merger_data_loc = self.merger_data_loc, location = [self.i,self.j,self.k])
        SP.angularPlot( theta_res = 100, phi_res = 100, use_gm=True, direction_point=False, savefig=savefig)
    def pointPlots(self, t, plot_tlim='timescale', savefig=False):
        self.MG.contour_plot(x = self.i, y = self.j, xmin = self.xmin, xmax = self.xmax, ymin = self.ymin, ymax = self.ymax)
        SP = SpinParams(t_sim = t, data_loc = self.chosenfile, merger_data_loc = self.merger_data_loc, location = [self.i,self.j,self.k])
        SP.angularPlot( theta_res = 50, phi_res = 50, use_gm=True, direction_point=False, savefig=savefig)
        H_resonant = SP.resonant_Hamiltonian()
        D = Diagonalizer(H_resonant)

        D.state_evolution_plotter(plot_tlim, init_array = np.diag((1,0,0,0,0,0)),savefig=savefig)
        visualizer(H_resonant, traceless=True, savefig=savefig)

#returns J, Jbar like the above function but this one works on h5 files (takes in the dictionary outputed by extract)
#keys must be of the form ['Fx00_Re', 'Fx00_Rebar', 'Fx01_Imbar', ... 'N00_Re', 'N00_Rebar', ... 'dz(cm)', 'it', 't(s)']>
#where the numbers denote flavor components       
#number of flavors is variable. 
#Returns (4, nF, nF, nz)
def four_current(h5_dict):
    num_flavors=max([int(key[2]) for key in list(h5_dict.keys()) if key[0]=='F'])+1
    component_shape=np.shape(h5_dict['N00_Re(1|ccm)'])
    components=['N', 'Fx', 'Fy', 'Fz']

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
    
    #make sure J_Im is antisymmetric so J is Hermitian
    for i in range(0,num_flavors):
        for j in range(0,i):
            J_Im_bar[:,i,j,:] = -J_Im_bar[:,i,j,:]
            J_Im[    :,i,j,:] = -J_Im[    :,i,j,:]

    #store complex numbers
    J    = (1+0*1j)*J_Re     + 1j*J_Im
    Jbar = (1+0*1j)*J_Re_bar + 1j*J_Im_bar

    # rearrange the order of indices so that the time index is first
    J    = np.transpose(J,    (3,0,1,2,4))
    Jbar = np.transpose(Jbar, (3,0,1,2,4))

    return (c**3*hbar**3)*J, (c**3*hbar**3)*Jbar


#data_base_directory is the .h5 file with the raw simulation data in it
#output_name is the name of the output if you want a specific one, otherwise it gives it the same name as the input h5 file and appends output_append at the end (so ijk.h5 becomes ijk_sfm.h5 by default)
#interact_J, which just outputs the flux to save space
#inputpath carries files of the form /i*j*k*/allData.h5. outputpath is the file where you want to store the output h5 files. output_append is an appended string to the name of the individual h5 files inside outputpath; default is _sfm (so files inside outputpath have filename i*j*k*_sfm.h5)
class Multipoint_interact:
    def __init__(self, inputpath, outputpath):
        self.outputpath = outputpath
        self.filelist = glob.glob(inputpath + "/i*j*k*/allData.h5")

    def run_single(self,infilename):
        coords = infilename[-26:-11] #just the coordinate part
        outputfilename = self.outputpath + "/"+ coords + '_sfmJ.h5'

        #runs interact for h5 files, outputs [filename]_spin_flip_matrices.h5 in same location
        # Read in the data
        infile = h5py.File(infilename, "r")
        data={key:np.array(infile[key]) for key in list(infile.keys())}
        infile.close()

        # get coordinages, times
        nz = np.shape(data['N00_Re(1|ccm)'])[1]
        z = np.arange(data['dz(cm)']/2., nz*data['dz(cm)'], data['dz(cm)'])
        t = data['t(s)']
    
        # [spacetime component, f1, f2, z]
        #particle, antiparticle and total neutrino four currents respectively
        J_ab = four_current(data)
        J_p = J_ab[0]
        J_a = J_ab[1]
        J = J_p-np.conj(J_a)

        #write the time, fluxes, and tetrad
        outputfile = h5py.File(outputfilename, "w")
        outputfile["t(s)"      ] = t
        outputfile["z(cm)"     ] = z
        outputfile["J_p(eV^3)R"] = np.real(J_p)
        outputfile["J_p(eV^3)I"] = np.imag(J_p)
        outputfile["J_a(eV^3)R"] = np.real(J_a)
        outputfile["J_a(eV^3)I"] = np.imag(J_a)
        outputfile["J(eV^3)R"  ] = np.real(J)
        outputfile["J(eV^3)I"  ] = np.imag(J)
        outputfile.close()



    def run_many(self):
        os.mkdir(self.outputpath)
        with Pool() as p:
             p.map(self.run_single,self.filelist)
