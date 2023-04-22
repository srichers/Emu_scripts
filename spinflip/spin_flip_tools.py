#!/usr/bin/env python
# coding: utf-8

#CHDIR COMMAND ONLY FOR JUPYTER
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../data_reduction")
import numpy as np
import emu_yt_module as emu
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import h5py
import gellmann as gm
import glob
import concurrent
import matplotlib as mpl
from multiprocessing import Pool
from constants import p_abs, M_p, hbar, c, G, M_3flavor, M_2flavor
from diagonalizer import Diagonalizer
from basis import Basis
from matrix import visualizer, dagger, trace_matrix
from four_current import calculator_h5, calculator_eds, four_current_h5, four_current_eds
from spin_params import SpinParams, sigma


#takes in value that depends on theta, phi and returns a theta_res by phi_res array of values 
def angularArray(func, theta_res, phi_res):   
    return np.array([[func(theta, phi) for phi in np.linspace(0, 2*np.pi, phi_res)]
                                      for theta in np.linspace(0, np.pi, theta_res)])
        
#generates plots vs time of spin parameters in fast flavor instability simulation
class TimePlots:
    def __init__(self, data_loc,  merger_data_loc, location,
           p_abs=10**7):
        
        self.precision = 1
        
        self.data_loc = data_loc
        self.h5file = h5py.File(self.data_loc, "r")
 
        self.time_axis = np.array(self.h5file["t(s)"])*1E9 #ns
        self.nt = self.time_axis.shape[0]-10
    
        self.spin_params_timearray = [SpinParams(t_sim= t, data_loc=data_loc, p_abs=p_abs, location=location, merger_data_loc=merger_data_loc) for t in np.arange(0,self.nt,self.precision)]
        
    #(spacetime, F, F, z)
    def J_spatial_flavoravg(self, avg_method = 'GM'): 
        if avg_method == 'GM':
            return np.array([[gm.magnitude(np.average(SPclass.J[n], axis = 2)) 
                              for n in range(1,4)] for SPclass in self.spin_params_timearray])
        elif avg_method == 'sum':
            return np.array([[gm.sum_magnitude(np.average(SPclass.J[n], axis = 2)) 
                              for n in range(1,4)] for SPclass in self.spin_params_timearray])

    def J_time_flavoravg(self, avg_method = 'GM'): 
        if avg_method == 'GM':
            return np.array([gm.magnitude(np.average(SPclass.J[0], axis = 2)) 
                               for SPclass in self.spin_params_timearray])
        elif avg_method == 'sum':
            return np.array([gm.sum_magnitude(np.average(SPclass.J[0], axis = 2)) 
                              for SPclass in self.spin_params_timearray])
        
    def H_LR(self, avg_method, theta, phi):
        if avg_method == 'GM':
            return np.array([gm.magnitude(SPclass.H_LR(theta,phi)) 
                               for SPclass in self.spin_params_timearray])
        elif avg_method == 'sum':
            return np.array([gm.sum_magnitude(SPclass.H_LR(theta,phi)) 
                               for SPclass in self.spin_params_timearray])
        

    def plot(self, quantity, avg_method = 'GM', theta=0, phi=0, set_title = None, savefig = False):

        direction = [np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]
        f, ax = plt.subplots()

        if quantity == 'J_spatial': 
            J = self.J_spatial_flavoravg(avg_method)
            J_directional_projection = np.array([np.dot(J_at_t,direction) for J_at_t in J])
            plt.semilogy(np.arange(0,self.nt,self.precision),J_directional_projection)
            ax.set_ylabel(r"$eV^3$")
            ax.set_title(r'Directional Component of $J_\mu$ vs time')

        elif quantity == 'J_time': 
            J = self.J_time_flavoravg(avg_method)
            J_directional_projection = np.array([np.dot(J_at_t,direction) for J_at_t in J])
            plt.semilogy(np.arange(0,self.nt,self.precision),J_directional_projection)
            ax.set_ylabel(r"$eV^3$")
            ax.set_title(r'Time Component of $J_\mu$ vs time')

            
        elif quantity == 'H_LR':
            H_LR=self.H_LR(avg_method, theta, phi)
            plt.semilogy(np.arange(0,self.nt,self.precision),H_LR)            
            ax.set_ylabel(r"$|H_{LR}| \ \ (eV)$")
            ax.set_title(r'$|H_{LR}|$ vs time')


            #NOT ADAPTED THESE YET
  #      elif quantity == 'H_LR_00':
  #          H_LR=total(readdata,"H_LR(eV)")
  #          ax.set_ylabel(r"$|H_{LR}| \ \ (eV)$")
  #          plt.semilogy(t_axis*1E9,np.average(H_LR[:,0,0,:], axis=1), label = 'electron component')
  #          plt.semilogy(t_axis*1E9,np.average(H_LR[:,1,1,:], axis=1), label = 'muon component')
  #          plt.semilogy(t_axis*1E9,np.average(H_LR[:,2,2,:], axis=1), label = 'tau component')
  #          ax.legend()
  #          ax.set_ylim(1E-25,1E-15)
            
  #      elif  quantity == 'kappa':
  #          kappa=total(data,'S_R_kappa(eV)')
  #          ax.set_ylabel(r"$eV$")
  #         plt.plot(t*1E9,np.average(kappa[:,0,0,:], axis=1))

  
        ax.set_xlabel("time (ns)")
        mpl.pyplot.tick_params(axis='both',which="both", direction="in",top=True,right=True)
        mpl.pyplot.minorticks_on()


        if savefig == True: 
            plt.tight_layout()
            plt.savefig(quantity+'.png', dpi=300)

        else: pass

        
        

        
#Calculates values from merger data, and generates adiabaticity / resonance contour plots.
#data_loc, unrotated_data_loc is the location for the merger data and unrotated merger data
class Merger_Grid:
    def __init__(self, zval, data_loc, 
                unrotated_data_loc,
                p_abs=10**7, theta=0, phi=0,
                rotate = True):
        
        self.zval = zval
            
        #rotated data has ELN flux along z
        if rotate == True:
            self.merger_grid = h5py.File(data_loc, 'r')
        elif rotated == False:
            self.merger_grid = h5py.File(unrotated_data_loc, 'r')
        else:
            raise TypeError('rotate = True or False')
        
        #Electron fraction, baryon n density
        self.rho=np.array(self.merger_grid['rho(g|ccm)'])[:,:,self.zval] #g/cm^3 (baryon mass density)
        self.Ye=np.array(self.merger_grid['Ye'])[:,:,self.zval]
        self.n_b=self.rho/M_p*(hbar**3 * c**3)#eV^3 (baryon number density)

        #x,y axes
        self.x_km = np.array(self.merger_grid['x(cm)'])[:,:,0] / 1e5
        self.y_km = np.array(self.merger_grid['y(cm)'])[:,:,0] / 1e5
       
        
        #basis
        self.basis = Basis(theta,phi)
        #momentum
        self.p_abs = p_abs
        
        #discriminant condition
        self.discriminant = np.array(self.merger_grid['crossing_discriminant'])[:,:,self.zval]
        self.positive = np.where(self.discriminant>=0)
        self.negative = np.where(self.discriminant<0)
        self.discriminant_sign=np.zeros_like(self.discriminant)
        
        #positive areas have ELN crossings
        self.discriminant_sign[self.positive] = 1
        self.discriminant_sign[self.negative] = 0
        
        self.J_e = (hbar**3 * c**3)*(np.array([self.merger_grid['n_e(1|ccm)'],
                                             self.merger_grid['fn_e(1|ccm)'][0],
                                             self.merger_grid['fn_e(1|ccm)'][1],
                                             self.merger_grid['fn_e(1|ccm)'][2]]))[:,:,:,self.zval]
                      
        self.J_a = (hbar**3 * c**3)*np.array([self.merger_grid['n_a(1|ccm)'],
                                              self.merger_grid['fn_a(1|ccm)'][0],
                                              self.merger_grid['fn_a(1|ccm)'][1],
                                              self.merger_grid['fn_a(1|ccm)'][2]])[:,:,:,self.zval]
        
        #technically just the ee component of J
        self.J = self.J_e - self.J_a
        
        #max, min values of the LHS of the resonance condition (ADD OTHER PARTS)
        self.mag_F=np.sqrt(np.sum(self.J[1:4,:,:]**2, axis=0))
        self.H_vv_min = 2**(1./2.)*G*(self.J[0]+self.mag_F)
        self.H_vv_max = 2**(1./2.)*G*(self.J[0]-self.mag_F)
        #min occurs when flux is along basis direction
        self.resonance_val_min = 2**(-1/2)*G*self.n_b*(3.*self.Ye-np.ones_like(self.Ye))+self.H_vv_max
        self.resonance_val_max = 2**(-1/2)*G*self.n_b*(3.*self.Ye-np.ones_like(self.Ye))+self.H_vv_min

        #conditions
        #max res >0 so could be 0 for some direction
        self.positive_resonance = np.where(self.resonance_val_max>=0)
        self.pos_resonance_sign=np.zeros_like(self.discriminant)
        self.pos_resonance_sign[self.positive_resonance]=1
        #min res <0 so could be 0 for some direction
        self.negative_resonance = np.where(self.resonance_val_min<=0)
        self.neg_resonance_sign=np.zeros_like(self.discriminant)
        self.neg_resonance_sign[self.positive_resonance]=1
        
        #both must be true for resonance to exist
        self.resonance_sign = self.neg_resonance_sign*self.pos_resonance_sign
        
        #resonance and discriminant satisfied for some direction
        self.both_conditions=self.resonance_sign*self.discriminant_sign
    
        #self.res_along_J = np.where(self.resonance_val_max>=0)
        
    #Basis dependent stuff
    #returns the value of the resonance condition with the given basis at level zval
    def resonance_val(self, basis):
        S_R = 2**(1./2.)*G*np.transpose(np.array([[2*self.J,np.zeros_like(self.J)],[np.zeros_like(self.J),self.J]]),(2,0,1,3,4))
        S_R_kappa = basis.kappa(S_R)

        return np.array(2**(-1/2)*G*self.n_b*(3.*self.Ye-np.ones_like(self.Ye))+S_R_kappa[0,0])

    #gets the gradient (has to get data for z+1 and z-1 too)
    def grad_magnitude(self, basis):
        J_e = (hbar**3 * c**3)*(np.array([self.merger_grid['n_e(1|ccm)'],
                                             self.merger_grid['fn_e(1|ccm)'][0],
                                             self.merger_grid['fn_e(1|ccm)'][1],
                                             self.merger_grid['fn_e(1|ccm)'][2]]))[:,:,:,(self.zval-1):(self.zval+1)]
                      
        J_a = (hbar**3 * c**3)*np.array([self.merger_grid['n_a(1|ccm)'],
                                              self.merger_grid['fn_a(1|ccm)'][0],
                                              self.merger_grid['fn_a(1|ccm)'][1],
                                              self.merger_grid['fn_a(1|ccm)'][2]])[:,:,:,(self.zval-1):(self.zval+1)]
        
        #technically just the ee component of J
        J = J_e - J_a
        
        S_R = 2**(1./2.)*G*np.transpose(np.array([[2*J,np.zeros_like(J)],[np.zeros_like(J),J]]),(2,0,1,3,4,5))
        S_R_kappa = basis.kappa(S_R)
        
        resonance = np.array(2**(-1/2)*G*self.n_b[:,:,np.newaxis]*(3.*self.Ye[:,:,np.newaxis]-np.ones_like(self.Ye[:,:,np.newaxis]))+S_R_kappa[0,0])

        grad=np.gradient(resonance, 0.67928832684*1e5/(c*hbar))
        projection=np.sum(basis.n_vector[1:4,np.newaxis,np.newaxis,np.newaxis]*grad, axis=0)

        return np.array(np.abs(projection))[:,:,1]

    #gets H_LR
    def H_LRf(self, basis):
        M=M_2flavor
        
        S_R = 2**(1./2.)*G*np.transpose(np.array([[2*self.J,np.zeros_like(self.J)],
                                                  [np.zeros_like(self.J),self.J]]),(2,0,1,3,4))
        S_L = -2**(1./2.)*G*np.transpose(np.array([[2*self.J,np.zeros_like(self.J)],
                                                   [np.zeros_like(self.J),self.J]]),(2,1,0,3,4))
        S_L_minus = basis.minus(S_L)
        S_R_plus = basis.plus(S_R)

        MSl = np.transpose(np.array([[ np.matmul(np.conj(M),S_L_minus[:,:,x,y]) 
                                                                                   for y in range(201)]
                                                                                   for x in range(201)]),(2,3,0,1))
        
        SrM = np.transpose(np.array([[ np.matmul(S_R_plus[:,:,x,y],np.conj(M))  
                                                                                   for y in range(201)]
                                                                                   for x in range(201)]) ,(2,3,0,1))
        return (-1/self.p_abs)*(SrM-MSl)
    
    #gets full H
    def H(self, basis):
        return
    #gets gamma
    def adiabaticity(self, basis):
        return np.real((self.H_LRf(basis)[0,0]**2)/self.grad_magnitude(basis))


        #gets indices of points where adiab is greater than value at z
        #
    def max_adiab(self,adiab,value):
        maxvals=np.where(adiab>value)
        coordlist=[]
        for n in range(0,len(maxvals[1])):
            coordlist.append([int(maxvals[0][n]),int(maxvals[1][n])])
        return np.array(coordlist)

    #gets the coordinates of the points of adiab higher than value (for plotting)
    def coords(self,adiab,value):
        maxpoints=self.max_adiab(adiab,value)
        xlist=[self.x_km[val[0],val[1]] for val in maxpoints]
        ylist=[self.y_km[val[0],val[1]] for val in maxpoints]
        return xlist,ylist
    
    
    def contour_plot(self, savefig = False, x = 0, y = 0, xmin = 0, xmax = 200, ymin = 0, ymax = 200):
        zval = self.zval
        
        f, ax = plt.subplots()
        
        #discriminant area
        ax.contourf(self.x_km[:,:],self.y_km[:,:], self.discriminant_sign[:,:],
                                   levels=[0.1,1,2,3], alpha=0.6, colors=['blue','red',[0.65,0,0.25]])
        
        #legend
        proxy = [plt.Rectangle((1, 1), 2, 2, fc='blue'),plt.Rectangle((1, 1), 2, 2, fc='red'),plt.Rectangle((1, 1), 2, 2, fc=[0.7,0,0.5],ec='yellow')]
        ax.legend(proxy, ["ELN Crossing Present", "Resonance Condition Satisfied","Both Conditions Satisfied"])
        
        #resonance val
        ax.contour(self.x_km[:,:],self.y_km[:,:],self.resonance_val_min[:,:], levels=[0,1], colors=['green'])
        ax.contour(self.x_km[:,:],self.y_km[:,:],self.resonance_val_max[:,:], levels=[0,1], colors=['red'])

        #both conditions
        ax.contourf(self.x_km[:,:],self.y_km[:,:],self.both_conditions[:,:], levels=[0.5,1], colors=['yellow'] )
        
        mpl.pyplot.tick_params(axis='both',which="both", direction="in",top=True,right=True)
        mpl.pyplot.minorticks_on()
        ax.set_xlabel('x (km)')
        ax.set_ylabel('y (km)')
        plt.scatter(self.x_km[x,y],self.y_km[x,y], color = 'red')
        
        ax.set_ylim(self.y_km[xmin,ymin],self.y_km[xmax,ymax])
        ax.set_xlim(self.x_km[xmin,ymin],self.x_km[xmax,ymax])
        ax.add_patch(plt.Rectangle((self.x_km[xmin,ymin],self.y_km[xmin,ymin]),self.x_km[xmax,ymax]-self.x_km[xmin,ymin],self.y_km[xmax,ymax]-self.y_km[xmin,ymin],
                             linewidth=1,edgecolor='magenta',facecolor='none'))
        #plt.contour(x[:,:,zval],y[:,:,zval],resonance_val_max[:,:,zval], levels=[0], colors=['blue'] )
        #plt.title('Resonance Conditions and ELN Crossing Regions')
        if savefig ==True:
            plt.savefig('contour_plot_'+str(zval)+'.png', dpi=300)
        else:
            pass

    #Adiabaticity Colorplot
    def adiab_plot(self, minval = 1E-5, crit_region = False, savefig = False, theta_res = np.pi, phi_res = 0, ):
        zval = self.zval
        adiabaticity = self.adiabaticity(self.basis)
        plt.grid(False)
        colorplot=plt.pcolormesh(self.x_km[:,:],self.y_km[:,:],np.log10(adiabaticity[:,:]),cmap=plt.cm.hot,shading='auto', vmax=-3, vmin=-10)
        
        plt.xlabel('x (km)')
        plt.ylabel('y (km)')
        plt.colorbar(label=r'$log(\gamma)$')
        
        if crit_region == True:
            contourf=plt.contourf(self.x_km[:,:],self.y_km[:,:],self.both_conditions[:,:], alpha=0.5, levels=[0.5,1], colors=['blue'] )
            proxy = [plt.Rectangle((1, 1), 2, 2, fc=pc.get_facecolor()[0]) for pc in
            contourf.collections]
        
        plt.contour(self.x_km[:,:],self.y_km[:,:],self.resonance_val(Basis(theta_res,phi_res)), levels=0)
        #mpl.pyplot.tick_params(axis='both',which="both", direction="in",top=True,right=True)
        mpl.pyplot.minorticks_on()
        scatter=plt.scatter(self.coords(adiabaticity, minval)[0],self.coords(adiabaticity,minval)[1],color=[0,0.9,0], marker='x', label='Simulated Point')
        red_patch = mpatches.Patch(color=[0.3,0,0.7], label='The red data')
        plt.legend(handles=[scatter,red_patch], labels=["Simulated Point","Both Conditions Satisfied"], frameon = 1, framealpha=0.5)
        plt.tight_layout()
        plt.scatter(self.x_km[95,123],self.y_km[95,123])
        if savefig ==True:
            plt.savefig('adiabaticity_plot_'+str(zval)+'.png', dpi=300)
 


# i,j,k are the coordinate to generate plots from. xmin,xmax,ymin,ymax are the limits of the array of points.
#append is the end of the filenames
class Multipoint:
    def __init__(self, i, j, k, sfm_file,
                xmin, xmax, ymin, ymax,
                merger_data_loc, unrotated_merger_data_loc,
                append = 'sfmJ', savefig=False):
        self.sfm_file = sfm_file
        self.merger_data_loc = merger_data_loc
        self.unrotated_merger_data_loc = unrotated_merger_data_loc
        self.filelist = glob.glob(self.sfm_file + "/i*j*k*.h5")
        
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        
        self.i = i
        self.j = j
        self.k = k
        self.chosenfile = self.sfm_file + '/i' + f"{i:03d}" + "_j" + f"{j:03d}" + "_k" + f"{k:03d}" + append + ".h5"
        
        self.MG = Merger_Grid(self.k, self.merger_data_loc, self.unrotated_merger_data_loc)
        
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
        
        
        
        
        
	
#z-derivative of (nF,nF,nz) matrix (returns nF,nF,nz)
#TODO - needs to implement periodic boundary conditions.
def der(data,ad):
	dq=ad['index','dz'].d
	shape=np.shape(data)
	der=1j*np.zeros(shape)
	for n in range(0,shape[2]):
		der[:,:,n]=(data[:,:,n]-(1+0j)*data[:,:,n-1])/dq[n]
	return der

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


                                    
                
## Non-Interacting Term ## [f1, f2]

def H_R_free(M):
    return 0.5*(1/p_abs)*np.matmul(M,dagger(M))
def H_L_free(M):
    return 0.5*(1/p_abs)*np.matmul(M,dagger(M))

H_R_free_3flavor = H_R_free(M_3flavor)
H_L_free_3flavor = H_L_free(M_3flavor)


#version of interact that just outputs the flux, from which all other quantities can be calculated, to save storage space.

def interact_J(d, outputfilename, time=0):
    #extract the data, depending on type
    if d[-3:]=='.h5':
        nz, t, z, J_p, J_a, J, nF = calculator_h5(d, outputfilename, time)
    else:
        nz, t, z, J_p, J_a, J, nF = calculator_eds(d, outputfilename)
        
    # open the hdf5 destination file
    outputfile = h5py.File(outputfilename, "a")
    
    #write the time
    append_to_hdf5(outputfile,"t(s)",t)
     
    # write the z grid
    if "z(cm)" not in outputfile:
            outputfile["z(cm)"] = z
            
    #write the fluxes
    append_to_hdf5(outputfile, "J_p(eV^3)", J_p)
    append_to_hdf5(outputfile, "J_a(eV^3)", J_a)
    append_to_hdf5(outputfile, "J(eV^3)", J)
    
     
    # close the output file
    outputfile.close()

    return
            
         
        
        
        
    
def interact(d, outputfilename, basis_theta, basis_phi, time=0):
    #extract the data, depending on type
    if d[-3:]=='.h5':
        nz, t, z, J_p, J_a, J, nF = calculator_h5(d, outputfilename, time)
    else:
        nz, t, z, J_p, J_a, J, nF = calculator_eds(d, outputfilename)
        
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
    S_R_plus = basis.plus(S_R)
    S_L_plus = basis.plus(S_L)
    S_R_minus = basis.minus(S_R)
    S_L_minus = basis.minus(S_L)
    S_R_kappa = basis.kappa(S_R)
    S_L_kappa = basis.kappa(S_L)
    append_to_hdf5(outputfile, "S_R_plus(eV)", S_R_plus)
    append_to_hdf5(outputfile, "S_L_plus(eV)", S_L_plus)
    append_to_hdf5(outputfile, "S_R_minus(eV)", S_R_minus)
    append_to_hdf5(outputfile, "S_L_minus(eV)", S_L_minus)
    append_to_hdf5(outputfile, "S_R_kappa(eV)", S_R_kappa)
    append_to_hdf5(outputfile, "S_L_kappa(eV)", S_L_kappa)
    
    ## Helicity-Flip Hamiltonian! ## [f1, f2, z]
    MSl = np.array([ np.matmul(dagger(M),S_L_plus[:,:,n]) for n in range(nz) ])
    SrM = np.array([ np.matmul(S_R_plus[:,:,n],dagger(M))  for n in range(nz) ])
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
    
  
    # close the output file
    outputfile.close()

    return
            
         
        
        
        
    
    
    
# Input: what folder do we want to process?
def interact_scalar(d, outputfilename, basis_theta, basis_phi, time=0):
    # Read in the data
    eds = emu.EmuDataset(d)
    nz = eds.Nz

    #extract the data, depending on type
    if d[-3:]=='.h5':
        nz, t, z, J_p, J_a, J, nF = calculator_h5(d, outputfilename, time)
    else:
        nz, t, z, J_p, J_a, J, nF = calculator_eds(d, outputfilename)
        
    #define the size of the mass matrices
    if nF == 2:
        M=M_2flavor
    elif nF == 3:
        M=M_3flavor
    else:
        print("unsupported flavor number.")
        return "unsupported flavor number."

    # open the hdf5 file
    outputfile = h5py.File(outputfilename, "a")

    t = eds.ds.current_time
    append_to_hdf5(outputfile,"t(s)",t)

    # write the free Hamiltonians
    if "H_R_free(eV)" not in outputfile:
            outputfile["H_R_free(eV)"] = H_R_free(M)
            outputfile["H_L_free(eV)"] = H_L_free(M)

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
    S_R_plus = basis.plus(S_R)
    S_L_plus = basis.plus(S_L)
    S_R_minus = basis.minus(S_R)
    S_L_minus = basis.minus(S_L)
    S_R_kappa = basis.kappa(S_R)
    S_L_kappa = basis.kappa(S_L)
    append_to_hdf5_scalar(outputfile, "S_R_plus(eV)", S_R_plus)
    append_to_hdf5_scalar(outputfile, "S_L_plus(eV)", S_L_plus)
    append_to_hdf5_scalar(outputfile, "S_R_minus(eV)", S_R_minus)
    append_to_hdf5_scalar(outputfile, "S_L_minus(eV)", S_L_minus)
    append_to_hdf5_scalar(outputfile, "S_R_kappa(eV)", S_R_kappa)
    append_to_hdf5_scalar(outputfile, "S_L_kappa(eV)", S_L_kappa)
    
    ## Helicity-Flip Hamiltonian ## [f1, f2, z]
    MSl = np.array([ np.matmul(dagger(M),S_L_plus[:,:,n]) for n in range(nz) ])
    SrM = np.array([ np.matmul(S_R_plus[:,:,n],dagger(M))  for n in range(nz) ])
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
         

        
#data_base_directory is the .h5 file with the raw simulation data in it
#output_name is the name of the output if you want a specific one, otherwise it gives it the same name as the input h5 file and appends output_append at the end (so ijk.h5 becomes ijk_sfm.h5 by default)
class Interact:
    def __init__(self, data_base_directory, output_name = None):
        self.data_base_directory = data_base_directory
        
        output_append = '_sfm'
        if output_name == None:
            self.output_filename = data_base_directory[0:-3] + output_append
        else: 
            self.output_filename = output_name + output_append
    
    def is_h5(self):
        if self.data_base_directory[-3:]=='.h5':
            return True
        else: 
            return False
    
    #old interact function (computes everything that SpinParams computes and stores all of it instead of just storing J. theta, phi specify the direction for computations of , for example, \Sigma_\kappa
    def run(self, anglename, theta=0, phi=0,):
      
        if self.is_h5() == True:      
        #runs interact for h5 files, outputs [filename]_spin_flip_matrices.h5 in same location
       
        #find number of time intervals
            File=h5py.File(self.data_base_directory,"r")
            nt=len(np.array(File["t(s)"]))
            File.close()

            for t in range(0,nt):
                interact(self.data_base_directory, self.output_filename + anglename + '.h5', theta, phi, t)
        
        else: #runs interact, for plt files. outputs h5 'spin_flip_matrices' file inside of data_base_directory
            directory_list = sorted(glob.glob(self.data_base_directory+"plt*"))
           
            if os.path.exists(self.output_filename):
                os.remove(self.output_filename)

            for d in directory_list:
                print(d)
                with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
                    executor.submit(interact, d, self.output_filename + anglename + '.h5', theta, phi)
    
    #interact_J, which just outputs the flux to save space
    def run_J(self):
      
        if self.is_h5() == True:      
        #runs interact for h5 files, outputs [filename]_spin_flip_matrices.h5 in same location
       
        #find number of time intervals
            File=h5py.File(self.data_base_directory,"r")
            nt=len(np.array(File["t(s)"]))
            File.close()

            for t in range(0,nt):
                interact_J(self.data_base_directory, self.output_filename+'J.h5', t)
        
        else: #runs interact, for plt files. outputs h5 'spin_flip_matrices' file inside of data_base_directory
            directory_list = sorted(glob.glob(self.data_base_directory+"plt*"))
           
            if os.path.exists(self.output_filename):
                os.remove(self.output_filename)

            for d in directory_list:
                print(d)
                with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
                    executor.submit(interact_J, d, self.output_filename)
          
        
    
        
#inputpath carries files of the form /i*j*k*/allData.h5. outputpath is the file where you want to store the output h5 files. output_append is an appended string to the name of the individual h5 files inside outputpath; default is _sfm (so files inside outputpath have filename i*j*k*_sfm.h5)
class Multipoint_interact:
    def __init__(self, inputpath, outputpath):
        self.inputpath = inputpath
        self.filelist = glob.glob(self.inputpath + "/i*j*k*/allData.h5")
        self.outputpath = outputpath
        
    def run_single_interact(self,h5file):
        coords = h5file[-26:-11] #just the coordinate part
        Interact(h5file, output_name = self.outputpath + coords).run_J()
         
    def run_many_interact(self):
        os.mkdir(self.outputpath)

        with Pool() as p:
             p.map(self.run_single_interact,self.filelist)
        
               
    
