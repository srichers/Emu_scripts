import matplotlib.pyplot as plt
import h5py
import gellmann as gm
from spin_flip_tools import SpinParams
import numpy as np

#generates plots vs time of spin parameters in fast flavor instability simulation
class TimePlots:
    def __init__(self,
                 it_lim, 
                 emu_data_loc,
                 merger_data_loc,
                 location,
                 p_abs, 
                 resonance_type = 'simplified',
                 initial_ket = np.array([1,0,0,0,0,0])):
                
        self.emu_data_loc = emu_data_loc
        self.h5file = h5py.File(self.emu_data_loc, "r")
 
        self.time_axis = np.array(self.h5file["t(s)"])[it_lim[0]:it_lim[1]]*1E9 #ns
        self.nt = self.time_axis.shape[0]
    
        self.spin_params_timearray = [SpinParams(t, 
                                                 emu_data_loc, 
                                                 merger_data_loc, 
                                                 location,
                                                 p_abs,
                                                 resonance_type = resonance_type, 
                                                 initial_ket = initial_ket)
                                                 for t in np.arange(it_lim[0],it_lim[1],1)]

        
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
        elif avg_method == None:
            return np.array([SPclass.H_LR(theta,phi) 
                               for SPclass in self.spin_params_timearray])
        
    #plots quantity vs time. thetas and phis should be a list of tuples, one per plot in the subplots.
    def plot(self, quantity, avg_method = 'GM', thetas_and_phis=[(0,0)], 
             labels = None, #list of labels for each subplot
             savefig = False
             ):

        directions = [[np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)] for theta, phi in thetas_and_phis]
        N = len(directions)
        f, ax = plt.subplots(N,1, sharex = True, squeeze = False, figsize = (8,N*5))
        for n, direction in enumerate(directions):
        
            if quantity == 'J_spatial': 
                J = self.J_spatial_flavoravg(avg_method)
                J_directional_projection = np.array([np.dot(J_at_t,direction) for J_at_t in J])
                ax[n,0].semilogy(self.time_axis,J_directional_projection)
                ax[0,0].set_ylabel(r"$eV^3$")

            elif quantity == 'J_time': 
                J = self.J_time_flavoravg(avg_method)
                J_directional_projection = np.array([np.dot(J_at_t,direction) for J_at_t in J])
                ax[n,0].semilogy(self.time_axis,J_directional_projection)
                ax[0,0].set_ylabel(r"$eV^3$")
                
            elif quantity == 'H_LR':
                theta, phi = thetas_and_phis[n]
                H_LR=self.H_LR(avg_method, theta, phi)
                ax[n,0].semilogy(self.time_axis,H_LR)            
                ax[0,0].set_ylabel(r"$|H_{LR}| \ \ (eV)$")

            elif quantity == 'H_LR_components' or quantity == 'H_LR_components_reduced' or quantity == 'H_LR_components_diagonals':
                flavor_labels = ['e',r'\mu',r'\tau']                
                theta, phi = thetas_and_phis[n]
                H_LR=self.H_LR(None, theta, phi)
                for k in range(len(flavor_labels)):
                    for m in range(len(flavor_labels)):
                        if m<=k: #dont do both e mu and mu e, for example
                           
                            if quantity == 'H_LR_components_reduced': #skip off diagonal tau components
                                if k==2 and m!=2:
                                    continue
                            elif quantity == 'H_LR_components_diagonals': #only do diagonal components
                                if k!=m:
                                    continue
                                
                            ax[n,0].semilogy(self.time_axis,np.abs(H_LR[:,k,m]), label = r'$H_{LR}^{'+rf'{flavor_labels[k]} {flavor_labels[m]}' +r'}$')
            
  
        #xlabel 
        ax[len(directions)-1,0].set_xlabel(r"Time ($ns$)")


        
        plt.tick_params(axis='both',which="both", direction="in",top=True,right=True)
        plt.minorticks_on()
        
        if type(labels)!= type(None):
            for n in range(len(directions)):
                ax[n,0].set_title(labels[n], fontsize = 20, loc = 'left')
        plt.tight_layout(pad = 0.45)
        
            
        #set y label for all plots, centered 
        f.text(-0.03, 0.55, r'$|H_{LR}^{ij}| (eV)$', va='center', rotation='vertical')
        #legend
        ax[0,0].legend(frameon = False, fontsize = 20, bbox_to_anchor = (1.01,1))
        
        if savefig: 
            plt.savefig(savefig +'.pdf', dpi=300, bbox_inches = 'tight')

