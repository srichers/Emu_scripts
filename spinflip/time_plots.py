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
 
        self.time_axis = np.array(self.h5file["t(s)"])[it_lim[0]:it_lim[1]+1]*1E9 #ns
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
        

    def plot(self, quantity, avg_method = 'GM', theta=0, phi=0, savefig = False):

        direction = [np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]
        f, ax = plt.subplots()

        if quantity == 'J_spatial': 
            J = self.J_spatial_flavoravg(avg_method)
            J_directional_projection = np.array([np.dot(J_at_t,direction) for J_at_t in J])

            plt.semilogy(self.time_axis,J_directional_projection)

            ax.set_ylabel(r"$eV^3$")

        elif quantity == 'J_time': 
            J = self.J_time_flavoravg(avg_method)
            J_directional_projection = np.array([np.dot(J_at_t,direction) for J_at_t in J])

            plt.semilogy(self.time_axis,J_directional_projection)

            ax.set_ylabel(r"$eV^3$")
            
        elif quantity == 'H_LR':
            H_LR=self.H_LR(avg_method, theta, phi)

            plt.semilogy(self.time_axis,H_LR)            

            ax.set_ylabel(r"$|H_{LR}| \ \ (eV)$")


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
        plt.tick_params(axis='both',which="both", direction="in",top=True,right=True)
        plt.minorticks_on()


        if savefig: 
            plt.savefig(savefig +'.png', dpi=300)

