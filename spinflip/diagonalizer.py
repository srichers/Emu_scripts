import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from constants import hbar

#This class takes a constant hamiltonian and evolves density matrices, generates plots
class Diagonalizer:
    def __init__(self, H):            
        #get Hamiltonian (use units eV)
        self.H = H
        self.Hsize = self.H.shape[1]
        
        #list of eigenvalues 
        self.eigenvals = (1+0*1j)*np.real(np.linalg.eig(self.H)[0] )
        
        #find timescale for plot
        self.eigenval_differences = np.array([abs(np.real(lambda_1) - np.real(lambda_2)) for lambda_1 in self.eigenvals
                                      for lambda_2 in self.eigenvals if lambda_1!=lambda_2]).flatten()
        self.timescale = max(2*np.pi*hbar/self.eigenval_differences)
        
    
        #(inverted) array of normalized eigenvectors
        #a.k.a. change of basis matrix from Energy to flavor/spin
        #ket_e = f_to_e(ket_f)
        #H_f = (f_to_e)^-1 H_e (f_to_e)
        self.f_to_e = (1+0*1j)*np.linalg.inv(np.linalg.eig(self.H)[1]) 
        
    #Time evolution operator in energy basis
    def U_energy(self, t):
        return np.diag([np.exp(-1j*eigenvalue*t/hbar) for eigenvalue in self.eigenvals])
   
    def U_flavor(self, t):
        return np.linalg.inv(self.f_to_e) @ self.U_energy(t) @ self.f_to_e
   
    def timescales(self, minmax = True):
        eigenvals = np.real(np.linalg.eig(self.H)[0])
        differences = np.array([[abs(eig1 - eig2) 
                                 for eig1 in eigenvals  if eig2 != eig1] 
                                for eig2 in eigenvals]
                              ).flatten()
        if minmax == True:
            print('Minimum Timescale: '+str(hbar/max(differences))+' s')
            print('Maximum Timescale: '+str(hbar/min(differences))+' s')
        else:
            return hbar/differences
        
        #time array of neutrino density matrix, with init_array as intial condition
    def state_evolution(self, resolution, t_lim, init_array):
        if t_lim == 'timescale':
           t_lim = self.timescale
           print('Largest timescale = '+str(t_lim)+ ' s')
        return np.array([self.U_flavor(t) @ init_array @ np.linalg.inv(self.U_flavor(t)) for t in np.linspace(0, t_lim, resolution)])
    
    #calculates the largest component of initial_ket_f and returns the component and the corresponding eigenvector
    def largest_ket_component(self, init_ket_f):
        components = self.f_to_e @ init_ket_f
        return np.max(np.abs(components)), (1+0j)*self.f_to_e[np.argmax(components)]
    
        
    def state_evolution_plotter(self, t_lim = 'timescale', resolution=500, quantity = 'state_right', ylim = None, init_array = np.diag((1,0,0,0,0,0)), savefig = False):
        if t_lim == 'timescale':
           t_lim = self.timescale
        
        flavornum = self.Hsize//2
        #s_vs_t.shape = t,2nf,2nf
        state_vs_time = np.real(self.state_evolution(resolution, t_lim, init_array))
        
        state_left  = np.trace(state_vs_time[:,0:flavornum,0:flavornum], axis1= 1, axis2 = 2)
        state_right = np.trace(state_vs_time[:,flavornum:2*flavornum,flavornum:2*flavornum], axis1= 1, axis2 = 2)
        left_minus_right = state_left - state_right
        
        f, ax = plt.subplots()
        ax.set_ylabel('diagonal elements')
        ax.set_xlabel('s')
        
        if ylim != None:
            ax.set_ylim(0,ylim)
        
        
        if type(quantity) == type([]):
            nth_diagonal = np.array([state_vs_time[:,n,n] for n in np.arange(0,2*flavornum)])
            for n in quantity:
                ax.plot(np.linspace(0,t_lim,resolution),nth_diagonal[n], label = str(n))
        elif quantity == 'state_left':
                ax.plot(np.linspace(0,t_lim,resolution),state_left, label = 'Left-handed trace')
        elif quantity == 'state_right':
                ax.plot(np.linspace(0,t_lim,resolution),state_right, label = 'Right-handed trace')
        elif quantity == 'left_minus_right':
                ax.plot(np.linspace(0,t_lim,resolution),left_minus_right, label = 'Left minus Right')

        plt.legend()
        
        
        if savefig == True: 
            plt.tight_layout()
            plt.savefig('evolvedstate.png', dpi=300)
        
        f.show()

    
