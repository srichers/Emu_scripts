from constants import hbar, c, M_p, M_3flavor, G
import h5py
import numpy as np
from matrix import trace_matrix, dagger
from basis import Basis
import matplotlib.pyplot as plt
import gellmann as gm
from scipy import optimize as opt

#total takes in read-only h5 dataset and the array you want to compute and just adds the real and imaginary parts for the array
def total(d, array):
    return (1+0*1j)*np.array(d[array+'R'])+1j*np.array(d[array+'I'])

def J(d):
    return total(d,'J(eV^3)')

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

#Given a spinflip dataset, finds the hamiltonian at some angle. Can also check resonant direction
#t_sim is ffi simulation timestep to calculate parameters at
#data_loc is the spinflip file to compute
#merger_data_loc is the merger grid data location
#location is where in the merger data to evaluate stuff like ye, rho (=[x,y,z])
class SpinParams:
    def __init__(self, t_sim, data_loc, merger_data_loc, location,
                 p_abs=10**7):
        
        self.data_loc = data_loc
        self.h5file = h5py.File(self.data_loc, "r")
 
        self.p_abs = p_abs
        self.t_sim = t_sim
         
        #Grid-dependent stuff: Electron fraction, baryon n density
        self.location=location
          
        self.merger_grid = h5py.File(merger_data_loc, 'r')
        self.rho = np.array(self.merger_grid['rho(g|ccm)'])[location[0],location[1],location[2]] #g/cm^3 (baryon mass density)
        self.Ye = np.array(self.merger_grid['Ye'])[location[0],location[1],location[2]]
        self.n_b = self.rho/M_p*(hbar**3 * c**3)#eV^3 (baryon number density)
    
            
        #Flux (spacetime, F, F, z)
        self.J = total(self.h5file, 'J(eV^3)')[self.t_sim]
       
        #length of 1d array 
        self.nz = self.J.shape[3]
        
        #neutrino part of Sigma
        self.S_R_nu = sigma(self.J)[0]
        self.S_L_nu = sigma(self.J)[1]
        
        #matter part of Sigma
        self.S_R_mat = np.zeros(np.shape(self.J))  
        for k in np.arange(0, self.nz):
            self.S_R_mat[0,:,:,k] = 2**(-1/2)*G*self.n_b*np.array([[3*self.Ye-1,    0,      0],
                                              [0,           self.Ye-1, 0],
                                              [0,              0,   self.Ye-1 ]])
        self.S_L_mat = (-1)*np.transpose(self.S_R_mat, axes=(0,2,1,3))   
        
        #Total Sigma
        self.S_R = self.S_R_nu + self.S_R_mat
        self.S_L = self.S_L_nu + self.S_R_mat
        
        #Mass part
        self.M = M_3flavor
        self.H_vac = 1/(2*self.p_abs)*np.matmul(self.M,dagger(self.M))
        
        

    def S_L_kappa(self, theta, phi):
        basis = Basis(theta,phi)
        return np.average(basis.kappa(self.S_L), axis = 2)

    def S_R_kappa(self, theta, phi):
        basis = Basis(theta,phi)
        return np.average(basis.kappa(self.S_R), axis = 2)

    def H_L_pm(self, theta, phi):
        basis = Basis(theta,phi)
        S_L_minus = basis.minus(self.S_L)
        S_L_plus = basis.plus(self.S_L)
        H_L_pm = 2./self.p_abs * np.array([np.matmul(S_L_minus[:,:,z], S_L_plus[:,:,z])
            for z in range(self.nz)]).transpose((1,2,0))
        return np.average(H_L_pm, axis=2)

    def H_R_pm(self, theta, phi):
        basis = Basis(theta,phi)
        S_R_minus = basis.minus(self.S_R)
        S_R_plus = basis.plus(self.S_R)
        H_R_pm = 2./self.p_abs * np.array([np.matmul(S_R_plus[:,:,z], S_R_minus[:,:,z])
            for z in range(self.nz)]).transpose((1,2,0))
        return  np.average(H_R_pm, axis = 2)


    #NO DERIVATIVE TERM
    def H_L(self, theta, phi):
        basis = Basis(theta,phi)
        return self.S_L_kappa(theta, phi) + self.H_vac + self.H_L_pm(theta, phi)

    def H_R(self, theta, phi):
        basis = Basis(theta,phi)
        return self.S_R_kappa(theta, phi) + self.H_vac + self.H_R_pm(theta, phi)

    def H_LR(self, theta, phi):          
        basis = Basis(theta, phi)
        S_L_plus = np.average(basis.plus(self.S_L), axis = 2)
        S_R_plus = np.average(basis.plus(self.S_R), axis = 2)
        
       # MSl = np.array([ np.matmul(conj(M),S_L_plus[:,:,n]) for n in range(nz) ])
       # SrM = np.array([ np.matmul(S_R_plus[:,:,n],conj(M))  for n in range(nz) ])
        MSl = np.array(np.matmul(dagger(self.M),S_L_plus))
        SrM = np.array(np.matmul(S_R_plus,dagger(self.M)))
        return (-1/self.p_abs)*(SrM-MSl)


    #full Hamiltonian
    def H(self, theta, phi):
        return np.concatenate((np.concatenate( (self.H_R(theta, phi), np.conjugate(self.H_LR(theta,phi).transpose(1,0))), axis=0),
                np.concatenate((self.H_LR(theta,phi), self.H_L(theta,phi)), axis = 0)), axis = 1)

    #NEED DEFINITION OF DENSITY MATRIX
    def P(self, theta, phi):
        
        return np.diag((1,0,0,0,0,0))

    def resonance_full(self, theta, phi):
        return gm.dotprod(self.H(theta,phi),self.P(theta,phi))
    
    #checks resonance condition [H_L]_f1f1 = [H_R]_f2f2
    #equals resonance from tian et al if f1=f2=0 (electron)
    def resonance_2f(self, theta, phi, f1, f2):
        return self.H(theta,phi)[f1,f1]-self.H(theta,phi)[f2,f2]

    def resonance_old(self, theta, phi):
        return np.real(2**(-1/2)*G*self.n_b*(3.*self.Ye-np.ones_like(self.Ye))+self.S_R_kappa(theta,phi)[0,0])

    def angularPlot(self, theta_res, phi_res, savefig=False, use_gm=True, direction_point=False):
        
        if use_gm==True:
            H_LR_array = np.array([[np.abs(np.trace(self.H_LR(theta, phi))) 
                                   for phi in np.linspace(0, 2*np.pi, phi_res)]
                                   for theta in np.linspace(0, np.pi, theta_res)])
        else: 
            H_LR_array = np.array([[gm.sum_magnitude(self.H_LR(theta, phi)) 
                                   for phi in np.linspace(0, 2*np.pi, phi_res)]
                                   for theta in np.linspace(0, np.pi, theta_res)])

        
        resonance_array = np.array([[self.resonance_old(theta,phi)
                                   for phi in np.linspace(0, 2*np.pi, phi_res)]
                                   for theta in np.linspace(0, np.pi, theta_res)]) 

        f = plt.figure()
        ax = f.add_subplot(projection = 'mollweide')
        ax.grid(False)
        #ax.set_title(r'Angular plot of $H_{LR}$')
        
        H_LR_im = ax.pcolormesh(np.linspace(-np.pi, np.pi, phi_res), 
                                np.linspace(0.5*np.pi, -0.5*np.pi, theta_res),
                                H_LR_array, 
                                cmap=plt.cm.hot, shading='auto')

        plt.colorbar(H_LR_im, label="eV")

        #resonance 
        res_im = ax.contour(np.linspace(-np.pi, np.pi, phi_res),
                            np.linspace(0.5*np.pi, -0.5*np.pi, theta_res),
                           resonance_array, levels=[0.], colors='cyan', label = "Resonant Directions" )
        
        h1,l1 = res_im.legend_elements()

        #add net flux point 
        J_avg = np.array([gm.magnitude(np.average(self.J[n], axis = 2)) for n in range(0,4)])
        flux_point = ax.scatter([np.arctan2(J_avg[2],J_avg[1])],[np.arctan2(J_avg[3],
                                                (J_avg[1]**2+J_avg[2]**2)**(1/2))],  label = 'ELN Flux Direction', color='lime')
        
        #add (electron) neutrino direction point
        
        flow_direction = np.array(self.merger_grid['fn_a(1|ccm)'])[:,self.location[0],self.location[1],self.location[2]]
        direction_point = ax.scatter([np.arctan2(flow_direction[1],flow_direction[0])],[np.arctan2(flow_direction[2], (flow_direction[0]**2+flow_direction[1]**2)**(1/2))],  label = 'Neutrino Flow Direction', color='magenta')

        plt.legend([h1[0], flux_point, direction_point], ["Resonant Directions", "Number Flux Direction", "Polar Direction"], loc=(0, 1.13))
            
        if savefig == True: 
            plt.tight_layout()
            plt.savefig('angularPlot.png', dpi=300)
        
    #uses scipy rootfinder to locate polar angle of resonance contour. Assumes rotational symmetry (picks phi=0)
    #Currently only works for the old resonance condition
    
    #uses scipy rootfinder to locate polar angle of resonance contour. Assumes rotational symmetry (picks phi=0)
    #Currently only works for the old resonance condition
    
    def J_avg(self): 
        return np.array([gm.sum_magnitude(np.average(self.J[n], axis = 2)) for n in range(0,4)])
    
    def resonant_theta(self, phi=0):
        res_function = self.resonance_old
        return opt.bisect(res_function,0,np.pi,args = (phi))
    
    #resonant Hamiltionian at azimuthal angle phi (should be independent of phi)
    def resonant_Hamiltonian(self, phi=0):
        theta = self.resonant_theta(phi)
        return self.H(theta,phi)
    
    def H_array(self):
        return np.array([[gm.sum_magnitude(self.H_LR(theta, phi)) 
                                   for phi in np.linspace(0, 2*np.pi, 50)]
                                   for theta in np.linspace(0, np.pi, 50)])
