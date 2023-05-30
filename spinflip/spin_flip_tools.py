#!/usr/bin/env python
# coding: utf-8

#CHDIR COMMAND ONLY FOR JUPYTER
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../data_reduction")
import numpy as np
import h5py
from constants import hbar, c, M_p, M_3flavor, G
from diagonalizer import Diagonalizer
from matrix import visualizer
from merger_grid import Merger_Grid
from matrix import trace_matrix, dagger
from basis import Basis
import matplotlib.pyplot as plt
import gellmann as gm
from scipy import optimize as opt
from four_current import four_current, read_gradients


# i,j,k are the coordinate to generate plots from. xmin,xmax,ymin,ymax are the limits of the array of points.
#append is the end of the filenames
#resonance_type is either 'full' or 'simplified'. 'full' uses the full resonance condition, 'simplified' uses the simplified resonance condition
class MultiPlot:
    def __init__(self, i, j, k, emu_file,
                xmin, xmax, ymin, ymax,
                merger_data_loc, p_abs, gradient_filename = None, resonance_type = 'simplified', 
                density_matrix = np.diag([1,0,0,0,0,0])
                ):
        self.emu_file = emu_file        
        self.merger_data_loc = merger_data_loc
        self.gradient_filename = gradient_filename
        self.p_abs = p_abs
        
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        
        self.i = i
        self.j = j
        self.k = k

        self.resonance_type = resonance_type
        self.density_matrix = density_matrix

        self.MG = Merger_Grid(self.k, self.merger_data_loc, p_abs=p_abs)
    def resHamiltonian(self, t):
        SP = SpinParams(t_sim = t, resonance_type = self.resonance_type, density_matrix = self.density_matrix, emu_file = self.emu_file, merger_data_loc = self.merger_data_loc, gradient_filename = self.gradient_filename, location = [self.i,self.j,self.k], p_abs=self.p_abs)
        return SP.resonant_Hamiltonian()    
    
    def angularPlot(self, t, savefig=False):
        SP = SpinParams(t_sim = t, resonance_type = self.resonance_type, density_matrix = self.density_matrix, emu_file = self.emu_file, merger_data_loc = self.merger_data_loc, gradient_filename = self.gradient_filename, location = [self.i,self.j,self.k], p_abs=self.p_abs)
        SP.angularPlot( theta_res = 100, phi_res = 100, use_gm=True, direction_point=False, savefig=savefig)
    
    #init array is density matrix at t=0 evolved by diagonalizer. default is same as the density matrix used for the Resoannce condition
    #diagonalier_quantity is either 'state_right', 'state_left', or a list of integers up to dim(H). 'state_right' plots the right eigenvectors, 'state_left' plots the left eigenvectors, the list plots the eigenvectors with those indices.
    def pointPlots(self, t, plot_tlim='timescale', savefig=False, traceless = True,  text='mag', diagonalizer_quantity = 'state_right', init_array = None):
        if init_array == None:
            init_array =self.density_matrix
        self.MG.contour_plot(x = self.i, y = self.j, xmin = self.xmin, xmax = self.xmax, ymin = self.ymin, ymax = self.ymax)
        SP = SpinParams(t_sim = t, resonance_type = self.resonance_type, density_matrix = self.density_matrix, emu_file = self.emu_file, merger_data_loc = self.merger_data_loc, gradient_filename = self.gradient_filename, location = [self.i,self.j,self.k], p_abs=self.p_abs)
        SP.angularPlot( theta_res = 50, phi_res = 50, use_gm=True, direction_point=False, savefig=savefig)
        H_resonant = SP.resonant_Hamiltonian()
        D = Diagonalizer(H_resonant)

        D.state_evolution_plotter(plot_tlim, init_array = init_array,savefig=savefig, quantity = diagonalizer_quantity)
        visualizer(H_resonant, traceless=traceless, savefig=savefig, text = text)

## Chiral Potentials ##
# input: flux[spacetime, f1, f2, ...]
# output: Sigma_R[spacetime, f1, f2, ...]
def sigma(flux):
        Sigma_R=0j*np.zeros(np.shape(flux)) 
        Sigma_L=0j*np.zeros(np.shape(flux))
        for n in range(0,4):
                Sigma_R[n]=2**(1./2.)*G*(flux[n]+trace_matrix(flux[n]))
        transpose_axes = np.arange(Sigma_R.ndim)
        transpose_axes[1:3] = [2,1] #switches f1 and f2. input can have extra vars at the end like z, lower gradient indices
        Sigma_L=(-1)*np.transpose(Sigma_R, axes=transpose_axes) #for majorana 
        return Sigma_R, Sigma_L

#Given a spinflip dataset, finds the hamiltonian at some angle. Can also check resonant direction
#t_sim is ffi simulation timestep to calculate parameters at
#data_loc is the spinflip file to compute
#merger_data_loc is the merger grid data location
#location is where in the merger data to evaluate stuff like ye, rho (=[x,y,z])
class SpinParams:
    def __init__(self, t_sim, emu_file, merger_data_loc, location, p_abs, resonance_type = 'full', density_matrix =np.diag([1,0,0,0,0,0]), gradient_filename = None):
        
        self.p_abs = p_abs
        self.t_sim = t_sim
         
        #Grid-dependent stuff: Electron fraction, baryon n density
        self.location=np.array(location)
          
        self.merger_grid = h5py.File(merger_data_loc, 'r')
        self.rho = np.array(self.merger_grid['rho(g|ccm)'])[location[0],location[1],location[2]] #g/cm^3 (baryon mass density)
        self.Ye = np.array(self.merger_grid['Ye'])[location[0],location[1],location[2]]
        self.n_b = self.rho/M_p*(hbar**3 * c**3)#eV^3 (baryon number density)

        #Flux (spacetime, F, F, z)
        self.J = four_current(emu_file)[self.t_sim]

        #length of 1d array 
        self.nz = self.J.shape[3]
        
        #neutrino part of Sigma
        self.S_R_nu = sigma(self.J)[0]
        self.S_L_nu = sigma(self.J)[1]
        
        #matter part of Sigma
        self.S_R_mat = np.zeros(np.shape(self.J))  
        for k in np.arange(0, self.nz):
            self.S_R_mat[0,:,:,k] = -2**(-1/2)*G*self.n_b*np.array([[3*self.Ye-1,    0,      0],
                                              [0,           self.Ye-1, 0],
                                              [0,              0,   self.Ye-1 ]])
        self.S_L_mat = (-1)*np.transpose(self.S_R_mat, axes=(0,2,1,3))   
        
        #Total Sigma
        self.S_R = self.S_R_nu + self.S_R_mat
        self.S_L = self.S_L_nu + self.S_L_mat
        
        #Mass part
        self.M = M_3flavor
        self.H_vac = 1/(2*self.p_abs)*np.matmul(self.M,dagger(self.M))
        
        #Gradients
        if gradient_filename != None:
            # flux gradient (spacetime up, spacetime (gradient) low, F, F)
            all_gradJ, x, y, z, it, limits = read_gradients(gradient_filename)
            assert(t_sim == it)
            assert(location[0] >= limits[0,0] and location[0] <= limits[0,1]) # make sure the location is within the limits
            assert(location[1] >= limits[1,0] and location[1] <= limits[1,1])
            assert(location[2] >= limits[2,0] and location[2] <= limits[2,1])
            self.gradJ = all_gradJ[:,:,location[0]-limits[0,0], location[1]-limits[1,0], location[2]-limits[2,0]]
            # access the gradient at the location. Index 0 means at the lower limit.

            # S_R_nu gradient (spacetime up, spacetime (gradient) low, F, F) [tranpose gradJ so new lower index is last and the sigma function works properly, then transpose back]
            self.grad_S_R_nu, self.grad_S_L_nu = sigma(np.transpose(self.gradJ, axes = (0,2,3,1)))
            print(self.grad_S_R_nu.shape)
            self.grad_S_R_nu, self.grad_S_L_nu = np.transpose(self.grad_S_R_nu, axes = (0,3,1,2)), np.transpose(self.grad_S_L_nu, axes = (0,3,1,2))
            print(self.grad_S_R_nu.shape)
            #need matter part gradients from changing Y_e, n_b
            self.grad_S_R_mat, self.grad_S_L_mat = np.zeros(np.shape(self.grad_S_R_nu)), np.zeros(np.shape(self.grad_S_L_nu))
            self.grad_S_R, self.grad_S_L = self.grad_S_R_nu + self.grad_S_R_mat, self.grad_S_L_nu + self.grad_S_L_mat

            
        #Resonance, initial density matrix
        self.density_matrix = density_matrix
        self.resonance_type = resonance_type

    def S_L_kappa(self, theta, phi):
        basis = Basis(theta,phi)

        return np.average(basis.kappa(self.S_L), axis = 2)

    def S_R_kappa(self, theta, phi):
        basis = Basis(theta,phi)
        return np.average(basis.kappa(self.S_R), axis = 2)
    
    #plus & minus Potential term
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
    
    #derivative terms (don't account for zsim gradient, since its approximately negligible)
    def H_L_grad(self, theta, phi):
        basis = Basis(theta,phi)
        x1 = basis.x1
        x2 = basis.x2
        #grad_S_L is the Jacobian, indices are [spacetime_up, spacetime_down, F, F]
        # we want the derivative along x_i of component x_j of the gradient
        #given by x_j @ grad_S_L @ x^i

        #d_2 S_L ^1
        d2_S1 = np.tensordot( x1.T, np.tensordot(self.grad_S_L, x2, axes =([1],[0])), axes = ([0],[0]))
        #d_1 S_L ^2
        d1_S2 = np.tensordot( x2.T, np.tensordot(self.grad_S_L, x1, axes =([1],[0])), axes = ([0],[0]))
        
        return 1/(2*self.p_abs)*(d1_S2 - d2_S1)
        
    def H_R_grad(self, theta, phi):
        basis = Basis(theta,phi)
        x1 = basis.x1
        x2 = basis.x2
        #grad_S_L is the Jacobian, indices are [spacetime_up, spacetime_down, F, F]
        # we want the derivative along x_i of component x_j of the gradient
        #given by x_j @ grad_S_L @ x^i

        #d_2 S_L ^1
        d2_S1 = np.tensordot( x1.T, np.tensordot(self.grad_S_R, x2, axes =([1],[0])), axes = ([0],[0]))
        #d_1 S_L ^2
        d1_S2 = np.tensordot( x2.T, np.tensordot(self.grad_S_R, x1, axes =([1],[0])), axes = ([0],[0]))
        
        return - 1/(2*self.p_abs)*(d1_S2 - d2_S1)

    def H_L(self, theta, phi):
        H_L_nogradient = self.S_L_kappa(theta, phi) + self.H_vac + self.H_L_pm(theta, phi)
        if self.gradient_filename != None:
            return H_L_nogradient + self.H_L_grad(theta, phi)
        else:
            return H_L_nogradient
        

    def H_R(self, theta, phi):
        H_R_nogradient = self.S_R_kappa(theta, phi) + self.H_vac + self.H_R_pm(theta, phi)
        if self.gradient_filename != None:
            return H_R_nogradient + self.H_R_grad(theta, phi)
        else:
            return H_R_nogradient

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

        
    ## kappa component of just neutrino part of potential. Need for Resonance condition so matter contribution isnt counted twice
    def S_L_nu_kappa(self, theta, phi):
        basis = Basis(theta,phi)
        return np.average(basis.kappa(self.S_L_nu), axis = 2)

    def S_R_nu_kappa(self, theta, phi):
        basis = Basis(theta,phi)
        return np.average(basis.kappa(self.S_R_nu), axis = 2)
    
    def grad_S_L_nu_kappa(self, theta, phi):
        basis = Basis(theta,phi)
        return np.average(basis.kappa(self.grad_S_L_nu), axis = 2)

    def grad_S_R_nu_kappa(self, theta, phi):
        basis = Basis(theta,phi)
        return np.average(basis.kappa(self.grad_S_R_nu), axis = 2)
    
    def resonance(self, theta, phi):
        if self.resonance_type == 'full':
            return gm.dotprod(self.H(theta,phi),self.density_matrix)
        elif self.resonance_type == 'simplified':
            return np.real(2**(-1/2)*G*self.n_b*(3.*self.Ye-np.ones_like(self.Ye))+self.S_L_nu_kappa(theta,phi)[0,0])

   
    def angularPlot(self, theta_res, phi_res, savefig=False, use_gm=True, direction_point=False):
        
        if use_gm==True:
            H_LR_array = np.array([[np.abs(np.trace(self.H_LR(theta, phi))) 
                                   for phi in np.linspace(0, 2*np.pi, phi_res)]
                                   for theta in np.linspace(0, np.pi, theta_res)])
        else: 
            H_LR_array = np.array([[gm.sum_magnitude(self.H_LR(theta, phi)) 
                                   for phi in np.linspace(0, 2*np.pi, phi_res)]
                                   for theta in np.linspace(0, np.pi, theta_res)])

        
        resonance_array = np.array([[self.resonance(theta,phi)
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
                           resonance_array, levels=[0.], colors='cyan')
        
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
    
    
    def J_avg(self): 
        return np.array([gm.sum_magnitude(np.average(self.J[n], axis = 2)) for n in range(0,4)])
    
    def resonant_theta(self, phi=0):
        theta = opt.bisect(self.resonance,0,np.pi,args = (phi))
        return theta
    
    #resonant Hamiltionian at azimuthal angle phi (should be independent of phi)
    def resonant_Hamiltonian(self, phi=0):
        theta = self.resonant_theta(phi)
        return self.H(theta,phi)
    
    def H_array(self):
        return np.array([[gm.sum_magnitude(self.H_LR(theta, phi)) 
                                   for phi in np.linspace(0, 2*np.pi, 50)]
                                   for theta in np.linspace(0, np.pi, 50)])
