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
import diagonalizer as dg
from matrix import visualizer
from merger_grid import Merger_Grid
from matrix import trace_matrix, dagger
from basis import Basis
import matplotlib as mpl
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
                initial_ket = np.array([1,0,0,0,0,0])
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

        self.initial_ket = initial_ket
        self.density_matrix = np.outer(self.initial_ket, np.conjugate(self.initial_ket))
        self.resonance_type = resonance_type

        self.MG = Merger_Grid(self.k, self.merger_data_loc, p_abs=p_abs)
    def resHamiltonian(self, t):
        SP = SpinParams(t_sim = t, resonance_type = self.resonance_type, density_matrix = self.density_matrix, emu_file = self.emu_file, merger_data_loc = self.merger_data_loc, gradient_filename = self.gradient_filename, location = [self.i,self.j,self.k], p_abs=self.p_abs)
        return SP.resonant_Hamiltonian()    
    
    def angularPlot(self, t, savefig=False):
        SP = SpinParams(t_sim = t, resonance_type = self.resonance_type, density_matrix = self.density_matrix, emu_file = self.emu_file, merger_data_loc = self.merger_data_loc, gradient_filename = self.gradient_filename, location = [self.i,self.j,self.k], p_abs=self.p_abs)
        SP.angularPlot( theta_resolution = 100, phi_resolution = 100, use_gm=True, direction_point=False, savefig=savefig)
    
    #init array is density matrix at t=0 evolved by diagonalizer. default is same as the density matrix used for the Resonance condition
    #diagonalier_quantity is either 'state_right', 'state_left', or a list of integers up to dim(H). 'state_right' plots the right eigenvectors, 'state_left' plots the left eigenvectors, the list plots the eigenvectors with those indices.
    def pointPlots(self, t, plot_tlim='timescale', savefig=False, traceless = True,  text='mag', diagonalizer_quantity = 'state_right', init_array = None):
        if init_array == None:
            init_array =self.density_matrix
        self.MG.contour_plot(x = self.i, y = self.j, xmin = self.xmin, xmax = self.xmax, ymin = self.ymin, ymax = self.ymax)
        SP = SpinParams(t_sim = t, resonance_type = self.resonance_type, density_matrix = self.density_matrix, emu_file = self.emu_file, merger_data_loc = self.merger_data_loc, gradient_filename = self.gradient_filename, location = [self.i,self.j,self.k], p_abs=self.p_abs)
        SP.angularPlot( theta_resolution = 50, phi_resolution = 50, use_gm=True, direction_point=False, savefig=savefig)
        H_resonant = SP.resonant_Hamiltonian()
        D = dg.Diagonalizer(H_resonant)

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




class Gradients:
    def __init__(self, gradient_filename, merger_data_loc):
        

        #data
        self.merger_data_loc = merger_data_loc
        self.gradient_filename = gradient_filename

        #allgrad object (dims: spacetime up, spacetime (gradient) low, x,y,z, F, F)
        self.gradJ, self.gradJb, self.gradYe, self.x, self.y, self.z, self.it, self.limits = read_gradients(gradient_filename)

        
        # S_R/L_nu gradient (spacetime up, spacetime (gradient) low, x,y,z, F, F) [tranpose gradJ so new lower index is last and the sigma function works properly, then transpose back]
        self.grad_S_R_nu, self.grad_S_L_nu = sigma(np.transpose(self.gradJ, axes = (0,5,6,1,2,3,4)))
        self.grad_S_R_nu = np.transpose(self.grad_S_R_nu, axes = (0,3,4,5,6,1,2))
        self.grad_S_L_nu = np.transpose(self.grad_S_L_nu, axes = (0,3,4,5,6,1,2))        
    
        #THIS NEEDS GR TREATMENT
        ####################################
        #Electron fraction, baryon density
        self.merger_grid = h5py.File(merger_data_loc, 'r')
        self.rho = np.array(self.merger_grid['rho(g|ccm)']) #g/cm^3 (baryon mass density)
        self.Ye = np.array(self.merger_grid['Ye'])[self.limits[0,0]:self.limits[0,1], self.limits[1,0]:self.limits[1,1], self.limits[2,0]:self.limits[2,1]] #electron fraction 
        self.n_b = self.rho[self.limits[0,0]:self.limits[0,1], self.limits[1,0]:self.limits[1,1], self.limits[2,0]:self.limits[2,1] ]/M_p*(hbar**3 * c**3) #eV^3 (baryon number density)
        #differentials for matter gradients
        dx = self.x[1]-self.x[0]
        dy = self.y[1]-self.y[0]
        dz = self.z[1]-self.z[0]
        #electron fraction gradients #probably wont use these since they don't account for christoffel symbols
        self.grad_Ye = np.array(np.gradient(self.Ye, dx,dy,dz)) #(3, x,y,z)
        #append lower t axis (what to make time derivative?)
        self.grad_Ye = np.append(self.grad_Ye, np.zeros((1,self.grad_Ye.shape[1],self.grad_Ye.shape[2],self.grad_Ye.shape[3])), axis = 0) #(4, x,y,z)
        #baryon number density gradients
        self.grad_nb = np.array(np.gradient(self.n_b, dx,dy,dz)) #(3, x,y,z)
        #fix time derivative
        self.grad_nb = np.append(self.grad_nb, np.zeros((1,self.grad_nb.shape[1],self.grad_nb.shape[2],self.grad_nb.shape[3])), axis = 0) #(4, x,y,z)
        ####################################
        #need matter part gradients from changing Y_e, n_b
        self.grad_S_R_mat = np.zeros(np.shape(self.grad_S_R_nu))
        self.grad_S_L_mat = np.zeros(np.shape(self.grad_S_L_nu))
            
        #total Sigma Gradients (spacetime up, spacetime (gradient) low, x,y,z, F, F
        self.grad_S_R = self.grad_S_R_nu + self.grad_S_R_mat
        self.grad_S_L = self.grad_S_L_nu + self.grad_S_L_mat

    #total H_L and H_R gradients (ignoring extra terms in inverse p/abs)
    #(spacetime low, x,y,z, F, F)
    def grad_H_L(self, theta, phi):
        basis = Basis(theta,phi)
        return basis.kappa(self.grad_S_L)
    def grad_H_R(self, theta, phi):
        basis = Basis(theta,phi)
        return basis.kappa(self.grad_S_R)
   
        
    # calculates magnitude of gradient along minimizing resonant direction in each grid cell
    def minGradients(self, emu_data_loc, p_abs, z = None, phi_resolution = 5):
        min_gradients = np.zeros((self.gradJ.shape[2], self.gradJ.shape[3], self.gradJ.shape[4])) #x,y,z
        if z == None:
            z_range = range(self.gradJ.shape[4])
        else:
            z_range = [z]
        for x in range(self.gradJ.shape[2]):
            for y in range(self.gradJ.shape[3]):
                for z in z_range:

                    location = [x + self.limits[0,0],
                                y + self.limits[1,0],
                                z + self.limits[2,0]]
                    emu_filename = emu_data_loc + "i{:03d}".format(location[0])+"_j{:03d}".format(location[1])+"_k{:03d}".format(location[2])+"/allData.h5"
                    SP = SpinParams(t_sim = self.it, resonance_type = 'simplified', emu_file = emu_filename, merger_data_loc = self.merger_data_loc, gradient_filename = self.gradient_filename, location = location, p_abs=p_abs)
                    if SP.resonant_theta(phi=0) == None:
                        min_gradients[x,y,z] = None
                    else:
                        gradients = []
                        for phi in np.linspace(0, 2*np.pi, phi_resolution): 
                            theta = SP.resonant_theta(phi=phi)
                            grad_H_L = self.grad_H_L(theta, phi)[:,x,y,z] 
                            direction = Basis(theta,phi).n_vector
                            grad_along_direction = np.tensordot(grad_H_L, direction, axes = ([0],[0])) # (F,F)
                            gradients.append(gm.magnitude(grad_along_direction))
                        min_gradients[x,y,z] = np.min(gradients)
        
        return min_gradients
    
    #plots output of the above for a z slice
    def gradientsPlot(self, emu_data_loc, p_abs, z, vmin = -14, vmax = -9, phi_resolution = 5 , savefig=False, min_gradients = None):
        if type(min_gradients) == type(None):
            min_gradients = self.minGradients(emu_data_loc=emu_data_loc, p_abs=p_abs, z=z,
                                          phi_resolution = phi_resolution)
        else:
            min_gradients = min_gradients   
        plt.figure(figsize=(8,6))
        plt.pcolormesh(np.mgrid[self.limits[0,0]:self.limits[0,1]+1:1,self.limits[1,0]:self.limits[1,1]+1:1][0,:,:],
                       np.mgrid[self.limits[0,0]:self.limits[0,1]+1:1,self.limits[1,0]:self.limits[1,1]+1:1][1,:,:], 
                       np.log10(min_gradients[:,:,z]), cmap = 'jet',
                       vmin = vmin, vmax = vmax)
        plt.colorbar()
        plt.title('Minimum Gradient at Each Grid Cell')
        plt.xlabel('x index')
        plt.ylabel('y index')
        if savefig == True:
            plt.savefig('min_gradient.png')
        plt.show()
        np.where

        

        


#resonance type = 'full' or 'simplified'. 'full' uses the full resonance condition, 'simplified' uses the simplified resonance condition
#if 'full', density_matrix is the initial density matrix. 
class SpinParams:
    def __init__(self, t_sim, emu_file, merger_data_loc, location, p_abs, resonance_type = 'full', initial_ket = np.array([1,0,0,0,0,0]), gradient_filename = None):
        
        self.p_abs = p_abs
        self.t_sim = t_sim

        self.gradient_filename = gradient_filename
         
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
            self.Gradients_instance = Gradients(gradient_filename, merger_data_loc)
            #check location and timestep are match gradients file
            it, limits = self.Gradients_instance.it, self.Gradients_instance.limits
            assert(t_sim == it) 
            assert(location[0] >= limits[0,0] and location[0] <= limits[0,1]) 
            assert(location[1] >= limits[1,0] and location[1] <= limits[1,1])
            assert(location[2] >= limits[2,0] and location[2] <= limits[2,1])

            #get the gradient at the location #Index 0 means at the lower limit. (spacetime up, spacetime low, F ,F)
            self.gradJ = self.Gradients_instance.gradJ[:,:,location[0]-limits[0,0], location[1]-limits[1,0], location[2]-limits[2,0]]

            # S_R/L_nu gradient (spacetime up, spacetime (gradient) low, F, F) [tranpose gradJ so new lower index is last and the sigma function works properly, then transpose back]
            self.grad_S_R_nu = self.Gradients_instance.grad_S_R_nu[:,:,location[0]-limits[0,0], location[1]-limits[1,0], location[2]-limits[2,0]]
            self.grad_S_L_nu = self.Gradients_instance.grad_S_L_nu[:,:,location[0]-limits[0,0], location[1]-limits[1,0], location[2]-limits[2,0]]
            
            #need matter part gradients from changing Y_e, n_b
            self.grad_S_R_mat = self.Gradients_instance.grad_S_R_mat[:,:,location[0]-limits[0,0], location[1]-limits[1,0], location[2]-limits[2,0]]
            self.grad_S_L_mat = self.Gradients_instance.grad_S_L_mat[:,:,location[0]-limits[0,0], location[1]-limits[1,0], location[2]-limits[2,0]]
            
            #total Sigma Gradients
            self.grad_S_R = self.Gradients_instance.grad_S_R[:,:,location[0]-limits[0,0], location[1]-limits[1,0], location[2]-limits[2,0]]
            self.grad_S_L = self.Gradients_instance.grad_S_L[:,:,location[0]-limits[0,0], location[1]-limits[1,0], location[2]-limits[2,0]]

            
        #Resonance, initial density matrix
        self.initial_ket = initial_ket
        self.density_matrix = np.outer(self.initial_ket, np.conjugate(self.initial_ket))
        self.resonance_type = resonance_type

    #total H gradients (direction dependent) (space time low, F, F)
    def grad_H_L(self, theta, phi):
        basis = Basis(theta,phi)
        return basis.kappa(self.grad_S_L)
    def grad_H_R(self, theta, phi):
        basis = Basis(theta,phi)
        return basis.kappa(self.grad_S_R)

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
    def H_L_der(self, theta, phi):
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
        
    def H_R_der(self, theta, phi):
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
            return H_L_nogradient + self.H_L_der(theta, phi)
        else:
            return H_L_nogradient
        

    def H_R(self, theta, phi):
        H_R_nogradient = self.S_R_kappa(theta, phi) + self.H_vac + self.H_R_pm(theta, phi)
        if self.gradient_filename != None:
            return H_R_nogradient + self.H_R_der(theta, phi)
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
    
    def resonance(self, theta, phi):
        if self.resonance_type == 'full':
            return gm.dotprod(self.H(theta,phi),self.density_matrix)
        elif self.resonance_type == 'simplified':
            return np.real(2**(-1/2)*G*self.n_b*(3.*self.Ye-np.ones_like(self.Ye))+self.S_L_nu_kappa(theta,phi)[0,0])
        elif type(self.resonance_type) == type([]):
            diag1 = self.resonance_type[0]
            diag2 = self.resonance_type[1]
            return self.H(theta,phi)[diag1,diag1]-self.H(theta,phi)[diag2,diag2]


    #uses scipy rootfinder to locate polar angle of resonance contour. Assumes rotational symmetry (picks phi=0)
    def resonant_theta(self, phi=0):
        if self.resonance(0,phi)*self.resonance(np.pi,phi) > 0:
            return None
        else:
            theta = opt.bisect(self.resonance,0,np.pi,args = (phi))
        return theta
    
    #resonant Hamiltionian at azimuthal angle phi (should be independent of phi)
    def resonant_Hamiltonian(self, phi=0):
        theta = self.resonant_theta(phi)
        return self.H(theta,phi)
    

    def angularPlot(self, theta_resolution, phi_resolution, savefig=False, use_gm=True, direction_point=False):
        
        if use_gm==True:
            H_LR_array = np.array([[np.abs(np.trace(self.H_LR(theta, phi))) 
                                   for phi in np.linspace(0, 2*np.pi, phi_resolution)]
                                   for theta in np.linspace(0, np.pi, theta_resolution)])
        else: 
            H_LR_array = np.array([[gm.sum_magnitude(self.H_LR(theta, phi)) 
                                   for phi in np.linspace(0, 2*np.pi, phi_resolution)]
                                   for theta in np.linspace(0, np.pi, theta_resolution)])

        
        resonance_array = np.array([[self.resonance(theta,phi)
                                   for phi in np.linspace(0, 2*np.pi, phi_resolution)]
                                   for theta in np.linspace(0, np.pi, theta_resolution)]) 

        f = plt.figure()
        ax = f.add_subplot(projection = 'mollweide')
        ax.grid(False)
        #ax.set_title(r'Angular plot of $H_{LR}$')
        
        H_LR_im = ax.pcolormesh(np.linspace(-np.pi, np.pi, phi_resolution), 
                                np.linspace(0.5*np.pi, -0.5*np.pi, theta_resolution),
                                H_LR_array, 
                                cmap=plt.cm.hot, shading='auto')

        plt.colorbar(H_LR_im, label="eV")

        #resonance 
        res_im = ax.contour(np.linspace(-np.pi, np.pi, phi_resolution),
                            np.linspace(0.5*np.pi, -0.5*np.pi, theta_resolution),
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





    #########################
    ##Eigenvector resonance##
    #########################


    #resonance that decomposes initvector into energy basis and finds right hand magnitude of most important basis component.
    #function  used for scipy in maxRightHanded
    #negative bc looking for max and using minimize
    def rightHandedPart(self, theta, phi, initvector):
        if type(theta) == np.ndarray: #opt is calling in theta as a list ( [theta] ) instead of just theta. This is leading to a ragged nested sequence bug. This fixes it (sloppily)
            theta = theta[0]
        return -1*np.linalg.norm(dg.Diagonalizer(self.H(theta,phi)).largest_ket_component(initvector)[1][3:6])

    #finds maximum right handed part of largest component of initial eigenvector of resonant Hamiltonian
    #returns theta and max value
    #works as a resonance condition specifically for initvector
    def maxRightHanded(self, initvector, phi=0, method='Nelder-Mead', bounds = [(np.pi/4, 3*np.pi/4)]):
        optimal = opt.minimize(self.rightHandedPart, x0 = np.pi/2, args = (phi, initvector), bounds = bounds,  method = method)
        return optimal.x[0], -1*optimal.fun #theta_optimal, max_colorplot_vals
    
    #initial ket -independent resonance condition. assures existence of resonance for some initial ket
    #min_eigenvec = True returns the eigenvector that minimizes leftMinusRight

    def leftMinusRight(self, theta, phi, min_eigenvec = False):
        if type(theta) == np.ndarray: #opt is calling in theta as a list ( [theta] ) instead of just theta. This is leading to a ragged nested sequence bug. This fixes it (sloppily)
            theta = theta[0]
        H = self.H(theta, phi)
        eigenvectors = np.linalg.eig(H)[1]
        left_minus_right = [abs(np.linalg.norm(eigenvectors[0:3,n]) - np.linalg.norm(eigenvectors[3:6,n]))
                            for n in range(0,6)]
        if min_eigenvec == True:
            return min(left_minus_right), eigenvectors[:,np.argmin(left_minus_right)]
        return min(left_minus_right)
    
    #finds minimum of leftMinusRight over theta for some phi
    #returns theta and min value
    def minLeftMinusRight(self, phi=0, method='Nelder-Mead', bounds = [(np.pi/4, 3*np.pi/4)], min_eigenvec = False):
        x0 = (bounds[0][0]+bounds[0][1])/2
        optimal = opt.minimize(self.leftMinusRight, x0 = x0, args = (phi), bounds = bounds,  method = method)
        if min_eigenvec == True:
            return optimal.x[0], optimal.fun, self.leftMinusRight(optimal.x[0], phi, min_eigenvec = True)[1]
        return optimal.x[0], optimal.fun
    
    #plots magnitude of right handed part of largest energy eigenvector component of initial ket vector (should be large for resonance)
    #phi max is the phi along which a theta_optimal is found (maximizing right handed part of largest eigenvector component)
    #zoom is None (giving a full mollweide plot like angularPlot) or a number (giving a zoomed in plot around phi_optimal, theta_optimal)
    def angularEigenvectorPlot(self, theta_resolution, phi_resolution,
                                value = 'rmax', # = 'lminusr' or 'rmax'
                                phi_optimal = np.pi, zoom = None, 
                                zoom_resolution = 50, initvector = 'default', 
                                method = 'Nelder-Mead', bounds =[(np.pi/4, 3*np.pi/4)], 
                                savefig=False,  linearPlot = True):
        
        if initvector == 'default':
            initvector =  self.initial_ket
        
        if value == 'lminusr':
            theta_optimal, max_right = self.minLeftMinusRight(phi=phi_optimal, method = method, bounds = bounds)
            colorplot_vals = np.log10(1 - np.array([[self.leftMinusRight(theta,phi)
                                for phi in np.linspace(0, 2*np.pi, phi_resolution)]
                                for theta in np.linspace(0, np.pi, theta_resolution)]))
            vmin = None
        elif value == 'rmax':
            theta_optimal, max_right = self.maxRightHanded(initvector, phi=phi_optimal, method = method, bounds = bounds)
            colorplot_vals = np.log10(np.array([[-1*self.rightHandedPart(theta,phi, initvector)
                                for phi in np.linspace(0, 2*np.pi, phi_resolution)]
                                for theta in np.linspace(0, np.pi, theta_resolution)]))
            vmin = -8
        
        print('theta_optimal = ', str(theta_optimal),
             ' along phi = ', str(phi_optimal),
             ' with max_right = ', str(max_right))

        f = plt.figure(figsize=(8,6))
        ax = f.add_subplot(projection = 'mollweide') 
        ax.grid(False)

         #colorplot
        colorplot_im = ax.pcolormesh(np.linspace(-np.pi, np.pi, phi_resolution), 
                                np.linspace(0.5*np.pi, -0.5*np.pi, theta_resolution),
                                colorplot_vals, 
                                cmap=plt.cm.hot, shading='auto', vmin = vmin)
        plt.colorbar(colorplot_im)

        #resonance 
        resonance_array = np.array([[self.resonance(theta,phi)
                                for phi in np.linspace(0, 2*np.pi, phi_resolution)]
                                for theta in np.linspace(0, np.pi, theta_resolution)]) 
        res_im = ax.contour(np.linspace(-np.pi, np.pi, phi_resolution),
                            np.linspace(0.5*np.pi, -0.5*np.pi, theta_resolution),
                           resonance_array, levels=[0.], colors='cyan')
        h1,l1 = res_im.legend_elements()
        
        #add net flux point 
        J_avg = np.array([gm.magnitude(np.average(self.J[n], axis = 2)) for n in range(0,4)])
        flux_point = ax.scatter([np.arctan2(J_avg[2],J_avg[1])],[np.arctan2(J_avg[3],
                                                (J_avg[1]**2+J_avg[2]**2)**(1/2))],  label = 'ELN Flux Direction', color='lime')
        
        
        #add maximum along phi_optimal
        max_point_plot = ax.scatter([phi_optimal - np.pi],
                                    [-1*theta_optimal + np.pi/2],
                                    label = 'maximum Right-Handed Component', 
                                    color='magenta')
        
        #legend
        plt.legend([h1[0], flux_point, max_point_plot], ["Resonant Directions", "Number Flux Direction", 'Maximum Right-Handed Component'], loc=(0, 1.13))
        

        
        #add zoomed in graphic
        if type(zoom) == float:
            f_z = plt.figure(figsize = (8,6))
            ax_z = f_z.add_subplot()
            
            if value == 'lminusr':
                 colorplot_vals_zoom = np.log10(1 - np.array([[self.leftMinusRight(theta,phi)
                                    for phi in np.linspace(phi_optimal - zoom, phi_optimal + zoom, zoom_resolution)]
                                    for theta in np.linspace(theta_optimal - zoom, theta_optimal + zoom, zoom_resolution)]))
            elif value == 'rmax':
                 colorplot_vals_zoom = np.log10(np.array([[-1*self.rightHandedPart(theta,phi, initvector)
                                    for phi in np.linspace(phi_optimal - zoom, phi_optimal + zoom, zoom_resolution)]
                                    for theta in np.linspace(theta_optimal - zoom, theta_optimal + zoom, zoom_resolution)]))
            
            #colorplot
            colorplot_im_z = ax_z.pcolormesh(
                                np.linspace(phi_optimal - zoom, phi_optimal + zoom, zoom_resolution), 
                                np.linspace(theta_optimal + zoom, theta_optimal - zoom, zoom_resolution),
                                colorplot_vals_zoom, 
                                cmap=plt.cm.hot, shading='auto', vmin = -8)
            plt.colorbar(colorplot_im_z)

            #resonance
            resonance_array_zoom = np.array([[self.resonance(theta,phi)
                                    for phi in np.linspace(phi_optimal - zoom, phi_optimal + zoom, zoom_resolution)]
                                    for theta in np.linspace(theta_optimal - zoom, theta_optimal + zoom, zoom_resolution)])
            res_im_z = ax_z.contour(
                                np.linspace(phi_optimal - zoom, phi_optimal + zoom, zoom_resolution), 
                                np.linspace(theta_optimal + zoom, theta_optimal - zoom, zoom_resolution),
                                resonance_array_zoom, levels=[0.], colors='cyan')
            
            #max point
            max_point_plot = ax_z.scatter([phi_optimal],
                                    [theta_optimal],
                                    label = 'maximum Right-Handed Component', 
                                    color='magenta')
            
            # Create a Rectangle patch
            rect = mpl.patches.Rectangle((phi_optimal-zoom - np.pi, np.pi/2-(theta_optimal+zoom)), 2*zoom, 2*zoom, linewidth=1, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)
                        
            plt.show()

    
        #add linearPlot graphic
        if linearPlot == True:
            self.linearEigenvectorPlot(zoom_resolution, initvector = initvector, value = value, phi_optimal= phi_optimal, zoom = zoom,  method = method, bounds =[(np.pi/4, 3*np.pi/4)])
    

        if savefig == True: 
            ax.tight_layout()
            plt.savefig('angularPlot.png', dpi=300)


        

        
    def linearEigenvectorPlot(self, theta_resolution,  
                              initvector ='default', value = 'rmax',
                              zoom = None, shift = 0, phi_optimal= np.pi,
                              method = 'Nelder-Mead', bounds =[(np.pi/4, 3*np.pi/4)],
                              extra_lines = None, extra_init_vectors = None):
    
        plt.figure(figsize = (8,6))
        plt.xlabel(r'$\theta$')
        plt.grid(True)

        if initvector == 'default':
            initvector =  self.initial_ket
        
        if value == 'lminusr':
            theta_optimal, max_right = self.minLeftMinusRight(phi=phi_optimal, method = method, bounds = bounds)
            plt.title(f'Linear Plot of L-minus-R vs theta (phi = {phi_optimal:.3})')
            plt.ylabel(r'$1-|L-R|$')
            if zoom == None:
                thetas = np.linspace(0, np.pi, theta_resolution)
                plt.xlim(0,np.pi)
            else:
                thetas = np.linspace(theta_optimal - zoom + shift, theta_optimal + zoom + shift, theta_resolution)
                plt.xlim( theta_optimal - zoom + shift, theta_optimal + zoom + shift)
            plot_vals = 1 - np.array([self.leftMinusRight(theta, phi_optimal)
                                   for theta in thetas]) 
        elif value == 'rmax':
            theta_optimal, max_right = self.maxRightHanded(initvector, phi=phi_optimal, method = method, bounds = bounds)
            plt.title(f'Linear Plot of Right-Handed Component of Ket vs theta (phi = {phi_optimal:.3})')
            if zoom == None:
                thetas = np.linspace(0, np.pi/2, theta_resolution)
                plt.xlim(0,np.pi)
            else:
                thetas = np.linspace(theta_optimal - zoom + shift, theta_optimal + zoom + shift, theta_resolution)
                plt.xlim(theta_optimal - zoom + shift, theta_optimal + zoom + shift)
            plot_vals = np.array([-1*self.rightHandedPart(theta, phi_optimal, initvector)
                                   for theta in thetas])
            
        #plot full resonance value
        plt.plot(thetas, plot_vals, color = 'r')

        #extra_init_vector to see specific resonance condition solutions over plot of the general resonance condition
        if extra_init_vectors != None:
            extra_thetas = np.array([self.maxRightHanded(extra_init_vector, phi=phi_optimal, method = method, bounds = bounds)[0]
                                    for extra_init_vector in extra_init_vectors])
            print('Extra Thetas = ', extra_thetas)
            extra_thetas_vlines = plt.vlines(extra_thetas, [0], [max(plot_vals)], linestyles = '--', label = 'Specified Initial Vectors', color='lime')

        #plot vlines
        theta_resonant = self.resonant_theta(phi=phi_optimal)  
        standard_resonance_vline = plt.vlines([theta_resonant],[0],[max(plot_vals)], linestyles = '-', label = 'Simplified Resonance', color='cyan')
        max_point_vline =          plt.vlines([theta_optimal],[0],[max(plot_vals)], linestyles = ':', label = 'Max value', color='magenta')
        bounds_vlines =            plt.vlines([bounds[0][0], bounds[0][1]],[0],[1/4*max(plot_vals)], linestyles = '-.', label = 'Bounds', color='orange')
        if extra_lines != None:
          extra_vlines = plt.vlines(extra_lines, [0], [max(plot_vals)], linestyles = '--', label = 'Extra Lines', color='lime')

        print("Optimal theta in Range = ", str(theta_optimal))
        plt.legend()

    ###################



    def azimuthalGradientsPlot(self, phi_resolution = 100, savefig=False, adiabaticity_threshold = 1):
        phis = np.linspace(0, 2*np.pi, phi_resolution)
        gradients = []
        H_LR_ee = []
        for phi in phis: 
            theta = self.resonant_theta(phi=phi)
            grad_H_L = self.grad_H_L(theta, phi) 
            direction = Basis(theta,phi).n_vector
            grad_along_direction = np.tensordot(grad_H_L, direction, axes = ([0],[0])) # (F,F)
            H_LR_ee_along_direction = self.H_LR(theta, phi)[0,0]
            gradients.append(grad_along_direction[0,0])
            H_LR_ee.append(H_LR_ee_along_direction)
        
        gradients = np.array(np.abs(gradients))
        H_LR_ee = np.array(np.abs(H_LR_ee))

        adiab = 2*H_LR_ee**2/gradients
        min_phi = np.linspace(0,2*np.pi, phi_resolution)[np.argmin(gradients)]
        print('minimum resonance magnitude = ', str(np.min(gradients)))
        print('minimizing resonant phi = ', str(min_phi))
        print('minimizing theta = ', str(self.resonant_theta(phi=min_phi)))
        f, ax = plt.subplots(1,2, figsize=(8,6))
        ax[0].plot(phis, gradients)
        #shade region of plot where adiab>1
        ax[0].fill_between(phis, 0, np.max(gradients), where=adiab > adiabaticity_threshold,
                color='red', alpha=0.5)
        ax[0].set_xlabel('Azimuthal Angle')
        ax[0].set_ylabel('Gradient Magnitude')
        ax[0].set_title('Gradient Along Resonant Directions at a Point')

        ax[1].plot(phis,adiab)
        ax[1].set_xlabel('Azimuthal Angle')
        ax[1].set_ylabel('Adiabaticity Parameter')
        ax[1].set_title('Adiabaticity Parameter Along Resonant Directions at a Point')
        #ax[1].set_ylim(0,1)
        plt.tight_layout()

        if savefig == True:
            plt.savefig('azimuthal_gradient.png')
        
        
    
