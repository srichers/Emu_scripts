import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../data_reduction")
import numpy as np
import h5py
from tqdm.notebook import tqdm
from constants import hbar, c, M_p, M_3flavor, G
import diagonalizer as dg
from matrix import visualizer
from merger_grid import Merger_Grid
from matrix import trace_matrix, dagger
from basis import Basis
import matplotlib as mpl
import matplotlib.pyplot as plt
import gellmann as gm
from scipy import signal
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
        self.merger_grid = h5py.File(merger_data_loc, 'r')
        self.gradient_filename = gradient_filename

        #gradJ(b) = allgrad object (dims: spacetime up, spacetime (gradient) low, x,y,z, F, F)
        self.gradJ, self.gradJb, self.gradYe, self.x, self.y, self.z, self.it, self.limits = read_gradients(gradient_filename)
        
        #matter parts and their gradients
        self.Ye = np.array(self.merger_grid['Ye'])[self.limits[0,0]:self.limits[0,1]+1,
                                                   self.limits[1,0]:self.limits[1,1]+1,
                                                   self.limits[2,0]:self.limits[2,1]+1] # electron fraction
        self.nb = np.array(self.merger_grid['rho(g|ccm)'])[self.limits[0,0]:self.limits[0,1]+1,
                                                           self.limits[1,0]:self.limits[1,1]+1,
                                                           self.limits[2,0]:self.limits[2,1]+1]/M_p*(hbar**3 * c**3)#eV^3 (baryon number density)
        self.gradYe = self.gradYe[:,:,:,:,0,0] #(spacetime down, x,y,z)
        self.gradnb = self.gradJb[0,:,:,:,:,0,0] #(spacetime down, x,y,z). Time part of Jb. last two indices are redundant

        # S_R/L_nu gradient (spacetime up, spacetime (gradient) low, x,y,z, F, F) [tranpose gradJ so new lower index is last and the sigma function works properly, then transpose back]
        self.grad_S_R_nu, self.grad_S_L_nu = sigma(np.transpose(self.gradJ, axes = (0,5,6,1,2,3,4)))
        self.grad_S_R_nu = np.transpose(self.grad_S_R_nu, axes = (0,3,4,5,6,1,2))
        self.grad_S_L_nu = np.transpose(self.grad_S_L_nu, axes = (0,3,4,5,6,1,2))        
    
        #need matter part gradients from changing Y_e, n_b
        self.grad_S_R_mat    = np.zeros_like(self.grad_S_R_nu)
        self.grad_S_R_mat[0] = -2**(-1/2)*G*(
                                            np.transpose(
                                                self.gradnb[:,np.newaxis,np.newaxis,:,:,:]
                                                * np.array([[3*self.Ye-1, np.zeros_like(self.Ye),  np.zeros_like(self.Ye)],
                                                            [np.zeros_like(self.Ye), self.Ye-1,    np.zeros_like(self.Ye)],
                                                            [np.zeros_like(self.Ye),  np.zeros_like(self.Ye),   self.Ye-1]]),
                                                axes = (0,3,4,5,1,2)) #multiplying (spacetime, 1, 1, x,y,z) by (1, F, F, x,y,z) gives (spacetime, F, F, x,y,z) and reshape to make (spacetime, x,y,z, F,F)
                                        +   np.transpose(
                                                self.nb
                                                * np.array([[3*self.gradYe,  np.zeros_like(self.gradYe),   np.zeros_like(self.gradYe)],
                                                            [ np.zeros_like(self.gradYe), self.gradYe,     np.zeros_like(self.gradYe)],
                                                            [ np.zeros_like(self.gradYe), np.zeros_like(self.gradYe),   self.gradYe]]),
                                                axes = (2,3,4,5,0,1)) #multiplying (1, 1, 1, x,y,z) by (F, F, spacetime, x,y,z) gives ( F, F, spacetime, x,y,z) and reshape to make (spacetime, x,y,z, F,F)
                                            )
                
        self.grad_S_L_mat = (-1)*np.transpose(self.grad_S_R_mat, axes=(0,1,2,3,4,6,5))   
            
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
    def minGradients(self, emu_data_loc, p_abs, z = None, phi_resolution = 5, method = 'simplified', resonance_type = None):
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
                    if SP.resonant_theta(phi=0, resonance_type=resonance_type) == None:
                        min_gradients[x,y,z] = None
                    else:
                        gradients = []
                        for phi in np.linspace(0, 2*np.pi, phi_resolution): 
                            theta = SP.resonant_theta(phi=phi, resonance_type=resonance_type)
                            grad_H_L = self.grad_H_L(theta, phi)[:,x,y,z] 
                            direction = Basis(theta,phi).n_vector
                            grad_along_direction = np.tensordot(grad_H_L, direction, axes = ([0],[0])) # (F,F)
                            if method == 'simplified':
                                gradients.append(grad_along_direction[0,0])
                            elif method == 'GM':
                                gradients.append(gm.magnitude(grad_along_direction))
                            else:
                                raise(ValueError('method must be "simplified" or "GM"'))
                        min_gradients[x,y,z] = np.min(gradients) 
        return min_gradients
    
    
    #plots output of the above for a z slice
    #set min_gradients to the output of the above if precomputed
    def gradientsPlot(self, emu_data_loc, p_abs, z, vmin = -14, vmax = -9, phi_resolution = 5 , savefig=False, min_gradients = None, resonance_type = None):
        if type(min_gradients) == type(None):
            min_gradients = self.minGradients(emu_data_loc=emu_data_loc, p_abs=p_abs, z=z,
                                          phi_resolution = phi_resolution, resonance_type=resonance_type)
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
            plt.savefig('min_gradient.pdf')
        else:
            plt.show()
       
    
    #finds average adiabaticity on resonant band at each grid cell over a z slice. Uses simplified resonance
    def averageAdiabaticities(self, zs, emu_data_loc, p_abs, phi_resolution = 20, resonance_type = None):
        phis = np.linspace(0, 2*np.pi, phi_resolution)
        avg_adiabaticity = np.zeros((self.gradJ.shape[2], self.gradJ.shape[3], len(zs))) #x,y
        for zn in enumerate(zs):
            z = zn[1] - self.limits[2,0] # z index in the gradient file
            n = zn[0]
            for x in range(self.gradJ.shape[2]):
                for y in range(self.gradJ.shape[3]):
                    location = [x + self.limits[0,0],
                                y + self.limits[1,0],
                                z + self.limits[2,0]]
                    emu_filename = emu_data_loc + "i{:03d}".format(location[0])+"_j{:03d}".format(location[1])+"_k{:03d}".format(location[2])+"/allData.h5"
                    SP = SpinParams(t_sim = self.it, resonance_type = 'simplified', emu_file = emu_filename, merger_data_loc = self.merger_data_loc, gradient_filename = self.gradient_filename, location = location, p_abs=p_abs)
                    if SP.resonant_theta(phi=0, resonance_type = resonance_type) == None:
                        avg_adiabaticity[x,y,n] = None
                    else:
                        adiabs = []
                        for phi in phis: 
                            theta = SP.resonant_theta(phi=phi, resonance_type = resonance_type)
                            grad_H_L_ee = self.grad_H_L(theta, phi)[:,x,y,z,0,0] 
                            direction = Basis(theta,phi).n_vector
                            grad_along_direction = np.abs(np.tensordot(grad_H_L_ee, direction, axes = ([0],[0])) )
                            H_LR_ee_along_direction = np.abs(SP.H_LR(theta, phi)[0,0])
                            adiabs.append(2*H_LR_ee_along_direction**2/grad_along_direction)
                        avg_adiabaticity[x,y,n] = np.average(adiabs)
        return avg_adiabaticity
    
    def plotAdiabaticities(self, zs, 
                           emu_data_loc, p_abs, vmin, vmax, 
                           phi_resolution = 2,
                           savefig = False, adiabaticities = None):
  
        if type(adiabaticities) == type(None):
            adiabaticities = self.averageAdiabaticities(zs, emu_data_loc, p_abs, phi_resolution = phi_resolution)
      
        xdim = 1E-5*self.merger_grid['x(cm)'][self.limits[0,0]:self.limits[0,1]+1,
                                                   self.limits[1,0]:self.limits[1,1]+1,
                                                   zs[0]]
        ydim = 1E-5*self.merger_grid['y(cm)'][self.limits[0,0]:self.limits[0,1]+1,
                                                    self.limits[1,0]:self.limits[1,1]+1,
                                                    zs[0]]
        zs_km = 1E-5*self.merger_grid['z(cm)'][0,0,zs]
        
        n = len(zs)
      
        f,ax = plt.subplots(1,n,figsize=(n*6,8), sharex = True, sharey = True, squeeze = False,)
        for k in range(n):
            #colorplot
            im = ax[0,k].pcolormesh(xdim, ydim, np.log10(adiabaticities[:,:,k]),
                                     vmin = np.log10(vmin), vmax = np.log10(vmax), 
                                     cmap = 'YlGnBu_r')
            
            #zval text
            ax[0,k].text((xdim[0,0] - xdim[-1,0])*0.99 + xdim[-1,0],
                         (ydim[0,-1] - ydim[0,0])*0.95 + ydim[0,0],rf'$z$ = {zs_km[k]:.1f} km', backgroundcolor = 'white')
        plt.tight_layout()

        f.colorbar(im, label=r'log$(\gamma)$', location = 'bottom',ax=ax.ravel().tolist(), pad = 0.1,aspect=30)
        middle_n = n//2
        ax[0,middle_n].set_xlabel(r'$x$-coordinate (km)', fontsize = 14)
        ax[0,0].set_ylabel(r'$y$-coordinate (km)', fontsize = 14)
        #ax[0,middle_n].set_title('Average Adiabaticity in Resonant Directions at Each Cell', fontsize = 16, pad = 20,)

        if type(savefig) == str: 
            f.savefig(savefig + '.pdf', dpi=300, bbox_inches = 'tight')

        
        

        


#resonance type = 'full' or 'simplified'. 'full' uses the full resonance condition, 'simplified' uses the simplified resonance condition
#if 'full', density_matrix is the initial density matrix. 
class SpinParams:
    def __init__(self, 
                 t_sim,
                 emu_file,
                 merger_data_loc,
                 location,
                 p_abs,
                 resonance_type = 'full',
                 initial_ket = np.array([1,0,0,0,0,0]),
                 gradient_filename = None):
        
        self.p_abs = p_abs
        self.t_sim = t_sim
        self.t_seconds = h5py.File(emu_file, 'r')['t(s)'][t_sim]

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
        #we want the derivative along x_i of component x_j of the gradient
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
    
    def resonance(self, theta, phi, resonance_type = None):
        if resonance_type == None:
            resonance_type = self.resonance_type

        if resonance_type == 'full':
            return gm.dotprod(self.H(theta,phi),self.density_matrix)
        elif resonance_type == 'simplified':
            return np.real(2**(-1/2)*G*self.n_b*(3.*self.Ye-np.ones_like(self.Ye))+self.S_L_nu_kappa(theta,phi)[0,0])            
        elif type(resonance_type) == type([]):
            diag1 = resonance_type[0]
            diag2 = resonance_type[1]
            return np.real(self.H_L(theta,phi)[diag1,diag1])-np.real(self.H_R(theta,phi)[diag2-3,diag2-3])



    #uses scipy rootfinder to locate polar angle of resonance contour. 
    def resonant_theta(self, phi=0, resonance_type = None):
        if self.resonance(0,phi, resonance_type)*self.resonance(np.pi,phi, resonance_type) > 0:
            return None
        else:
            theta = opt.bisect(self.resonance,0,np.pi,args = (phi,resonance_type),  xtol=9.88178e-10)
        return np.float64(theta)
    
    #resonant Hamiltionian at azimuthal angle phi (should be independent of phi)
    def resonant_Hamiltonian(self, phi=0, resonance_type = None):
        theta = self.resonant_theta(phi, resonance_type = resonance_type)
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
            
        if type(savefig) == str: 
            plt.tight_layout()
            plt.savefig(savefig + '.pdf', dpi=300)


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
    
    #Omega parameter at resonant theta as described in paper. Mostly defined for use in scipy optimization functions.
    #negate = True returns -1*omega for minimize
    def Omega(self, theta, phi, negate = False): 
        if type(theta)  == np.ndarray: #opt is calling in theta as a list ( [theta] ) instead of just theta. This is leading to a ragged nested sequence bug. This fixes it (sloppily)
                theta = theta[0]
        if type(negate) == np.ndarray:
            negate = negate[0]
        H = self.H(theta, phi)
        eigenvectors = np.linalg.eig(H)[1]
        left_minus_right = [abs(np.linalg.norm(eigenvectors[0:3,n])**2 - np.linalg.norm(eigenvectors[3:6,n])**2)
                            for n in range(0,6)]
        if negate:
            return -1*(np.sqrt(2)*3)*np.sqrt(1 - min(left_minus_right))
        else:
            return    (np.sqrt(2)*3)*np.sqrt(1 - min(left_minus_right))
    
    def maxOmega(self, phi=0, method='Nelder-Mead', bounds = [(np.pi/4, 3*np.pi/4)], min_eigenvec = False):
        x0 = (bounds[0][0]+bounds[0][1])/2
        optimal = opt.minimize(self.Omega, x0 = x0, args = (phi, True), bounds = bounds,  method = method)
        return optimal.x[0], optimal.fun
    
    #returns the resonant ket at each resonant theta in thetas
    def resonant_states(self, thetas, P = True):
        init_state_array = []
        for theta in thetas:
            ket = self.leftMinusRight(theta = theta, phi= np.pi, min_eigenvec = True)[1]
            ket[3:6] = 0
            ket = ket/np.linalg.norm(ket)
            if P == True:
                density_matrix = np.outer(ket, ket.conj())
                init_state_array.append(density_matrix)
            else: 
                init_state_array.append(ket)
        return np.array(init_state_array)
    
    #plots magnitude of right handed part of largest energy eigenvector component of initial ket vector (should be large for resonance)
    #phi max is the phi along which a theta_optimal is found (maximizing right handed part of largest eigenvector component)
    #zoom is None (giving a full mollweide plot like angularPlot) or a number (giving a zoomed in plot around phi_optimal, theta_optimal)
    def angularEigenvectorPlot(self, theta_resolution, phi_resolution,
                                value = 'Omega', # = 'lminusr' or 'rmax' or 'Omega'
                                phi_optimal = np.pi, zoom = None, shift = [0,0],
                                vmin = -8, vmax = -5,
                                zoom_resolution = 50, initvector = None, 
                                flavor_resonances = [(0,0,'cyan'), (1,1,'lime'), (0,1,'magenta')],
                                method = 'Nelder-Mead', bounds =[(np.pi/4, 3*np.pi/4)], 
                                savefig=False,  linearPlot = True):
        
        if type(initvector) == type(None):
            initvector =  self.initial_ket
        
        if value == 'lminusr':
            theta_optimal, max_right = self.minLeftMinusRight(phi=phi_optimal, method = method, bounds = bounds)
            colorplot_vals = np.log10(1 - np.array([[self.leftMinusRight(theta,phi)
                                for phi   in np.linspace(0, 2*np.pi, phi_resolution)]
                                for theta in np.linspace(0, np.pi,   theta_resolution)]))

        elif value == 'rmax':
            theta_optimal, max_right = self.maxRightHanded(initvector, phi=phi_optimal, method = method, bounds = bounds)
            colorplot_vals = np.log10(np.array([[-1*self.rightHandedPart(theta,phi, initvector)
                                for phi   in np.linspace(0, 2*np.pi, phi_resolution)]
                                for theta in np.linspace(0, np.pi,   theta_resolution)]))
        
        elif value == 'Omega':
            theta_optimal, max_right = self.maxOmega(phi=phi_optimal, method = method, bounds = bounds)
            colorplot_vals = np.log10(np.array([[self.Omega(theta,phi)
                                for phi   in np.linspace(0, 2*np.pi, phi_resolution)]
                                for theta in np.linspace(0, np.pi,   theta_resolution)]))
           
        
        print('theta_optimal = ', str(theta_optimal),
             ' along phi = ', str(phi_optimal),
             ' with max_right = ', str(max_right))
        
        f = plt.figure(figsize=(8,6))
        if zoom == None:
            ax = f.add_subplot(projection = 'mollweide') 
        else:
            ax = f.add_subplot(121,projection = 'mollweide') 
        ax.grid(False)

        #colorplot
        colorplot_im = ax.pcolormesh(np.linspace(-np.pi, np.pi, phi_resolution), 
                                np.linspace(0.5*np.pi, -0.5*np.pi, theta_resolution),
                                colorplot_vals, 
                                cmap=plt.cm.hot, shading='auto', vmin = vmin, vmax = vmax)
        if zoom == None:
            plt.colorbar(colorplot_im)

        #add net flux point 
        #J_avg = np.array([gm.magnitude(np.average(self.J[n], axis = 2)) for n in range(0,4)])
        #flux_point = ax.scatter([np.arctan2(J_avg[2],J_avg[1])],[np.arctan2(J_avg[3],
        #                                        (J_avg[1]**2+J_avg[2]**2)**(1/2))],  label = 'ELN Flux Direction', color='lime')
        
        
        #add maximum along phi_optimal
        #max_point_plot = ax.scatter([phi_optimal - np.pi],
        #                            [-1*theta_optimal + np.pi/2],
        #                            label = 'maximum Right-Handed Component', 
        #                            color='magenta')
        
        #axes
        yT=[np.pi/2, np.pi/3, np.pi/6, 0, -np.pi/6, -np.pi/3, -np.pi/2]
        yL=[0, r'$\frac{\pi}{6}$', r'$\frac{\pi}{3}$',r'$\frac{\pi}{2}$',
            r'$\frac{2\pi}{3}$',r'$\frac{5\pi}{6}$',
            r'$\pi$']
        plt.yticks(yT, yL)
        ax.set_ylabel(r'$\theta$', rotation=0, labelpad = 8, fontsize = 14)
        
        #xT=[ -2*np.pi/3, -np.pi/3, 0, np.pi/3, 2*np.pi/3]
        #xL=[  r'$\frac{\pi}{3}$', r'$\frac{2\pi}{3}$',
        #    r'$\pi$',r'$\frac{4\pi}{3}$',
        #    r'$\frac{5\pi}{3}$']
        plt.xticks([],[], zorder = 10, fontsize = 12)
        #ax.tick_params(axis='x', pad=100)        
        #ax.set_zorder(10)
        
        #add zoomed in graphic
        if type(zoom) == float:
            f.set_size_inches(12.3,4.5)
            ax_z = f.add_subplot(122)
            
            if value == 'lminusr':
                 colorplot_vals_zoom = np.log10(1 - np.array([[self.leftMinusRight(theta,phi)
                                    for phi in np.linspace(phi_optimal - zoom + shift[1], phi_optimal + zoom + shift[1], zoom_resolution)]
                                    for theta in np.linspace(theta_optimal - zoom + shift[0], theta_optimal + zoom + shift[0] , zoom_resolution)]))
            elif value == 'rmax':
                 colorplot_vals_zoom = np.log10(np.array([[-1*self.rightHandedPart(theta,phi, initvector)
                                    for phi in np.linspace(phi_optimal - zoom + shift[1], phi_optimal + zoom + shift[1], zoom_resolution)]
                                    for theta in np.linspace(theta_optimal - zoom + shift[0], theta_optimal + zoom + shift[0], zoom_resolution)]))
            elif value == 'Omega':
                 colorplot_vals_zoom = np.log10(np.array([[self.Omega(theta, phi)
                                    for phi in np.linspace(phi_optimal - zoom + shift[1], phi_optimal + zoom + shift[1], zoom_resolution)]
                                    for theta in np.linspace(theta_optimal - zoom + shift[0], theta_optimal + zoom + shift[0], zoom_resolution)]))
            
            #colorplot
            colorplot_im_z = ax_z.pcolormesh(
                                np.linspace(phi_optimal - zoom + shift[1], phi_optimal + zoom + shift[1], zoom_resolution), 
                                np.linspace(theta_optimal - zoom + shift[0], theta_optimal + zoom + shift[0], zoom_resolution),
                                colorplot_vals_zoom, 
                                cmap=plt.cm.hot, shading='auto',  vmin = vmin, vmax = vmax)
            cbar = f.colorbar(colorplot_im_z)
            cbar.ax.set_ylabel(r'log$(\Omega)$', 
                         fontsize = 14, labelpad=10)

            #resonance
            resonance_array_zoom = np.array([[self.resonance(theta,phi)
                                    for phi in np.linspace(phi_optimal - zoom + shift[1], phi_optimal + zoom + shift[1], zoom_resolution)]
                                    for theta in np.linspace(theta_optimal - zoom + shift[0], theta_optimal + zoom + shift[0], zoom_resolution)])
            res_im_z = ax_z.contour(
                                np.linspace(phi_optimal - zoom + shift[1], phi_optimal + zoom + shift[1], zoom_resolution), 
                                np.linspace(theta_optimal + zoom + shift[0], theta_optimal - zoom + shift[0], zoom_resolution),
                                resonance_array_zoom, levels=[0.], colors='cyan')

            #loop through chosen simplified resonances, find them and plot them
            neutrino_flavors = {0:'e', 1:'x', 2:r'\tau'} #label names (x because its for a general heavy lepton flavor.)
            legend_arts   = [] #for legend
            legend_labels = []
            for n,k,color in flavor_resonances:
                resonance_array = np.array([[self.resonance(theta, phi, resonance_type = [n,k+3]) 
                                    for phi in np.linspace(phi_optimal - zoom + shift[1], phi_optimal + zoom + shift[1], zoom_resolution)]
                                    for theta in np.linspace(theta_optimal - zoom + shift[0], theta_optimal + zoom + shift[0], zoom_resolution)])
                contour = ax_z.contour(
                                np.linspace(phi_optimal - zoom + shift[1], phi_optimal + zoom + shift[1], zoom_resolution), 
                                np.linspace(theta_optimal + zoom + shift[0], theta_optimal - zoom + shift[0], zoom_resolution),
                                resonance_array, levels=[0.], colors=color,)       
                art, label = contour.legend_elements()
                legend_arts.append(art[0])
                legend_labels.append(rf'${neutrino_flavors[n]}_L \rightleftharpoons {neutrino_flavors[k]}_R$')  
                                          
            # Create a Rectangle patch
            rect = mpl.patches.Rectangle((phi_optimal-zoom - np.pi, np.pi/2-(theta_optimal+zoom)),
                                          2*zoom, 2*zoom, linewidth=1, 
                                          edgecolor='white', facecolor='none', linestyle = 'dashed')
            ax.add_patch(rect)

            #legend
            f.legend(legend_arts,
                       legend_labels,
                       fontsize = 14,
                       bbox_to_anchor = (0.51, 0.35),
                       frameon = False
                       )
        
            #x,y labels
            ax_z.set_xlabel(r'$\phi$', fontsize = 14)
            ax_z.set_ylabel(r'$\theta$', rotation=0, labelpad = 8, fontsize = 14)
            plt.xticks(fontsize = 10)
            plt.yticks(fontsize = 10)

            #x,y limits
            ax_z.set_xlim(phi_optimal - zoom + shift[1], phi_optimal + zoom + shift[1])
            ax_z.set_ylim(theta_optimal - zoom + shift[0], theta_optimal + zoom + shift[0])
            
        #f.suptitle(r'Angular Plot of Resonance Parameter and Simplified')
        plt.tight_layout(pad = 2)           

        
        #savefig
        if type(savefig) == str: 
            f.savefig(savefig + '.pdf', dpi=300)
        else:
            plt.show()
            
            
        #add linearPlot graphic
        if linearPlot == True:
            self.linearEigenvectorPlot(zoom_resolution, initvector = initvector, value = value, phi_optimal= phi_optimal, zoom = zoom,  method = method, bounds =[(np.pi/4, 3*np.pi/4)])
    


    #resonant thetas is a list of tuples (n,k,color) corresponding to classical resonances in the nth to kth diagonal, 
    # whose location will be plotted in color 'color'   
    #function also returns the resonant thetas.
    def linearEigenvectorPlot(self, theta_resolution,  
                              initvector = None, zoom_on_vector = None, value = 'Omega',
                              zoom = None, shift = 0, phi_optimal= np.pi,
                              method = 'Nelder-Mead', vmax = None,
                              bounds =[(np.pi/4, 3*np.pi/4)], max_point = False,
                              extra_lines = None, extra_init_vectors = None, flavor_resonances = [(0,0,'cyan'), (1,1,'lime'), (0,1,'magenta')],
                              savefig = False):
    
        #factor to multiply the y axis by. Have to manually change the label 
        factor = 10000

        plt.figure(figsize = (8,6))
        plt.xlabel(r'$\theta$', fontsize = 14)

        

        #find theta_optimal and set y-axis label
        if value == 'lminusr':
            theta_optimal, max_right = self.minLeftMinusRight(phi=phi_optimal, method = method, bounds = bounds)
            plt.ylabel(r'$1- |L - R|$ $(\times 10^{-4})$')
        elif value == 'rmax':
            if type(initvector) == type(None):
                initvector =  self.initial_ket
            theta_optimal, max_right = self.maxRightHanded(initvector, phi=phi_optimal, method = method, bounds = bounds)
            plt.ylabel(r'$r_{max}$ $(\times 10^{-4})$')
        elif value == 'Omega':
            theta_optimal, max_right = self.maxOmega(phi=phi_optimal, method = method, bounds = bounds)
            plt.ylabel(r'$\Omega$ $(\times 10^{-4})$')
            
        #find bounds of plot accourding to zoom
        if zoom == None:
            thetas = np.linspace(0, np.pi, theta_resolution)
            plt.xlim(0,np.pi)
        else:
            if not zoom_on_vector:
                 zoom_on_vector = theta_optimal
            thetas = np.linspace(zoom_on_vector - zoom + shift, zoom_on_vector + zoom + shift, theta_resolution)
            plt.xlim( zoom_on_vector - zoom + shift, zoom_on_vector + zoom + shift)
            
        
        if value == 'lminusr':
            plot_vals = 1 - np.array([self.leftMinusRight(theta, phi_optimal)
                                   for theta in thetas]) 
            
        elif value == 'rmax':
            plot_vals = np.array([-1*self.rightHandedPart(theta, phi_optimal, initvector)
                                   for theta in thetas])
        
        elif value == 'Omega':
            plot_vals = np.array([self.Omega(theta, phi_optimal)
                                   for theta in thetas])
        
        #plot full resonance value
        plt.plot(thetas, plot_vals*factor, color = 'r')

        if vmax != None:
            plt.ylim(0,vmax)
        
        #extra_init_vector to see specific resonance condition solutions over plot of the general resonance condition
        if extra_init_vectors != None:
            extra_thetas = np.array([self.maxRightHanded(extra_init_vector, phi=phi_optimal, method = method, bounds = bounds)[0]
                                    for extra_init_vector in extra_init_vectors])
            print('Extra Thetas = ', extra_thetas)
            extra_thetas_vlines = plt.vlines(extra_thetas, [0], [max(plot_vals)], linestyles = '--', label = 'Specified Initial Vectors', color='lime')

        #plot vlines
        neutrino_flavors = {0:'e', 1:'x', 2:r'\tau'}
        resonant_thetas = []
        for n,k,color in flavor_resonances:
            resonant_thetas.append((self.resonant_theta(phi=phi_optimal, resonance_type = [n,k+3]), n,k,color))
        for theta,n,k,color in resonant_thetas:
            plt.vlines([theta],[0],[factor*max(plot_vals)], linestyles = '--', label = rf'${neutrino_flavors[n]}_L \rightleftharpoons {neutrino_flavors[k]}_R$', color = color)
        
        if max_point:
            max_point_vline =  plt.vlines([theta_optimal],[0],[factor*max(plot_vals)], linestyles = ':', label = 'Max value', color='magenta')
            bounds_vlines =    plt.vlines([bounds[0][0], bounds[0][1]],[0],[1/4*factor*max(plot_vals)], linestyles = '-.', label = 'Bounds', color='orange')
            print("Optimal theta in Range = ", str(theta_optimal))

        if extra_lines != None:
            extra_vlines = plt.vlines(extra_lines, [0], [factor*max(plot_vals)], linestyles = '--', label = 'Extra Lines', color='lime')

        plt.legend(frameon = False)
        plt.minorticks_on()

        if type(savefig) == str: 
            plt.savefig(savefig + '.pdf', dpi=300)
        
        if max_point:
            return theta_optimal
            
        return np.array(resonant_thetas)[:,0]
    
    
    #finds width of lminusr resonance condition
    #works best if limits are reduced to near the resonance band
    def findResonantRegions(self, theta_resolution = 300, phi_optimal = np.pi,
                                          min_dist_between_peaks = 10,
                                          limits = [0,np.pi],
                                          resonance_threshold = 1,
                                          max_peak_count = 6,
                                          method = 'Nelder-Mead',
                                          xtol = 1E-13,
                                          return_arrays = False,
                                          makeplot = False,
                                          printvalues = False,
                                          **kwargs):

        #find approximate maxima of resonance parameter 
        thetas = np.linspace(limits[0], limits[1], theta_resolution)
        resonances = np.array([self.Omega(theta, phi_optimal) for theta in thetas])
        max_thetas_approx = thetas[signal.find_peaks(resonances, distance = min_dist_between_peaks)[0]]
        #use only the max_peak_count largest maxima
        if len (max_thetas_approx) > max_peak_count:
            max_resonances_approx = np.sort([self.Omega(theta, phi_optimal) for theta in max_thetas_approx])[-1*max_peak_count:]
            max_thetas_approx = np.array([theta for theta in max_thetas_approx if self.Omega(theta, phi_optimal) in max_resonances_approx])
        
        #define function for scipy optimization
        def find_intercepts(theta):
            if type(theta) == np.ndarray: #opt is calling in theta as a list ( [theta] ) instead of just theta. This is leading to a ragged nested sequence bug. This fixes it (sloppily)
                theta = theta[0]
            return self.Omega(theta,phi_optimal, negate = True) + resonance_threshold
        
        #find the exact maxima of the adiabaticity azimuthal function
        search_dist = min_dist_between_peaks/theta_resolution*2*np.pi
        max_thetas = np.array([opt.minimize(find_intercepts, x0 = theta_max_approx, method = method, options = {'xtol':xtol},
                                          bounds = [(theta_max_approx-search_dist/2,theta_max_approx+search_dist/2)]).x[0]
                             for theta_max_approx in max_thetas_approx])
        
        if printvalues:
            print()
            print('max_thetas = ', max_thetas)
            print(f'computed Omega (only registered if greater than {resonance_threshold}) = ', [self.Omega(theta,phi_optimal) for theta in max_thetas])
            
        #find locations of intercepts near maxima, if the maxima is above the threshold (so that there is an intercept to the left and right)
        ranges = []
        bounds = []
        optimal_thetas = []
        for theta in max_thetas:
            if find_intercepts(theta) < 0:
                #make sure the two boundaries have different signs so an intercept can be found
                counter = 0
                while find_intercepts(theta + search_dist) < 0 or find_intercepts(theta - search_dist) < 0:
                   search_dist = search_dist*2
                   counter += 1
                   if counter == 3:
                       print('Warning: search_dist is too small or too big to find intercepts. Try changing min_dist_between_peaks')
                       continue
                bound_1 = opt.brentq(find_intercepts, theta, theta + search_dist)
                bound_2 = opt.brentq(find_intercepts, theta, theta - search_dist)
                ranges.append(abs(bound_1-bound_2))
                bounds.append((bound_1, bound_2))
                optimal_thetas.append(theta)
        
        if printvalues:  
            print('individual widths = ', ranges) 
            print()
            
        #total resonant width
        total_resonant_width = np.sum(ranges)
       
        if printvalues:
            print('Total Resonant Width =', str(total_resonant_width), 'Radians') 
      
        #render plots of locations of maxima
        if makeplot == True:
            if len(bounds)>0:
                f, ax = plt.subplots(1,len(bounds), figsize=(5*len(bounds),5), squeeze = False)
    
            for n, bound in enumerate(bounds):
                diff = np.sort(bound)[1] - np.sort(bound)[0]
                plotting_thetas = np.linspace(np.sort(bound)[0] - diff, np.sort(bound)[1] + diff, theta_resolution)
                plotting_Omegas = np.array([self.Omega(theta,phi_optimal) for theta in plotting_thetas])
                ax[0,n].plot(plotting_thetas, plotting_Omegas)
                ax[0,n].axvline(np.sort(bound)[0], color = 'red')
                ax[0,n].axvline(np.sort(bound)[1], color = 'red')
                ax[0,n].set_ylim(0, 2*resonance_threshold)
        if return_arrays:        
            return np.array(optimal_thetas), np.array(ranges)
        else:
            return {'resonant_thetas':optimal_thetas, 'total_width':total_resonant_width}
        

    ###################

    #initial adiabaticity and gradient computations (using simplified conditions)
    #flavor is the flavor of the neutrino that is being considered (0,1,2) default is electron
    def resonantGradAndAdiabaticity(self, phi, flavor = 0, resonance_type = None):
        theta = self.resonant_theta(phi=phi, resonance_type = resonance_type)
        grad_H_L = self.grad_H_L(theta, phi) 
        direction = Basis(theta,phi).n_vector
        grad_along_direction = np.abs(np.tensordot(grad_H_L, direction, axes = ([0],[0]))[flavor,flavor]) #grad[H_L]_ee
        H_LR_along_direction = np.abs(self.H_LR(theta, phi)[flavor,flavor])
        adiabaticity = 2*H_LR_along_direction**2/grad_along_direction
        return grad_along_direction, adiabaticity

    #generates plot of gradient and of adiabaticity along the resonance band, parameterized by phi
    def azimuthalGradientsPlot(self, phi_resolution = 300, savefig=False, vmax = 1E-5, resonance_type = None):
        #factors to scale plots by to avoid the scale number in the corner. Have to manually change the label
        factor_grad = 1E14
        factor_gamma = 1E8
        
        phis = np.linspace(0, 2*np.pi, phi_resolution)
        gradients = np.array([self.resonantGradAndAdiabaticity(phi, resonance_type=resonance_type)[0] for phi in phis])
        adiab     = np.array([self.resonantGradAndAdiabaticity(phi, resonance_type=resonance_type)[1] for phi in phis])        
        
        f, ax = plt.subplots(2,1, figsize=(8,8), sharex = True)
        
        #gradient plot
        ax[0].plot(phis, gradients*factor_grad)
        ax[0].set_ylabel(r'$|\nabla_{\nu} [H_{L}]_{ee}| \ \  (eV^2 \times 10^{-14}) $', fontsize = 14)

        #adiabaticity
        ax[1].plot(phis,adiab*factor_gamma)
        ax[1].set_xlabel(r'$\phi$', fontsize = 14)
        ax[1].set_ylabel(r'$\gamma$  $(\times 10^{-8})$', fontsize = 14)
        ax[1].set_ylim(-vmax/30,vmax)
        ax[1].set_xlim(0,2*np.pi)

        #label x ticks in radians, fractions of pi
        xT=[0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
        xL=[r'$0$', r'$\frac{1}{2}\pi$', r'$\pi$', r'$\frac{3}{2}\pi$', r'$2\pi$']
        plt.xticks(xT, xL, fontsize = 14)

        plt.tight_layout()
        plt.minorticks_on()
        plt.xticks(xT, xL, fontsize = 14)

        if type(savefig) == str: 
            plt.savefig(savefig + '.pdf', dpi=300)
    
    
    #calculates total angular width of resonance band that satisfies adiabaticity> adiabaticity_threshold 
    def findAdiabaticRegions(self, phi_resolution = 200, min_dist_between_peaks = 10,
                                          adiabaticity_threshold = 1, max_peak_count = 2,
                                          method = 'Nelder-Mead',
                                          resonance_type = None,
                                          return_arrays = False,
                                          makeplot = False,
                                          printvalues = False,
                                          **kwargs):
        
        #find approximate maxima of adiabaticity azimuthal function
        phis = np.linspace(0, 2*np.pi, phi_resolution)
        adiabs = np.array([self.resonantGradAndAdiabaticity(phi, resonance_type=resonance_type)[1] for phi in phis])
        max_phis_approx = phis[signal.find_peaks(adiabs, distance = min_dist_between_peaks)[0]]
        
        #use only the max_peak_count largest maxima
        if len (max_phis_approx) > max_peak_count:
            max_adiabs_approx =  np.sort([self.resonantGradAndAdiabaticity(phi, resonance_type=resonance_type)[1] for phi in max_phis_approx])[-1*max_peak_count:]
            max_phis_approx = np.sort(max_phis_approx)[-1*max_peak_count:]
        
        #define function for scipy optimization
        def find_intercepts(phi):
            if type(phi) == np.ndarray: #opt is calling in phi as a list ( [phi] ) instead of just phi. This is leading to a ragged nested sequence bug. This fixes it (sloppily)
                phi = phi[0]
            return -1*self.resonantGradAndAdiabaticity(phi, resonance_type=resonance_type)[1] + adiabaticity_threshold
        
        #find the exact maxima of the adiabaticity azimuthal function
        search_dist = min_dist_between_peaks/phi_resolution*2*np.pi
        max_phis = np.array([opt.minimize(find_intercepts, x0 = phi_max_approx, method = method, options = {'xtol':1E-11},
                                          bounds = [(phi_max_approx-search_dist/2,phi_max_approx+search_dist/2)]).x[0]
                             for phi_max_approx in max_phis_approx])

        if printvalues:
            print()
            print('max_phis = ', max_phis)
            print('computed adiabaticities (only registered if greater than 1) = ', [self.resonantGradAndAdiabaticity(phi, resonance_type=resonance_type)[1] for phi in max_phis])
            
        #find locations of intercepts near maxima, if the maxima is above the threshold (so that there is an intercept to the left and right)
        ranges = []
        bounds = []
        for phi in max_phis:
            if find_intercepts(phi) < 0:
               #make sure the two boundaries have different signs so an intercept can be found
               counter = 0
               while find_intercepts(phi+search_dist) < 0 or find_intercepts(phi-search_dist) < 0:
                   search_dist = search_dist*2
                   counter += 1
                   if counter == 3:
                       print('Warning: search_dist is too small or too big to find intercepts. Try changing min_dist_between_peaks')
                       continue
               
               #find bounds
               bound_1 = opt.brentq(find_intercepts, phi, phi + search_dist)
               bound_2 = opt.brentq(find_intercepts, phi, phi - search_dist)
               ranges.append(abs(bound_1-bound_2))
               bounds.append((bound_1, bound_2))
        
        if printvalues:
            print('individual widths = ', ranges) 
            print()
        
        #total adiabatic width
        total_adiabatic_width = np.sum(ranges)
        if printvalues:
            print('Total Adiabatic Width =', str(total_adiabatic_width), 'Radians') 
      
        #render plots of locations of maxima
        if makeplot == True:
            if len(bounds)>0:
                f, ax = plt.subplots(1,len(bounds), figsize=(5*len(bounds),5), squeeze = False)
    
            for n, bound in enumerate(bounds):
                diff = np.sort(bound)[1] - np.sort(bound)[0]
                plotting_phis = np.linspace(np.sort(bound)[0] - diff, np.sort(bound)[1] + diff, phi_resolution)
                plotting_adiabs = np.array([self.resonantGradAndAdiabaticity(phi, resonance_type=resonance_type)[1] for phi in plotting_phis])
                ax[0,n].plot(plotting_phis, plotting_adiabs)
                ax[0,n].axvline(np.sort(bound)[0], color = 'red')
                ax[0,n].axvline(np.sort(bound)[1], color = 'red')
                ax[0,n].set_ylim(0, 2*adiabaticity_threshold)
        
        if return_arrays:
            return np.array(ranges), np.array(max_phis)
        else:
            return total_adiabatic_width
        
    #find total solid angle at this point that is both resonant and adiabatic
    def solidAngle(self, separate_ranges = False,
                   resonance_type = None,
                   **kwargs # if true, computes a single resonant angle for all adiabatic ones instead of one for each (more efficient)
                   ):
        
        solid_angle = 0
        
        #check resonance exists
        if self.resonant_theta(phi=0, resonance_type = resonance_type) == None:
            return 0
        
        #compute adiabatic ranges and their locations
        adiabatic_ranges, max_phis = self.findAdiabaticRegions(return_arrays=True, resonance_type = resonance_type, **kwargs)
        
        #check if adiabatic ranges is empty
        if adiabatic_ranges.size == 0:
            return 0
        
        #if not separate_ranges, make the following for-loop just loop through the most significant phi, instead of all of them
        if not separate_ranges:
            max_phis = np.array([max_phis[np.argmax(adiabatic_ranges)]])
            adiabatic_ranges = np.array([np.sum(adiabatic_ranges)])
        
        #loop through phis
        for k, max_phi in enumerate(max_phis):
            #find resonant thetas and ranges
            resonant_thetas, resonant_ranges = self.findResonantRegions(
                                                    phi_optimal = max_phi,
                                                    return_arrays=True,
                                                    **kwargs)
            #compute and add on solid angle at this direction
            solid_angle += np.sum(resonant_ranges*adiabatic_ranges[k]*np.sin(resonant_thetas))
       
        return solid_angle
        
#plots H_LR angular distribution for an initial and final time (before, after instability.)
def multi_HLR_Plotter( 
                    t_sim_1, t_sim_2, #time indices for inital and final plot
                    emu_file,         #file location of emu data
                    merger_data_loc,  #file location of merger data
                    location,         #location of point for angular dist
                    p_abs,            
                    theta_resolution, #resolution of angular plot
                    phi_resolution,   #resolution of angular plot
                    resonance_type = 'simplified', #resonance type for resonance band
                    savefig=False,
                    use_gm=True,
                    direction_point=False):
    
    SP1 = SpinParams(t_sim_1,
                    emu_file,
                    merger_data_loc,
                    location,
                    p_abs,
                    resonance_type=resonance_type
                    )
    SP2 = SpinParams(t_sim_2,
                    emu_file,
                    merger_data_loc,
                    location,
                    p_abs,
                    resonance_type=resonance_type
                    )
    
    scalefactor = 1E13 #change labels manually! 
    
    if use_gm==True:
        H_LR_array_1 = np.array([[gm.magnitude(SP1.H_LR(theta, phi))
                                for phi in np.linspace(0, 2*np.pi, phi_resolution)]
                                for theta in np.linspace(0, np.pi, theta_resolution)])
        H_LR_array_2 = np.array([[gm.magnitude(SP2.H_LR(theta, phi))
                                for phi in np.linspace(0, 2*np.pi, phi_resolution)]
                                for theta in np.linspace(0, np.pi, theta_resolution)])
    elif type(use_gm)==list:
        H_LR_array_1 = np.array([[np.abs(SP1.H_LR(theta, phi)[use_gm[0],use_gm[1]])
                                for phi in np.linspace(0, 2*np.pi, phi_resolution)]
                                for theta in np.linspace(0, np.pi, theta_resolution)])
        H_LR_array_2 = np.array([[np.abs(SP2.H_LR(theta, phi)[use_gm[0],use_gm[1]])
                                for phi in np.linspace(0, 2*np.pi, phi_resolution)]
                                for theta in np.linspace(0, np.pi, theta_resolution)])
    else: 
        H_LR_array_1 = np.array([[gm.sum_magnitude(SP1.H_LR(theta, phi)) 
                                for phi in np.linspace(0, 2*np.pi, phi_resolution)]
                                for theta in np.linspace(0, np.pi, theta_resolution)])
        H_LR_array_2 = np.array([[gm.sum_magnitude(SP2.H_LR(theta, phi)) 
                                for phi in np.linspace(0, 2*np.pi, phi_resolution)]
                                for theta in np.linspace(0, np.pi, theta_resolution)])

    resonance_array_1 = np.array([[SP1.resonance(theta,phi)
                                   for phi in np.linspace(0, 2*np.pi, phi_resolution)]
                                   for theta in np.linspace(0, np.pi, theta_resolution)]) 


    resonance_array_2 = np.array([[SP2.resonance(theta,phi)
                                   for phi in np.linspace(0, 2*np.pi, phi_resolution)]
                                   for theta in np.linspace(0, np.pi, theta_resolution)]) 

    f, ax = plt.subplots(1,2, subplot_kw=dict(projection='mollweide'), figsize=(12,4))
    ax[0]
    
    #find vmin, vmax
    vmax = np.max([np.max(H_LR_array_1*scalefactor), np.max(H_LR_array_2*scalefactor)])
    vmin = np.min([np.min(H_LR_array_1*scalefactor), np.min(H_LR_array_2*scalefactor)])
    
    #colorplot
    H_LR_im_1 = ax[0].pcolormesh(np.linspace(-np.pi, np.pi, phi_resolution), 
                            np.linspace(0.5*np.pi, -0.5*np.pi, theta_resolution),
                            H_LR_array_1*scalefactor, 
                            cmap=plt.cm.hot, shading='auto',
                            vmin = vmin, vmax = vmax)

    H_LR_im_2 = ax[1].pcolormesh(np.linspace(-np.pi, np.pi, phi_resolution), 
                            np.linspace(0.5*np.pi, -0.5*np.pi, theta_resolution),
                            H_LR_array_2*scalefactor, 
                            cmap=plt.cm.hot, shading='auto',
                            vmin = vmin, vmax = vmax)



    #resonance 
    res_im_1 = ax[0].contour(np.linspace(-np.pi, np.pi, phi_resolution),
                        np.linspace(0.5*np.pi, -0.5*np.pi, theta_resolution),
                        resonance_array_1, levels=[0.], colors='cyan')
    res_im_2 = ax[1].contour(np.linspace(-np.pi, np.pi, phi_resolution),
                        np.linspace(0.5*np.pi, -0.5*np.pi, theta_resolution),
                        resonance_array_2, levels=[0.], colors='cyan')
        
        
    
    #time text
    ax[0].text(-0.7*np.pi, 0.25*np.pi, rf'$t$ = {SP1.t_seconds*1E9:.1f} ns', backgroundcolor = 'white')
    ax[1].text(-0.7*np.pi, 0.25*np.pi, rf'$t$ = {SP2.t_seconds*1E9:.1f} ns', backgroundcolor = 'white')

    h1,l1 = res_im_1.legend_elements()
    
    

    #add net flux point 
    J_avg_1 = np.array([gm.magnitude(np.average(SP1.J[n], axis = 2)) for n in range(0,4)])
    flux_point_1 = ax[0].scatter([np.arctan2(J_avg_1[2],J_avg_1[1])],[np.arctan2(J_avg_1[3],
                                        (J_avg_1[1]**2+J_avg_1[2]**2)**(1/2))],  label = 'ELN Flux Direction', color='lime')
    
    J_avg_2 = np.array([gm.magnitude(np.average(SP2.J[n], axis = 2)) for n in range(0,4)])        
    flux_point_2 = ax[1].scatter([np.arctan2(J_avg_2[2],J_avg_2[1])],[np.arctan2(J_avg_2[3],
                                        (J_avg_2[1]**2+J_avg_2[2]**2)**(1/2))],  label = 'ELN Flux Direction', color='lime')
    
    #add (electron) neutrino direction point 
    #flow_direction = np.array(self.merger_grid['fn_a(1|ccm)'])[:,self.location[0],self.location[1],self.location[2]]
    #direction_point = ax.scatter([np.arctan2(flow_direction[1],flow_direction[0])],[np.arctan2(flow_direction[2], (flow_direction[0]**2+flow_direction[1]**2)**(1/2))],  label = 'Neutrino Flow Direction', color='magenta')

    f.legend([h1[0], flux_point_2], [r"$e\rightarrow e$ resonance", r"$|J^{i}|$ Direction"], loc = (0.435,0.85))
        
        
    #axes
    yT=[np.pi/2, np.pi/3, np.pi/6, 0, -np.pi/6, -np.pi/3, -np.pi/2]
    yL=['0', r'$\frac{1}{6}\pi$', r'$\frac{1}{3}\pi$',r'$\frac{1}{2}\pi$',
        r'$\frac{2}{3}\pi$',r'$\frac{5}{6}\pi$',
        r'$\pi$']
    ax[0].set_yticks(yT, yL)
    ax[1].set_yticks(yT, yL)

    xT=[ -2*np.pi/3, -np.pi/3, 0, np.pi/3, 2*np.pi/3]
    xL=[  r'$\frac{1}{3}\pi$', r'$\frac{2}{3}\pi$',
          r'$\pi$',r'$\frac{4}{3}\pi$',
          r'$\frac{5}{3}\pi$']
    ax[0].set_xticks([], [])
    ax[1].set_xticks([], [])
    
    ax[0].set_ylabel(r'$\theta$', rotation = 0, labelpad = 8, fontsize = 14)
    f.tight_layout(pad = 1.6)
    f.colorbar(H_LR_im_2, label=r"$|H_{LR}| \ (eV \times 10^{-13})$", location = 'bottom',ax=ax.ravel().tolist(),
               pad = 0.1, aspect = 30)

    if type(savefig) == str: 
        plt.savefig(savefig + '.pdf', dpi=300)

    plt.show()
    return



    
    
