import h5py
import numpy as np
from constants import hbar, c, M_p, G, M_2flavor
from basis import Basis
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#Calculates values from merger data, and generates adiabaticity / resonance contour plots.
#merger_data_loc, unrotated_data_loc is the location for the merger data and unrotated merger data
class Merger_Grid:
    def __init__(self, zval, merger_data_loc, p_abs):
        
        self.zval = zval
            
        #rotated data has ELN flux along z
        self.merger_grid = h5py.File(merger_data_loc, 'r')
        
        #Electron fraction, baryon n density
        self.rho=np.array(self.merger_grid['rho(g|ccm)'])[:,:,self.zval] #g/cm^3 (baryon mass density)
        self.Ye=np.array(self.merger_grid['Ye'])[:,:,self.zval]
        self.n_b=self.rho/M_p*(hbar**3 * c**3)#eV^3 (baryon number density)

        #x,y axes
        self.x_km = np.array(self.merger_grid['x(cm)'])[:,:,0] / 1e5
        self.y_km = np.array(self.merger_grid['y(cm)'])[:,:,0] / 1e5
        self.z_km = np.array(self.merger_grid['z(cm)'])[0,0,:] / 1e5
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
        
        self.J_e = (hbar**3 * c**3)*np.array([self.merger_grid['n_e(1|ccm)'],
                                             self.merger_grid['fn_e(1|ccm)'][0],
                                             self.merger_grid['fn_e(1|ccm)'][1],
                                             self.merger_grid['fn_e(1|ccm)'][2]])[:,:,:,self.zval]
                      
        self.J_a = (hbar**3 * c**3)*np.array([self.merger_grid['n_a(1|ccm)'],
                                              self.merger_grid['fn_a(1|ccm)'][0],
                                              self.merger_grid['fn_a(1|ccm)'][1],
                                              self.merger_grid['fn_a(1|ccm)'][2]])[:,:,:,self.zval]
        
        #just the ee component of J
        self.J = self.J_e - self.J_a
        
        #max, min values of the LHS of the resonance condition (ADD OTHER PARTS)
        self.mag_F=np.sqrt(np.sum(self.J[1:4]**2, axis=0))
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
        self.neg_resonance_sign[self.negative_resonance]=1
        
        #both must be true for resonance to exist
        self.resonance_sign = self.neg_resonance_sign*self.pos_resonance_sign
        
        #resonance and discriminant satisfied for some direction
        self.both_conditions=self.resonance_sign*self.discriminant_sign
    
        
    #x,y are coordinates of chosen points to examine in later sections. should be list of same length as zval with -1 for no point on the nth plot
    #xmin, xmax, ymin, ymax are the limits of the plot
    #rect_xmin, rect_xmax, rect_ymin, rect_ymax are the limits of the adiabaticity computation region (green rectangle) must be list of same length as zval
    def contour_plot(self, savefig = False, points = None, #points = [[x1,y1,z1],[x2,y2,z2],...] and first one should be example point marked x
                     xmin = 30, xmax = 170, ymin = 30, ymax = 170,
                     rect_xmin = None, rect_xmax = None, rect_ymin = None, rect_ymax = None):
        zval = self.zval
        zval_km = self.z_km[zval]
        n = len(zval)

        ELNcolor = '#005070'
        resonancecolor = '#FFB380' 
        bothcolor_fill = '#E61F39'
        adiabcolor = 'yellow'
        f,ax = plt.subplots(1,n,figsize=(n*6,6), sharex = True, sharey = True, squeeze = False)
        for k in range(n):
            #FFI contour
            ax[0,k].contourf(self.x_km,self.y_km, self.discriminant_sign[:,:,k],
                                   levels=[-1,0.5,2], alpha=1, colors=[[1,0.98,0.95],ELNcolor])
      
            #resonance val
            ax[0,k].text(0.85*self.x_km[xmin,0],0.8*self.y_km[0,ymax],rf'$z$ = {zval_km[k]:.1f} km', 
                        bbox = dict(edgecolor = 'black', facecolor = 'white', alpha = 0.8, boxstyle = 'round', pad = 0.2))
            
            ax[0,k].contourf(self.x_km[:,:],self.y_km[:,:],self.resonance_sign[:,:,k], levels=[0.5,1], colors=[resonancecolor] )

            #both conditions
            ax[0,k].contourf(self.x_km[:,:],self.y_km[:,:],self.both_conditions[:,:,k], levels=[0.5,1], colors=[bothcolor_fill] )
            
            #adiabaticity computation region
            if type(rect_xmin) != type(None) and type(rect_xmax) != type(None) and type(rect_ymin) != type(None) and type(rect_ymax) != type(None):
                if type(rect_xmin[k]) != type(None) and type(rect_xmax[k]) != type(None) and type(rect_ymin[k]) != type(None) and type(rect_ymax[k]) != type(None):
                    ax[0,k].add_patch(plt.Rectangle((self.x_km[rect_xmin[k],rect_ymin[k]],self.y_km[rect_xmin[k],rect_ymin[k]]),
                                    self.x_km[rect_xmax[k],rect_ymax[k]]-self.x_km[rect_xmin[k],rect_ymin[k]],
                                    self.y_km[rect_xmax[k],rect_ymax[k]]-self.y_km[rect_xmin[k],rect_ymin[k]],
                                    linewidth=3,edgecolor=adiabcolor, facecolor='none', linestyle = 'dashed'))

            ax[0,k].grid(False)
            
        #add selected point markers
        if type(points) != type(None):
            letters = ['x', '$A$', '$B$', '$C$', '$D$', ]
            for k,point in enumerate(points):
                x, y, zi = point
                ax[0,zi].scatter(self.x_km[x,y], self.y_km[x,y],s=120, c = 'lime', marker = letters[k])
        plt.tight_layout()
        
        #legend
        proxy = [plt.Rectangle((1, 1), 2, 2, fc=ELNcolor),
                 plt.Rectangle((1, 1), 2, 2, fc=resonancecolor),
                 plt.Rectangle((1, 1), 2, 2, fc=bothcolor_fill),
                 (mpatches.Patch(color='gray'),
                  plt.Rectangle((0, 0), 1,1, fill = False, ec=adiabcolor, linestyle = 'dashed', linewidth = 3))]
        
        ax[0,-1].legend(proxy, ["Fast Flavor Instability Exists", 
                             "Spin-flip Resonance Exists",
                             "Both Conditions Satisfied",
                             "Adiabaticity Computed",
                             ],
                             loc = 'upper right', 
                             bbox_to_anchor = (1.,-0.13),
                             ncol = 4,
                             fontsize = 20,
                            frameon = False,
                             )
                             
        
        
       
        plt.tick_params(axis='both',which="both", direction="in",top=True,right=True)
        plt.minorticks_on()

        
        ax[0,0].set_xlim([self.x_km[xmin,ymin],self.x_km[xmax,ymax]])
        ax[0,0].set_ylim([self.y_km[xmin,ymin],self.y_km[xmax,ymax]])
        middle_n = n//2
        ax[0,middle_n].set_xlabel(r'$x$-coordinate (km)', fontsize = 20)
        ax[0,0].set_ylabel(r'$y$-coordinate (km)', fontsize = 20)
        #ax[0,middle_n].set_title('Resonant Spin-Flip and Fast-Flavor Instability Regions', fontsize = 16, pad = 20,)

        if type(savefig) == str: 
            plt.savefig(savefig + '.png', dpi=300, bbox_inches = 'tight')

