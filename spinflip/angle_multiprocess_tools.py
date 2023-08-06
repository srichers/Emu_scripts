### Multiprocessing tools for solid angle calculations ###
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../data_reduction")
import numpy as np
import h5py
from tqdm.notebook import tqdm
from constants import hbar, c, M_p, M_3flavor, G
import diagonalizer as dg
import spin_flip_tools as sft
from matrix import visualizer
from merger_grid import Merger_Grid
from matrix import trace_matrix, dagger
from basis import Basis
import matplotlib as mpl
import matplotlib.pyplot as plt
import gellmann as gm
from scipy import signal
from scipy import optimize as opt
import multiprocessing as mp
from multiprocessing import Pool
from four_current import four_current, read_gradients

#compute angle quantities at a single point
#location = (x,y,z)
def angle_at_point(args):
    location = args["location"]
    assert location[0] in args["x_range"], f"x = {location[0]} out of range ({x_range[0]}, {x_range[-1]}))"
    assert location[1] in args["y_range"], f"y = {location[1]} out of range ({y_range[0]}, {y_range[-1]}))"
    assert location[2] in args["zs"], f"z = {location[2]} out of range ({zs[0]}, {zs[-1]}))"
    #write file location and initialize SpinParams
    emu_filename = args["emu_data_loc"] + "i{:03d}".format(location[0])+"_j{:03d}".format(location[1])+"_k{:03d}".format(location[2])+"/allData.h5"
    SP = sft.SpinParams(t_sim = args["it"], 
                        emu_file = emu_filename,
                        merger_data_loc = args["merger_data_file"],
                        gradient_filename = args["gradient_data_file"],
                        location = location,
                        p_abs = args["p_abs"])
    
    #check if there is resonance 
    if SP.resonant_theta(phi=0) == None:
        return
    else: #compute quantity
        if   args["method"] == 'solid angle':
            return SP.solidAngle(separate_ranges=args["separate_ranges"], **args["kwargs"])
        elif args["method"] == 'adiabatic angle': 
            return SP. findAdiabaticRegions(**args["kwargs"])
        elif args["method"] == 'resonant angle':
            return SP.findResonantRegions(**args["kwargs"])['total_width']




#class for computing angles, so I can store arguments without feeding directly into the function we'll iterate on
class Angles:
    def __init__(self, 
                it,
                xy_limits, #if None, use all data available from gradient file
                zs, 
                method, # = solid angle, adiabatic angle, resonant angle
                emu_data_loc,
                merger_data_file,
                gradient_data_file, 
                p_abs, 
                separate_ranges = False,
                **kwargs):
        self.it = it
        self.zs = zs
        self.method = method
        self.emu_data_loc = emu_data_loc
        self.merger_data_file = merger_data_file
        self.gradient_data_file = gradient_data_file
        self.p_abs = p_abs
        self.separate_ranges = separate_ranges
        self.kwargs = kwargs
        
        #find limits from Gradients object
        self.Gr = sft.Gradients(gradient_data_file,merger_data_file)
        if type(xy_limits) == (None):
            self.xy_limits = self.Gr.limits[0:2,:]
        else:
            assert(xy_limits[0,0] >= self.Gr.limits[0,0] and xy_limits[0,1] <= self.Gr.limits[0,1]), "x limits out of range"
            assert(xy_limits[1,0] >= self.Gr.limits[1,0] and xy_limits[1,1] <= self.Gr.limits[1,1]), "y limits out of range"
            self.xy_limits = xy_limits
        for z in zs: #check z is within the computed datapoints
            assert(z >= self.Gr.limits[2,0] and z <= self.Gr.limits[2,1]) 
        
        #intialize ranges for the computation
        self.x_range = range(self.xy_limits[0,0], self.xy_limits[0,1])
        self.y_range = range(self.xy_limits[1,0], self.xy_limits[1,1])
     
        
    #multiprocess above function for multiple points
    def multiprocess_angles(self,
                            n_cores = 32,
                            h5_filename = None): #set to filename to store to
        xs = self.x_range
        ys = self.y_range
        zs = self.zs
        #define iterable going over ranges
        iterable = [ {"location":[x,y,z],
                      "x_range":self.x_range,
                      "y_range":self.y_range,
                      "zs":self.zs,
                      "emu_data_loc":self.emu_data_loc,
                      "it":self.it,
                      "merger_data_file":self.merger_data_file,
                      "gradient_data_file":self.gradient_data_file,
                      "p_abs":self.p_abs,
                      "separate_ranges":self.separate_ranges,
                      "kwargs":self.kwargs,
                      "method":self.method
                      } for x in xs for y in ys for z in zs]
        
        #compute angles at each point in ranges using multiprocess
        with Pool(n_cores) as pool:
            output_angles = pool.map(angle_at_point, iterable)

        #reshape output
        output_angles = np.array(output_angles).reshape(len(xs),len(ys),len(zs))
        
        if h5_filename:
            #store in h5 file
            with h5py.File(h5_filename+'.h5', 'w') as f:
                f.create_dataset('solid_angles', data = output_angles)
                f.close()
        #return
        return output_angles


    #plot processed angles
    def solid_angles_plot(self,
                          angles = 'compute', #output of last function
                          vmin = None,
                          vmax = None,
                          savefig = False
                          ):
        
        if angles == 'compute':
            angles = self.multiprocess_angles()
        
        #prepare plot axes
        xdim  = 1E-5*self.Gr.merger_grid['x(cm)'][self.xy_limits[0,0]:self.xy_limits[0,1]+1,
                                                    self.xy_limits[1,0]:self.xy_limits[1,1]+1,
                                                    self.zs[0]]
        ydim  = 1E-5*self.Gr.merger_grid['y(cm)'][self.xy_limits[0,0]:self.xy_limits[0,1]+1,
                                                    self.xy_limits[1,0]:self.xy_limits[1,1]+1,
                                                    self.zs[0]]
        zs_km = 1E-5*self.Gr.merger_grid['z(cm)'][0,0,self.zs]
        
        #vmin, vmax 
        if vmin != None:
            vmin = np.log10(vmin)
        if vmax != None:
            vmax = np.log10(vmax)
        n = len(self.zs)
        
        #plot
        f,ax = plt.subplots(1,n,figsize=(n*6,8), sharex = True, sharey = True, squeeze = False,)
        for k in range(n):
            #colorplot
            im = ax[0,k].pcolormesh(xdim, ydim, np.log10(angles[:,:,k]),
                                        vmin = vmin, vmax = vmax, 
                                        cmap = 'YlGnBu_r')
            
            #zval text
            ax[0,k].text((xdim[0, 0] - xdim[-1,0])*0.99 + xdim[-1,0],
                         (ydim[0,-1] - ydim[ 0,0])*0.95 + ydim[ 0,0],
                            rf'$z$ = {zs_km[k]:.1f} km', 
                            backgroundcolor = 'white')
    
        #colorbar
        plt.tight_layout()
        f.colorbar(im, label=r'Solid Angle (log)', location = 'bottom',ax=ax.ravel().tolist(), pad = 0.1,aspect=30)
        
        #axis labels
        middle_n = n//2
        ax[0,middle_n].set_xlabel(r'$x$-coordinate (km)', fontsize = 14)
        ax[0,0].set_ylabel(r'$y$-coordinate (km)', fontsize = 14)
        #ax[0,middle_n].set_title('Average Adiabaticity in Resonant Directions at Each Cell', fontsize = 16, pad = 20,)

        #savefig
        if type(savefig) == str: 
            f.savefig(savefig + '.png', dpi=300, bbox_inches = 'tight')
          
        if angles == 'compute':
            return f,ax,angles  
        else:
            return f,ax
            
            
def main(args):
    #parse arguments
    it = args.it
    y_limits = args.y_limits
    x_limits = args.x_limits
    if type(x_limits) == type(None) or type(y_limits) == type(None):
        xy_limits = None
    else:
        xy_limits = np.array([x_limits,y_limits])
    zs = args.zs
    method = args.method
    emu_data_loc = args.emu_data_loc
    merger_data_file = args.merger_data_file
    gradient_data_file = args.gradient_data_file
    p_abs = args.p_abs
    separate_ranges = args.separate_ranges
    savefig = args.savefig
    vmin = args.vmin
    vmax = args.vmax
    h5_filename = args.h5_filename
    kwargs = args.kwargs
    
    #initialize angles object
    Angles_obj = Angles(it, 
                        xy_limits,
                        zs,
                        method, # = solid angle, adiabatic angle, resonant angle
                        emu_data_loc,
                        merger_data_file,
                        gradient_data_file, 
                        p_abs, 
                        separate_ranges = separate_ranges,
                        **kwargs)
    
    #compute angles
    angles = Angles_obj.multiprocess_angles(h5_filename)
    
    #plot angles
    Angles_obj.solid_angles_plot(angles = angles,
                             vmin = vmin,
                             vmax = vmax,
                             savefig = savefig,
                             )
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--it", help="timestep of the simulation to analyze", type=int, default=0)
    parser.add_argument("-x", "--x_limits", help="limits of x range to analyze", type=int, nargs=2, default=[75,83])
    parser.add_argument("-y", "--y_limits", help="limits of y range to analyze", type=int, nargs=2, default=[67,77])
    parser.add_argument("-z", "--zs", help="z values to analyze", type=int, nargs='+', default=[97,98,99])
    parser.add_argument("-m", "--method", help="method to use for analysis", type=str, default='solid angle')
    parser.add_argument("-e", "--emu_data_loc", help="location of emu data", type=str, default="/mnt/scratch/shared/3-Henry_NSM_box/")
    parser.add_argument("-f", "--merger_data_file", help="location of merger data", type=str, default="/mnt/scratch/shared/2-orthonormal_distributions/model_rl0_orthonormal_rotated.h5")
    parser.add_argument("-g", "--gradient_data_file", help="location of gradient data", type=str, default="/mnt/scratch/shared/4-gradients/gradients_start.h5")
    parser.add_argument("-p", "--p_abs", help="neutrino momentum", type=float, default=1e7)
    parser.add_argument("-s", "--separate_ranges", help="separate ranges for solid angle calculation", action="store_true", default=False)
    parser.add_argument("-v", "--vmin", help="vmin for plot", type=float, default=None)
    parser.add_argument("-w", "--vmax", help="vmax for plot", type=float, default=None)
    parser.add_argument("-o", "--savefig", help="where to save plot", type=str, default='solidangleplot')
    parser.add_argument("-k", "--h5_filename", help="where to save h5 file", type=str, default=None)
    parser.add_argument("-a", "--kwargs", help="kwargs for solid angle calculation", type=dict, default={})
    args = parser.parse_args()
    main(args)
