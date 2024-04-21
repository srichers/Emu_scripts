import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.optimize
import matplotlib as mpl


##########
# INPUTS #
##########
# unrotated initial conditions
Nee = 1.421954234999705e+33
Neebar = 1.9146237131657563e+33
Nxx = 1.9645407875568215e+33
Nxxbar = 1.9645407875568215e+33
fee = np.array([0.0290911, 0.08385218, 0.14643947])
feebar = np.array([0.08343537, 0.02458396, 0.34272995])
fxx = np.array([0.20897386, -0.06627688, 0.49461718])
fxxbar = np.array([0.20897386, -0.06627688, 0.49461718])

# initial conditions
#Nee = 1.421954234999705e+33
#Neebar = 1.9146237131657563e+33
#Nxx = 1.9645407875568215e+33
#Nxxbar = 1.9645407875568215e+33
#fee = np.array([0.0974572, 0.04217632, -0.13433261])
#feebar = np.array([0.07237959, 0.03132354, -0.3446878])
#fxx = np.array([-0.02165833, 0.07431613, -0.53545951])
#fxxbar = np.array([-0.02165833, 0.07431613, -0.53545951])


# first plateau
#Nee = 9.164e+32 #cm^{-3}
#Nxx = 9.967e+32
#Neebar = 14.090e+32
#Nxxbar = 9.967e+32
#fee = np.array([0.0149, 0.0125, -0.0778])
#fxx = np.array([0.0185, 0.0128, -0.0430])
#feebar = np.array([0.0105, 0.0090, -0.1100])
#fxxbar = np.array([0.0173, 0.0117, -0.0773])

# second plateau
#Nee = 8.959e+32
#Neebar = 13.886e+32
#Nxx = 10.171e+32
#Nxxbar = 10.172e+32
#fee = np.array([0.0182, 0.0138, -0.0735])
#feebar = np.array([0.0104, 0.0089, -0.1091])
#fxx = np.array([0.0139, 0.0100, -0.0375])
#fxxbar = np.array([0.0175, 0.0117, -0.0790])

mpl.rcParams['font.size'] = 22
mpl.rcParams['font.family'] = 'serif'
mpl.rc('text', usetex=True)
mpl.rcParams['xtick.major.size'] = 7
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['xtick.major.pad'] = 8
mpl.rcParams['xtick.minor.size'] = 4
mpl.rcParams['xtick.minor.width'] = 2
mpl.rcParams['ytick.major.size'] = 7
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.minor.size'] = 4
mpl.rcParams['ytick.minor.width'] = 2
mpl.rcParams['axes.linewidth'] = 2

def mag(u):
    return np.sqrt(np.sum(u**2, axis=0))

def rotate(R,p):
    return np.sum(R * p[np.newaxis,:], axis=1)
    
def get_rotation_matrix(fn_e, fn_a):
    # get the angle of rotation to the z axis
    fn_eln = fn_e - fn_a
    fn_eln_mag = np.sqrt(np.sum(fn_eln**2,axis=0))
    costheta = fn_eln[2] / mag(fn_eln)
    costheta_2 = np.sqrt((1+costheta)/2.)
    sintheta_2 = np.sqrt((1-costheta)/2.)
    one = costheta_2**2 + sintheta_2**2
    
    # get axis of rotation (the axis normal to the e and a fluxes)
    u = np.cross(fn_eln, np.array([0,0,1]))
    u /= mag(u)
    
    # get rotation quaternion
    q = np.array([costheta_2,
                  sintheta_2 * u[0],
                  sintheta_2 * u[1],
                  sintheta_2 * u[2]
    ])
    
    # construct rotation matrix ala wikipedia
    R = np.zeros((3,3))
    for i in range(3):
        # R[0,0] = q0^2 + q1^2 - q2^2 - q3^2
        #  = costheta^2 + 2q1^2 - sintheta^2 (assuming u is a unit vector)
        # Let the loop over j take into accout the 2q1^2
        R[i,i] = 2.*costheta_2**2 - 1.
        
        for j in range(3):
            R[i,j] += 2.*q[i+1]*q[j+1] # indexing = q is size 4
    R[0,1] -= 2.*q[0]*q[2+1]
    R[1,0] += 2.*q[0]*q[2+1]
    R[0,2] += 2.*q[0]*q[1+1]
    R[2,0] -= 2.*q[0]*q[1+1]
    R[1,2] -= 2.*q[0]*q[0+1]
    R[2,1] += 2.*q[0]*q[0+1]
    
    # rotate the eln flux to check that the rotation matrix works as intended
    #fn_eln = rotate(R, fn_eln)
    #print("rotated eln flux magnitude / old eln flux magnitude = " , mag(fn_eln) / fn_eln_mag)

    return R

###################
# MINERBO CLOSURE #
###################
# compute inverse Langevin function for closure
# Levermore 1984 Equation 20
def function(Z,fluxfac):
    return (1./np.tanh(Z) - 1./Z) - fluxfac
def dfunctiondZ(Z,fluxfac):
    return 1./Z**2 - 1./np.sinh(Z)**2
def get_Z(fluxfac):
    initial_guess = 1
    Z = scipy.optimize.fsolve(function, initial_guess, fprime=dfunctiondZ, args=fluxfac)[0]
    residual = np.max(np.abs(function(Z,fluxfac)) )
    # when near isotropic, use taylor expansion
    # f \approx Z/3 - Z^3/45 + O(Z^5)
    return Z, residual



# make a plot of the distributions
def distribution(N, Z,theta):
    mu = np.cos(theta)
    return N/(4.*np.pi) * Z/np.sinh(Z) * np.exp(Z*mu)

def makeplot(N,Nbar,f,fbar, label):
    print("########## ")
    print("Computing for "+label)
    print("########## ")
    print("N = ", N, Nbar)
    print("F    = ", f)
    print("Fbar = ", fbar)
    
    fluxfac = mag(f)
    fluxfacbar = mag(fbar)
    print("fluxfac = ",fluxfac, fluxfacbar)
    
    Z, residual = get_Z(fluxfac)
    Zbar, residualbar = get_Z(fluxfacbar)
    print()
    print("Z = ", Z, Zbar)
    print("residual = ", residual, residualbar)
    
    
    # get the crossing descriminant
    fhat = f / mag(f)
    fhatbar = fbar / mag(fbar)
    costheta = np.sum(fhat*fhatbar)
    eta = np.log( (N*Z/np.sinh(-Z)) / (Nbar*Zbar/np.sinh(-Zbar)) )
    beta = Zbar**2 + Z**2 - 2.*Z*Zbar*costheta
    gamma = -2.*(Zbar * costheta - Z)
    epsilon = eta**2 - Zbar**2 * (1.-costheta**2)
    descriminant = -epsilon/beta + (gamma*eta/(2.*beta))**2

    # determine the original eln direction
    eln = N*f - Nbar*fbar
    theta_eln = np.pi/2 - np.arccos(-eln[2] / np.linalg.norm(eln))
    print("theta_eln = ",theta_eln)
    
    # get rotation matrix for plotting
    R = get_rotation_matrix(f*N, fbar*Nbar)
    print()
    print("Rotation matrix:")
    print(R)
    f = -rotate(R, f)
    fbar = -rotate(R, fbar)
    eln = N*f - Nbar*fbar
    print("F_rotated    = ",f)
    print("Fbar_rotated = ",fbar)
    print("ELN_rotated = ",eln)

    print()
    print("descriminant = ",descriminant)
    
    fig,ax = plt.subplots(subplot_kw={'projection':'polar'})
    
    theta = np.arccos(fhat[2]) + theta_eln
    thetabar = np.arccos(fhatbar[2]) + theta_eln

    thetaplot = np.arange(0,2.*np.pi,np.pi/100)
    dist    = distribution(N   , Z   , thetaplot-theta   )
    distbar = distribution(Nbar, Zbar, thetaplot-thetabar)
    
    plt.plot( thetaplot, dist, color='blue', label=r"$\nu$")
    plt.scatter(theta, np.max(dist), color='blue')

    plt.plot( thetaplot, distbar, color='red', label=r"$\bar{\nu}$")
    plt.scatter(thetabar, np.max(distbar), color='red')

    plt.arrow(theta,0,0,N*fluxfac, color='blue')
    plt.arrow(thetabar+np.pi,0,0, Nbar*fluxfacbar, color='red')
    plt.arrow(theta_eln+np.pi, 0,0,np.abs(eln[2]), color="purple", linewidth=2)

    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])

    lim = np.max([Nbar*fluxfacbar,N*fluxfac, np.max(dist),np.max(distbar)]) * 1.05
    ax.set_ylim(0,lim)

    if descriminant>0:
        print("YES crossing")
        theta1_cross = np.arccos(-gamma*eta / (2.*beta) + np.sqrt(descriminant))
        theta2_cross = np.arccos(-gamma*eta / (2.*beta) - np.sqrt(descriminant))
        #plt.arrow(theta+theta1_cross, 0,0,N/10., color='purple')
        #plt.scatter(theta+theta2_cross, distribution(N,Z,theta2_cross), color='purple')
        #plt.scatter(theta-theta1_cross, distribution(N,Z,theta1_cross), color='purple')
        #plt.arrow(theta-theta2_cross, 0,0,N/10., color='purple')
    else:
        print("NO crossing")
        
    ax.legend()
    
    plt.savefig(label+".pdf")

    plt.clf()
    plt.axhline(0,color="k",linestyle="--")
    plt.axvline(theta_eln/np.pi, color="purple")
    plt.text(theta_eln/np.pi, -0.05, "net ELN direction", color='purple', rotation=-90)
    plt.plot(thetaplot/np.pi, (dist-distbar)/(N+Nbar), color='purple')
    plt.plot(thetaplot/np.pi, (dist)/(N+Nbar), color='blue')
    plt.plot(thetaplot/np.pi, (-distbar)/(N+Nbar), color='red')
    plt.xlim(0,2)
    plt.xlabel(r"$\theta/\pi$")
    plt.ylabel(r"$f_{\nu_e} - f_{\bar{\nu}_e}$")
    plt.minorticks_on()
    plt.gca().yaxis.set_tick_params(which='both', direction='in', right=True,top=True)
    plt.gca().xaxis.set_tick_params(which='both', direction='in', bottom=True, top=True)
    plt.savefig("eln.pdf",bbox_inches='tight')

def color_plot(Nee, Neebar, Nxx, Nxxbar, fee, feebar, fxx, fxxbar):
    ffee = mag(fee)
    ffxx = mag(fxx)
    ffeebar = mag(feebar)
    ffxxbar = mag(fxxbar)

    Zee, r = get_Z(ffee)
    Zxx, r = get_Z(ffxx)
    Zeebar, r = get_Z(ffeebar)
    Zxxbar, r = get_Z(ffxxbar)

    feehat = fee / mag(fee)
    fxxhat = fxx / mag(fxx)
    feebarhat = feebar / mag(feebar)
    fxxbarhat = fxxbar / mag(fxxbar)

    thetagrid = np.arange(0,np.pi,np.pi/100.)
    phigrid = np.arange(0,2.*np.pi, np.pi/100.)
    x = np.array([[ np.sin(theta)*np.cos(phi) for phi in phigrid] for theta in thetagrid])
    y = np.array([[ np.sin(theta)*np.sin(phi) for phi in phigrid] for theta in thetagrid])
    z = np.array([[ np.cos(theta)             for phi in phigrid] for theta in thetagrid])

    theta_ee    = np.arccos(x*feehat[0] + y*feehat[1] + z*feehat[2])
    theta_xx    = np.arccos(x*fxxhat[0] + y*fxxhat[1] + z*fxxhat[2])
    theta_eebar = np.arccos(x*feebarhat[0] + y*feebarhat[1] + z*feebarhat[2])
    theta_xxbar = np.arccos(x*fxxbarhat[0] + y*fxxbarhat[1] + z*fxxbarhat[2])

    ex_eln = distribution(Nee, Zee, theta_ee) - distribution(Nxx, Zxx, theta_xx) - distribution(Neebar, Zeebar, theta_eebar) + distribution(Nxxbar, Zxxbar, theta_xxbar)
    lim = max(np.max(ex_eln),-np.min(ex_eln))
    plt.clf()
    plt.pcolormesh(phigrid/np.pi, thetagrid/np.pi, ex_eln, cmap='seismic', vmin=-lim, vmax=lim)
    if(np.min(ex_eln)*np.max(ex_eln)<0):
        plt.contour(phigrid/np.pi, thetagrid/np.pi, ex_eln, levels=(0), colors="k")
    plt.xlabel(r"$\phi/\pi$")
    plt.ylabel(r"$\theta/\pi$")
    plt.savefig("ex_eln.pdf")
    
#color_plot(Nee, Neebar, Nxx, Nxxbar, fee, feebar, fxx, fxxbar)
makeplot(Nee, Neebar, fee, feebar, "eebar")
#makeplot(Nxx, Nxxbar, fxx, fxxbar, "xxbar")
