import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.optimize
import matplotlib as mpl


##########
# INPUTS #
##########

test_np = 3

# define arrays
arr_Nee = np.empty([test_np])
arr_Neebar = np.empty([test_np])
arr_Nxx = np.empty([test_np])
arr_Nxxbar = np.empty([test_np])
arr_fee = np.empty([test_np,3])
arr_feebar = np.empty([test_np,3])
arr_fxx = np.empty([test_np,3])
arr_fxxbar = np.empty([test_np,3])

# initial conditions: fid.
arr_Nee[0] = 4.89e+32
arr_Neebar[0] = 4.89e+32
arr_Nxx[0] = 0.0
arr_Nxxbar[0] = 0.0
arr_fee[0,:] = np.array([0.0, 0.0, 1.0/3.0])
arr_feebar[0,:] = np.array([0.0, 0.0, -1.0/3.0])
arr_fxx[0,:] = np.array([0.0, 0.0, 0.0])
arr_fxxbar[0,:] = np.array([0.0, 0.0, 0.0])

# initial conditions: 90d
arr_Nee[1] = 4.89e+32
arr_Neebar[1] = 4.89e+32
arr_Nxx[1] = 0.0
arr_Nxxbar[1] = 0.0
arr_fee[1,:] = np.array([0.0, 0.0, 1.0/3.0])
arr_feebar[1,:] = np.array([0.0, 1.0/3.0, 0.0])
arr_fxx[1,:] = np.array([0.0, 0.0, 0.0])
arr_fxxbar[1,:] = np.array([0.0, 0.0, 0.0])

# initial conditions: 2/3
arr_Nee[2] = 4.89e+32
arr_Neebar[2] = 3.26e+32
arr_Nxx[2] = 0.0
arr_Nxxbar[2] = 0.0
arr_fee[2,:] = np.array([0.0, 0.0, 0.0])
arr_feebar[2,:] = np.array([0.0, 0.0, -1.0/3.0])
arr_fxx[2,:] = np.array([0.0, 0.0, 0.0])
arr_fxxbar[2,:] = np.array([0.0, 0.0, 0.0])


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

def makepolar(N,Nbar,f,fbar, label):
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
    #fhat = f / mag(f)
    #fhatbar = fbar / mag(fbar)
    if fluxfac != 0.0:
        fhat = f / mag(f)
    else:
        fhat = f
    if fluxfacbar != 0.0:
        fhatbar = fbar / mag(fbar)
    else:
        fhatbar = fbar
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
    #R = get_rotation_matrix(f*N, fbar*Nbar)
    #print()
    #print("Rotation matrix:")
    #print(R)
    #f = -rotate(R, f)
    #fbar = -rotate(R, fbar)
    #eln = N*f - Nbar*fbar
    #print("F_rotated    = ",f)
    #print("Fbar_rotated = ",fbar)
    #print("ELN_rotated = ",eln)

    f = -f
    fbar = -fbar
    eln = N*f - Nbar*fbar
    fhat = -fhat
    fhatbar = -fhatbar
    theta_eln = np.pi/2 - np.arccos(-eln[2] / np.linalg.norm(eln))

    print()
    print("descriminant = ",descriminant)

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

    theta = np.arccos(fhat[2]) + theta_eln
    thetabar = np.arccos(fhatbar[2]) + theta_eln
    
    thetaplot = np.arange(0,2.*np.pi,np.pi/100)
    dist    = distribution(N   , Z   , thetaplot-theta   )
    distbar = distribution(Nbar, Zbar, thetaplot-thetabar)

    return thetaplot, theta, thetabar, theta_eln, dist, distbar, fluxfac, fluxfacbar, eln





#fig, axes = plt.subplots(2,1, figsize=(6,10))
#plt.subplots_adjust(hspace=0)
#plt.subplots_adjust(vspace=1)
#fig = plt.figure(figsize=(6,10))
fig = plt.figure(figsize=(6*test_np,10))
fig.subplots_adjust(wspace=0)

test_titles = [r'${\rm Fiducial}$', r'${\rm 90Degree}$', r'${\rm TwoThirds}$']

for i in range(test_np):

        ax1 = plt.subplot2grid((10,6*test_np), (0,6*i), rowspan=5, colspan=6, projection='polar')
        if i == 0:
                ax2 = plt.subplot2grid((10,6*test_np), (5,6*i), rowspan=5, colspan=6)
        else:
                ax2 = plt.subplot2grid((10,6*test_np), (5,6*i), rowspan=5, colspan=6, sharey=ax2_shared)

        Nee = arr_Nee[i]
        Neebar = arr_Neebar[i]
        fee = arr_fee[i,:]
        feebar = arr_feebar[i,:]
        
        th_r, th_max_nu, th_max_bnu, th_eln, fa_nu, fa_bnu, ff_nu, ff_bnu, eln = makepolar(Nee, Neebar, fee, feebar, "eebar")
        
        ax1.plot(th_r, fa_nu, color='blue', label=r"$\nu_e$")
        ax1.scatter(th_max_nu, np.max(fa_nu), color='blue')
        ax1.plot(th_r, fa_bnu, color='red', label=r"$\bar{\nu}_e$")
        ax1.scatter(th_max_bnu, np.max(fa_bnu), color='red')
        
        ax1.arrow(th_max_nu,0,0,Nee*ff_nu, color='blue')
        ax1.arrow(th_max_bnu+np.pi,0,0, Neebar*ff_bnu, color='red')
        ax1.arrow(th_eln+np.pi, 0,0,np.abs(eln[2]), color="purple", linewidth=2)
        
        ax1.get_xaxis().set_ticklabels([])
        ax1.get_yaxis().set_ticklabels([])
        
        lim = np.max([Neebar*ff_bnu,Nee*ff_nu, np.max(fa_nu),np.max(fa_bnu)]) * 1.05
        ax1.set_ylim(0,lim)

        ax1.set_title(test_titles[i])
            
        if i == 0:
                ax1.legend(loc='lower right')
        
        
        ax2.axhline(0,color="k",linestyle="--")
        if i == 0:
                ax2.text(th_eln/np.pi, -0.08, "net ELN direction", color='purple', rotation=-90)
        ax2.plot(th_r/np.pi, (fa_nu)/(Nee+Neebar), color='blue', label=r"$f_{\nu_e}$")
        ax2.plot(th_r/np.pi, (-fa_bnu)/(Nee+Neebar), color='red', label=r"$-f_{\bar{\nu}_e}$")
        ax2.plot(th_r/np.pi, (fa_nu-fa_bnu)/(Nee+Neebar), color='purple', label=r"$f_{\nu_e} - f_{\bar{\nu}_e}$")
        ax2.axvline(th_eln/np.pi, color="purple")
        ax2.set_xlim(0,2)
        ax2.set_xlabel(r"$\theta/\pi$")
        ax2.minorticks_on()
        ax2.yaxis.set_tick_params(which='both', direction='in', right=True,top=True)
        ax2.xaxis.set_tick_params(which='both', direction='in', bottom=True, top=True)

        if i >= 1:
                plt.setp(ax2.get_yticklabels(), visible=False)

        if i <= 1:
                ax2.get_xaxis().set_ticklabels([r'$0.0$', r'$0.5$', r'$1.0$', r'$1.5$'])

        if i == 0:
                ax2.legend(loc='lower right')
                ax2_shared = ax2


plt.savefig("polar_comp_3test_eln.pdf", bbox_inches="tight")

plt.clf()



