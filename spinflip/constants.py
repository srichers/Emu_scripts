import numpy as np
from matrix import dagger

#fermi coupling constant: G/(c*hbar)^3=1.166 378 7(6)×10−5 GeV−2 --> G=1.166 378 7×10−23 eV^−2 (natural units:c=hbar=1)
G=1.1663787e-23       # eV^-2
c=2.99792458e10     # cm/s
hbar =6.582119569E-16 #eV s
M_p=1.6726219e-24     #grams (Proton mass)

#mixing angles (rad): (values from NuFit)
a12=33.44*np.pi*2/360
a13=8.57 *np.pi*2/360
a23=49.0 *np.pi*2/360

#CP phase:
delta=195*np.pi*2/360
#majorana angles are all 0 and dont influence the matrix

#calculate masses from experimental values
sq_21 = 7.42e-5 #normal ordering!
sq_31 = 2.514e-3 
sum = 0.26

m1_approx = sum/3 
m2_approx = np.sqrt(m1_approx**2 + sq_21)
m3_approx = np.sqrt(m1_approx**2 + sq_31)
sum_approx = m1_approx + m2_approx + m3_approx
m_1 = m1_approx*sum/sum_approx
m_2 = m2_approx*sum/sum_approx
m_3 = m3_approx*sum/sum_approx

## Mass Matrix ##	
m23=np.array([[1,0*1j,0],
              [0,np.cos(a23),np.sin(a23)],
              [0,-np.sin(a23),np.cos(a23)]])

m13=np.array([[np.cos(a13),0,np.sin(a13)*np.exp(-1j*delta)],
              [0,1,0],
              [-np.sin(a13)*np.exp(1j*delta),0,np.cos(a13)]])

m12=np.array([[np.cos(a12),np.sin(a12),0],
              [-np.sin(a12),np.cos(a12),0],
              [0,0*1j,1]])

m=m23 @ m13 @ m12
#m is the mass mixing (MNS) matrix -- m*(1,2,3)=(e,mu,tau)
M_mass_basis=([[m_1,0*1j,0],[0,m_2,0],[0,0,m_3]])
M_3flavor = m @ M_mass_basis @ dagger(m) #(3,3)
M=M_3flavor

m2 = np.array([[np.cos(a12),np.sin(a12)],[-np.sin(a12),np.cos(a12)]])
M_mass_basis_2 =([[m_1,0*1j],[0,m_2]])
M_2flavor = m2 @ M_mass_basis_2 @ dagger(m2)
