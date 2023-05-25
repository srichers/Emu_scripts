import numpy as np
from matrix import dagger

#fermi coupling constant: G/(c*hbar)^3=1.166 378 7(6)×10−5 GeV−2 --> G=1.166 378 7×10−23 eV^−2 (natural units:c=hbar=1)
G=1.1663787e-23       # eV^-2
c=2.99792458e10     # cm/s
hbar =6.582119569E-16 #eV s
M_p=1.6726219e-24     #grams (Proton mass)

#mixing angles (rad): (values from NuFit)
a12=33.4*np.pi*2/360
a13=49.2*np.pi*2/360
a23=8.57*np.pi*2/360

#CP phase:
delta=194*np.pi*2/360
#majorana angles are all 0 and dont influence the matrix

#masses (eV) (Negative mass? 0 mass?)
m_1=0.608596511
m_2=0.608
m_3=0.649487372

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
