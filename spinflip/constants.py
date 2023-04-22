import numpy as np

#fermi coupling constant: G/(c*hbar)^3=1.166 378 7(6)×10−5 GeV−2 --> G=1.166 378 7×10−23 eV^−2 (natural units:c=hbar=1)
G=1.1663787*10**(-23) # eV^-2
c=299792458         # cm/s
hbar =6.582119569E-16 #eV s
M_p=1.6726219*10**(-24)#grams (Proton mass)

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

#test neutrino momentum:
p_abs=10**7#eV
