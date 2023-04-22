import numpy as np

# set of orthonormal vectors, where n_vector is pointed along theta,phi
class Basis:
    def __init__(self, theta, phi): #theta is polar, phi is azimuthal
        self.phi=phi
        self.n_vector=np.array([1,np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)])
        self.x1=np.array([0,np.cos(phi)*np.cos(theta),np.sin(phi)*np.cos(theta),(-1)*np.sin(theta)])
        self.x2=np.array([0,-np.sin(phi),np.cos(phi),0])
	
    #potential projected onto the basis
    def dot(self,potential,vector):
        projection=-vector[0]*potential[0]
        for k in range(1,4):
            projection=projection+vector[k]*potential[k]
        return -projection

    def plus(self, potential): #(3,3,nz)
        vector=0.5*(self.x1+1j*self.x2)
        plus=np.exp(1j*self.phi)*self.dot(potential,vector)
        return plus

    def minus(self, potential): #(3,3,nz)
        vector=0.5*(self.x1-1j*self.x2)
        minus=np.exp(-1j*self.phi)*self.dot(potential,vector)
        return minus

    def kappa(self,potential):
        return self.dot(potential,self.n_vector)
	
