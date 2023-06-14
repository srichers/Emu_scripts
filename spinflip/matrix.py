import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

#unitary trace matrix
def trace_matrix(data):#takes in (nF,nF,nz)
        
        nF = data.shape[0]
        # calculate trace
        diagonals = np.array([data[n,n,:] for n in range(nF)])
        trace = np.sum(diagonals, axis=0) # length of nz
        
        # create an identity matrix multiplied by the trace
        matrix=np.zeros_like(data)
        for n in range(nF):
                matrix[n,n,:]=trace
        
        return matrix
	
#conjugate a matrix
def dagger(matrix):
	conjugate=np.transpose(np.conjugate(matrix))
	return conjugate

def rm_trace(M):
    return np.array(M) - np.trace(M)*np.identity(np.array(M).shape[0])/np.array(M).shape[0]

def visualizer(M, log=True,  text='mag', traceless = True, vmin=None,vmax=None, savefig=False):
    if traceless ==True:
        M=rm_trace(M)
    else:
        M=np.array(M)
    
    matrix_size = M.shape[0]
    grid_subdivisions = 1/matrix_size
    vertices_x = [n*grid_subdivisions for n in np.arange(0,matrix_size+1)]
    vertices_y = [n*grid_subdivisions for n in np.arange(matrix_size,-1,-1)]

    if vmin == None:
        min_value=float(np.min(np.abs(M)))+1E-15
    else:
        min_value = vmin
    if vmax == None:
        max_value=np.max(np.abs(M))
    else: 
        max_value = vmax

    if log ==True: scale = norm=mpl.colors.LogNorm(vmin=min_value, vmax=max_value)
    else: scale = None
    
    
    f, ax = plt.subplots(figsize = (8,6))
    ax.set_ylim(0,1)
    ax.set_xlim(0,1)
    
    image = ax.pcolormesh(vertices_x,vertices_y,np.abs(M),cmap='hot',norm=scale)
    plt.colorbar(image)
    
    if text=='mag':
        for n in np.arange(0,matrix_size):
            for m in np.arange(0,matrix_size):
                xcoord = (n+1/6)*grid_subdivisions
                ycoord = 1 - (m+1/2)*grid_subdivisions
                if log == True:
                    ax.text(xcoord, ycoord, str(round(np.log10(np.abs(M[m,n])),0)), color='cyan', size=10)
                else:
                    ax.text(xcoord, ycoord, str(round(np.abs(M[m,n]),2)), color='cyan', size=10)
                
    elif text=='arg':
        for n in np.arange(0,matrix_size):
            for m in np.arange(0,matrix_size):
                xcoord = (n)*grid_subdivisions
                ycoord = 1 - (m+1/2)*grid_subdivisions
                ax.text(xcoord, ycoord, str(round(np.real(M[m,n]/np.abs(M[m,n])), ndigits=1))+'+'+str(round(np.imag(M[m,n])/np.abs(M[m,n]), ndigits=2))+'i', color='cyan', size=9)
    elif text == 'full':
        for n in np.arange(0,matrix_size):
            for m in np.arange(0,matrix_size):
                xcoord = (n)*grid_subdivisions
                ycoord = 1 - (m+1/2)*grid_subdivisions
                ax.text(xcoord, ycoord, str(round(np.real(M[m,n]), ndigits=2))+'+'+str(round(np.imag(M[m,n]), ndigits=2))+'i', color='cyan', size=9)
    f.show()
    if savefig == True: 
        plt.tight_layout()
        plt.savefig('visualizer.png', dpi=300)

def derivative(H,P, n=1):
    from constants import hbar
    for k in np.arange(n):
        P = -1.j/hbar*( H @ P - P @ H)
    return P