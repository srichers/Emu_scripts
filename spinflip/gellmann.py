import numpy as np
from itertools import product

#### Functions ####
###################
#Gell-Mann matrices (for scalarfunc)
GM=np.array([[[0, 1, 0],
              [1, 0, 0],
              [0, 0, 0]],
             
             [[ 0, -1j, 0],
              [1j,  0 , 0],
              [ 0,  0 , 0]],
             
             [[1,  0, 0],
              [0, -1, 0],
              [0,  0, 0]],
             
             [[0, 0, 1],
              [0, 0, 0],
              [1, 0, 0]],
             
             [[0 , 0, -1j],
              [0 , 0,  0 ],
              [1j, 0,  0 ]],
             
             [[0, 0, 0],
              [0, 0, 1],
              [0, 1, 0]],
             
             [[0, 0 ,  0 ],
              [0, 0 , -1j],
              [0, 1j,  0 ]],
             
             [[3**(-1/2), 0        , 0           ],
              [0        , 3**(-1/2), 0           ],
              [0        , 0        , -2*3**(-1/2)]]
])

#scalarfunc: averages square magnitude of components for every location in nz and returns a list of these
# input: array of shape (3,3,nz)
# ouptut: array of length nz
def scalarfunc(array):
    nz = np.shape(array)[2]

    # get the coefficients of each Gell-Mann matrix
    # GM matrices are normalized so that Tr(G_a G_b) = 2 delta_ab
    # result has shape (nz,8)
    components = 0.5 * np.array([[
            np.trace( np.matmul( GM[k], array[:,:,n] ) )
            for k in range(8)] for n in range(nz)] )

    # Construct a scalar out of the GM coefficients as sqrt(G.G) where G is the vector of GM matrix coefficients
    # even though for a Hermitian matrix coefficients must be real, the input could be multiplied by an overall phase
    # Return a real number by using a*conj(a)
    # result has length nz
    scalars = np.sqrt( np.sum( components*np.conj(components), axis=1) )

    return np.real(scalars)

#like scalarfunc, but takes in time dependent arrays (nt, nF, nF, nz) and applies scalarfunc at every time point
def scalarfunc_time(array):
    return np.array([scalarfunc(array[t]) for t in range(0,np.shape(array)[0])])

#like scalarfunc but just squares all the entries in the array and adds them. Ideal for finding mag of HLR
def scalarfunc_sum(M):
    size=M.shape[0]
    nz=M.shape[2]
    return np.array([np.linalg.norm(M[:,:,n].reshape(size**2)) for n in np.arange(0,nz)])

#same difference as scalarfunc_time.
def scalarfunc_sum_time(array):
    return np.array([scalarfunc_sum(array[t]) for t in range(0,np.shape(array)[0])])

def gellmann(j, k, d):
    r"""Returns a generalized Gell-Mann matrix of dimension d. According to the
    convention in *Bloch Vectors for Qubits* by Bertlmann and Krammer (2008),
    returns :math:`\Lambda^j` for :math:`1\leq j=k\leq d-1`,
    :math:`\Lambda^{kj}_s` for :math:`1\leq k<j\leq d`,
    :math:`\Lambda^{jk}_a` for :math:`1\leq j<k\leq d`, and
    :math:`I` for :math:`j=k=d`.

    :param j: First index for generalized Gell-Mann matrix
    :type j:  positive integer
    :param k: Second index for generalized Gell-Mann matrix
    :type k:  positive integer
    :param d: Dimension of the generalized Gell-Mann matrix
    :type d:  positive integer
    :returns: A genereralized Gell-Mann matrix.
    :rtype:   numpy.array

    """

    if j > k:
        gjkd = np.zeros((d, d), dtype=np.complex128)
        gjkd[j - 1][k - 1] = 1
        gjkd[k - 1][j - 1] = 1
    elif k > j:
        gjkd = np.zeros((d, d), dtype=np.complex128)
        gjkd[j - 1][k - 1] = -1.j
        gjkd[k - 1][j - 1] = 1.j
    elif j == k and j < d:
        gjkd = np.sqrt(2/(j*(j + 1)))*np.diag([1 + 0.j if n <= j
                                               else (-j + 0.j if n == (j + 1)
                                                     else 0 + 0.j)
                                               for n in range(1, d + 1)])
    else:
        gjkd = np.diag([1 + 0.j for n in range(1, d + 1)])

    return gjkd


def get_GM(d):
    r'''Return a basis of orthogonal Hermitian operators on a Hilbert space of
    dimension d, with the identity element in the last place.

    '''
    return np.array([gellmann(j, k, d) for j, k in product(range(1, d + 1), repeat=2)])

# finds Gell Mann projection vector of a matrix
def projection(M):
    matrix_size = np.shape(M)[0]
    GM_components = get_GM(matrix_size)
    return np.array([np.trace(0.5*GM_components[n] @ M) for n in np.arange(0, matrix_size**2-1)])

#scalar magnitude of GM projection
def magnitude(M):
    return np.linalg.norm(projection(M))

#magnitude of array by just squaring and adding all components. this is equivalent to the above if you're looking at an off-diagonal subset of an array (eg HLR).
def sum_magnitude(M):
    size=M.shape[0]
    M_flattened = M.reshape(size**2)
    return np.linalg.norm(M_flattened)

#returns H dot P as Gell Mann projection vectors. Resonance occurs when this is 0 
def dotprod(H, P):
    if np.shape(H)[0] != np.shape(P)[0]:
        raise TypeError('H and P have different dimensions')
        return
    else: 
        return np.dot(np.real(projection(H)),np.real(projection(P)))

#GM gets the 8 GM components of the flux matrix
def components(d):
    flux=J(d)
    return np.array([[[[np.trace( np.matmul( GM[k], flux[t,x,:,:,n] ) ) 
                        for k in range (0,8)] for n in range (0,np.shape(flux)[-1])]
                        for x in range (0,4)] for t in range (0,np.shape(flux)[0])])
