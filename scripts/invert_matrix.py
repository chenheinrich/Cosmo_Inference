import numpy as np
import matplotlib.pyplot as plt

from theory.utils.profiler import profiler

def isdiag(M):
    """Returns a boolean for whether M is a diagonal matrix.
    
    Note: it does so by counting the number of nonzero 
      elements after subtracting the diagonal.
    """
    assert M.shape[0]==M.shape[1], ('Expecting square matrix, but M.shape = {}'.format(M.shape))
    count = np.count_nonzero(M - np.diag(np.diagonal(M)))
    return not bool(count)

def invert_diagonal_matrix(M):
    #HACK
    #assert(isdiag(M))
    return np.diag(1./np.diagonal(M))


def invert_block_matrix(M, n_block):
    """
    Returns the inverse of a block matrix where each block is diagonal.
    
    Returns:
        N: 2d numpy array of shape (n_block*block_size, n_block*block_size) 
            as the matrix inverse of input M
    Args:
        M: 2d numpy array of shape (n_block*block_size, n_block*block_size) 
            made of n_block diagonal blocks of shape (block_size, block_size).
            M = [M11 M12 ... M1n
                  M21 ...
                  ...
                  Mn1 Mn2 ... Mnn];
            where each Mij was a diagonal matrix. 
            (Not the same as a “block diagonal matrix”)
        n_block: A integer for the number of blocks on each side of M.
    Reference: https://en.wikipedia.org/wiki/Invertible_matrix#Blockwise_inversion
    Code adapted from Alec: https://www.alecjacobson.com/weblog/?p=3994
    Note: The following algorithm also does not depend on whether Mij=Mji.
    """

    if n_block == 1:
        return invert_diagonal_matrix(M)
    else:

        assert(M.shape[0] == M.shape[1]), ('Matrix must be square')
        assert(M.shape[0]%n_block==0), ('Matrix size must be divisble by k')
        block_size = int(M.shape[0]/n_block)

        # Extract 4 blocks
        start1 = 0
        end1 = (n_block-1)*block_size
        start2 = (n_block-1)*block_size 
        end2 = n_block * block_size

        #print(start1, end1)
        A = M[start1:end1, start1:end1]
        B = M[start1:end1, start2:end2]
        C = M[start2:end2, start1:end1]
        D = M[start2:end2, start2:end2]
        
        #print('A = ', A)
        #print('B = ', B)
        #print('C = ', C)
        #print('D = ', D)
        #HACK
        #assert(isdiag(D)), ('Matrix sub-blocks are not diagonal \
        #  when using n_block = {}'.format(n_block))

        Ainv = invert_A(A, n_block)

        Sinv = make_and_invert_S(Ainv, B, C, D)

        N = form_block(Sinv, Ainv, B, C)

        return N

def invert_A(A, n_block):
    return invert_block_matrix(A, n_block-1)
    
def make_and_invert_S(Ainv, B, C, D):
    S = D - np.matmul(C, np.matmul(Ainv, B))
    return invert_diagonal_matrix(S)

def form_block(Sinv, Ainv, B, C):
    X = np.matmul(Sinv, np.matmul(C, Ainv))
    a = Ainv + np.matmul(Ainv, np.matmul(B, X))
    b = -np.matmul(Ainv, np.matmul(B, Sinv))
    c = -np.matmul(Sinv, np.matmul(C, Ainv))
    d = Sinv
    N = np.block([[a, b], [c, d]])
    return N

def plot_log_matrix_and_inverse(M, n_block, fname='./plots/plot_test_invert_block_matrix.png'):
    invM = invert_block_matrix(M, n_block)
    print('invM = ', invM)
    M_and_invM = np.hstack((M, invM))
    plt.imshow(np.log(M_and_invM))
    plt.savefig(fname)
    print('Saved plot: {}'.format(fname))
    # results: invM looks block diagonal too

@profiler
def timing(M):
  timing = [invert_block_matrix(M, 3) for i in range(10000)]

if __name__ == '__main__':
  M = np.array([\
        [1,0,3,0,2,0],\
        [0,1,0,2,0,5],\
        [3,0,1,0,3,0],\
        [0,2,0,1,0,1],\
        [2,0,3,0,1,0],\
        [0,5,0,1,0,1]])
  plot_log_matrix_and_inverse(M, 3)
