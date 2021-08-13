import numpy as np
from scripts.invert_matrix import invert_block_matrix

def test_invert_block_matrix():
    n_block = 3
    M = np.array([\
        [1,0,3,0,2,0],\
        [0,1,0,2,0,5],\
        [3,0,1,0,3,0],\
        [0,2,0,1,0,1],\
        [2,0,3,0,1,0],\
        [0,5,0,1,0,1]])
    invM = invert_block_matrix(M, n_block)
    
    I1 = np.matmul(M, invM)
    I2 = np.matmul(invM, M)
    I0 = np.identity(M.shape[0])
    diff1 = I1 - I0
    diff2 = I2 - I0

    assert(np.allclose(I1, I0)), ('M M^{-1} - I = ', diff1)
    assert(np.allclose(I2, I0)), ('M^{-1} M - I = ', diff2)
