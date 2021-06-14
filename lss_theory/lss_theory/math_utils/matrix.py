import numpy as np

def check_matrix_inverse(a, inv_a, atol=1e-06, feedback_level=0):
    """
    Args:
        a: A square matrix
        inv_a: The inverse of a to be tested
        atol: A float for the absolute tolerance of a*inv_a and 
            inv_a*a against the identity matrix
        feedback_level: integer 0 or 1."""

    id1 = np.matmul(a, inv_a)
    id2 = np.matmul(inv_a, a)
    
    if feedback_level >= 1:
        print('id1 = ', id1)
        print('id2 = ', id2)

    id0 = np.identity(a.shape[0])
    check1 = np.allclose(id1, id0, atol=atol)
    check2 = np.allclose(id2, id0, atol=atol)

    is_inverse_test_passed = (check1 and check2)

    if feedback_level >= 1:
        print('Passed inverse test? - {}'.format(is_inverse_test_passed))
    
    if is_inverse_test_passed == False:
        max_diff1 = np.max(np.abs(id1 - id0))
        max_diff2 = np.max(np.abs(id2 - id0))

        if feedback_level >= 0:
            print('max diff1 = {}, max_diff2 = {}'.format(max_diff1, max_diff2))

    return is_inverse_test_passed
    
def check_matrix_symmetric(a, rtol=1e-05, atol=1e-08):
    """Check if 2d matrix is symmetric given 
    relative and absolute tolerance"""
    
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def split_matrix_into_blocks(a, nrows, ncols):
    """Split a 2d matrix into nblock^2 blocks of shape (nrows, ncols), 
    and return a 3d matrix of shape (nblock*nblock, nrows, ncols)."""

    r, h = a.shape
    return (a.reshape(h//nrows, nrows, -1, ncols)
                 .swapaxes(1, 2)
                 .reshape(-1, nrows, ncols))

def assemble_matrix_from_blocks(a, nblock):
    """Assemble a 3d matrix of shape (nblock*nblock, nrows, ncols)
    into a 2d matrix of shape (nblock*nrows, nblock*ncols) which 
    are formed of the nblock^2 blocks of shape (nrows, ncols).

    For now only support nrows = ncols = block_size.
    
    Example:

    """    
    
    block_size = a.shape[1] #TODO assume now a.shape[1] = a.shape[2]
    assert a.shape[1] == a.shape[2]

    assert nblock == int(np.sqrt(a.shape[0]))

    shape = (nblock*block_size, nblock*block_size)
    b = np.zeros(shape, dtype=a.dtype)

    ib = 0

    for iblock in range(nblock):
        for jblock in range(nblock):
            
            block = a[ib,:,:]
            
            Istart = iblock*block_size
            Iend = (iblock+1)*block_size
            Jstart = jblock*block_size
            Jend = (jblock+1)*block_size

            b[Istart:Iend, Jstart:Jend] = block

            ib = ib + 1

    return b

def reshape_blocks_of_matrix(a, nblock_old, block_size_old):
    """
    Given a matrix a composed of nblock_old of blocks of shape 
    (block_size_old, block_size_old), return a matrix of same 
    shape as a, but composed of nblock_new of blocks of shape
    (block_size_new, block_size_new), where nblock_new = block_size_old
    and block_size_new = nblock_old. 

    Example:

    a = array([[ 0,  1,  2,  3,  4,  5],
       [ 6,  7,  8,  9, 10, 11],
       [12, 13, 14, 15, 16, 17],
       [18, 19, 20, 21, 22, 23],
       [24, 25, 26, 27, 28, 29],
       [30, 31, 32, 33, 34, 35]])
    b = reshape_blocks_of_matrix(a, 3, 2)

    gives
    b = array([[ 0.,  2.,  4.,  1.,  3.,  5.],
       [12., 14., 16., 13., 15., 17.],
       [24., 26., 28., 25., 27., 29.],
       [ 6.,  8., 10.,  7.,  9., 11.],
       [18., 20., 22., 19., 21., 23.],
       [30., 32., 34., 31., 33., 35.]])

    Note: Highly dependent on the return format of split_matrix_into_blocks()!
    """
    nblock_new = block_size_old 
    block_size_new = nblock_old

    b = split_matrix_into_blocks(a, block_size_old, block_size_old)
    c = np.zeros(a.shape)
    for iblock in range(nblock_new):
        for jblock in range(nblock_new):
            
            block_new = b[:, iblock, jblock].reshape(block_size_new, block_size_new)
            
            Istart = iblock*block_size_new
            Iend = (iblock+1)*block_size_new
            Jstart = jblock*block_size_new
            Jend = (jblock+1)*block_size_new
            
            c[Istart:Iend, Jstart:Jend] = block_new

    return c

