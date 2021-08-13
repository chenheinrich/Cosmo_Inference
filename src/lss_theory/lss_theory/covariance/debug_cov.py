import numpy as np
from numpy.lib.function_base import delete
from scipy import linalg

import lss_theory.math_utils.matrix_utils as matrix_utils

do_unique_multitracer = True
do_one_term = False
do_split_signal_noise = True

if do_unique_multitracer == True:
    
    fn_all_terms = './results/bis_mult/covariance/cosmo_planck2018_fiducial/nk_11/lmax_2/do_folded_signal_False/theta_phi_2_4/35x35/cov_all_terms.npy'
    fn_one_term = './results/bis_mult/covariance/cosmo_planck2018_fiducial/nk_11/lmax_2/do_folded_signal_False/theta_phi_2_4/35x35/cov_one_term.npy'
    if do_one_term == True:
        cov = np.load(fn_one_term)
    else:
        cov = np.load(fn_all_terms)

    if do_split_signal_noise == True:
        fn_cov = './results/bis_mult/covariance/cosmo_planck2018_fiducial/nk_11/lmax_2/do_folded_signal_False/theta_phi_2_4/35x35/cov.npy'
        fn_cov_signal = './results/bis_mult/covariance/cosmo_planck2018_fiducial/nk_11/lmax_2/do_folded_signal_False/theta_phi_2_4/35x35/cov_signal.npy'
        fn_cov_noise = './results/bis_mult/covariance/cosmo_planck2018_fiducial/nk_11/lmax_2/do_folded_signal_False/theta_phi_2_4/35x35/cov_noise.npy'

        cov_tot = np.load(fn_cov)
        cov_signal = np.load(fn_cov_signal)
        cov_noise = np.load(fn_cov_noise)
else:
    fn = './results/bis_mult/covariance/cosmo_planck2018_fiducial/nk_11/lmax_2/do_folded_signal_False/theta_phi_2_4/cov.npy'
    cov_in = np.load(fn)

    iz = 0; itri = 0
    cov = cov_in[:, :, iz, itri]


def test_inverse_large_cov():
    invcov = linalg.inv(cov)

    is_inverse_test_passed = matrix_utils.check_matrix_inverse(cov, \
        invcov, atol=1e-3, feedback_level=1)
    print('is_inverse_test_passed = {}'.format(is_inverse_test_passed))

def get_small_cov(iblock_start, iblock_end, jblock_start, jblock_end):
    nlm = 6

    i_start = iblock_start * nlm
    i_end = iblock_end * nlm
    j_start = jblock_start * nlm
    j_end = jblock_end * nlm

    cov_small = cov[i_start:i_end, j_start:j_end]

    return cov_small

def test_inverse_small_cov(iblock_start, iblock_end, jblock_start, jblock_end):
    cov_small = get_small_cov(iblock_start, iblock_end, jblock_start, jblock_end)

    invcov_small = linalg.inv(cov_small)
    is_inverse_test_passed = matrix_utils.check_matrix_inverse(cov_small, \
        invcov_small, atol=1e-3, feedback_level=1)


def plot_small_cov(iblock_start, iblock_end, jblock_start, jblock_end):

    cov_small = get_small_cov(iblock_start, iblock_end, jblock_start, jblock_end)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    im = plt.imshow(abs(cov_small), cmap='jet')

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)

    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    
    cbar.ax.tick_params(labelsize=12)

    pfname = './lss_theory/lss_theory/covariance/debug_cov/small_cov_nblock_%s.png'%nblock
    plt.savefig(pfname)
    print('Saved plot: %s'%pfname) 

#test_inverse_large_cov()

#for nblock in np.arange(12, 14):
#    print('nblock = ', nblock)
#    test_inverse_small_cov(nblock)

#block = 13
#plot_small_cov(nblock)

#test_inverse_small_cov(iblock_start, iblock_end, jblock_start, jblock_end)

#def get_inv_by_blocks(iblock_start, iblock_end, jblock_start, jblock_end):
def get_inv_by_blocks(mat):
    # assuming iblock_start = 0, and iblock_start/end same as jblock_start/end
    #split = int((iblock_start + iblock_end)/2)
    #end = iblock_end 

    #A = get_small_cov(0, split, 0, split)
    #B = get_small_cov(0, split, split, end)
    #C = get_small_cov(split, end, 0, split)
    #D = get_small_cov(split, end, split, end)

    (nrow, ncol) = mat.shape
    assert nrow == ncol
    split = int(np.floor(ncol/2))
    end = ncol

    print('split = {}, end = {}'.format(split, end))

    A = mat[0:split, 0:split]
    B = mat[0:split, split:end]
    C = mat[split:end, 0:split]
    D = mat[split:end, split:end]
    
    Ainv = linalg.inv(A)
    Dinv = linalg.inv(D)

    A2_inv = A - np.matmul(B, np.matmul(Dinv, C))
    A2 = linalg.inv(A2_inv)
    D2_inv = D - np.matmul(C, np.matmul(Ainv, B))
    D2 = linalg.inv(D2_inv)

    B3 = np.matmul(-B, Dinv)
    C3 = np.matmul(-C, Ainv)

    A4 = A2.copy()
    B4 = np.matmul(A2, B3)
    C4 = np.matmul(D2, C3)
    D4 = D2.copy()

    invcov = np.block([
        [A4, B4],
        [C4, D4] 
    ])

    return invcov

def get_conditional_number(cov):
    w, v = linalg.eig(cov)
    #print('w = {}, v = {}'.format(w,v)) # conditional number 1e10
    conditional_number = np.abs(w[0])/np.abs(w[-1])
    print('conditional number = {:e}'.format(conditional_number))
    return conditional_number

if do_unique_multitracer == True:
    
    conditional_number = get_conditional_number(cov)

    end = 210

    #152-156
    for n in range(152,209):
        print('\n n =',n)
        print('Inverse 0, %s, 0, %s'%(n, n))
        #cov_small = get_small_cov(0, n, 0, n)
        cov_small = cov[0:n, 0:n]
        invcov_small = linalg.inv(cov_small)
        is_inverse_test_passed = matrix_utils.check_matrix_inverse(cov_small, \
                invcov_small, atol=1e-3, feedback_level=0)

        print('Inverse %s, %s, %s, %s'%(n, end, n, end))
        #cov_small = get_small_cov(n, end, n, end)
        cov_small = cov[n:end, n:end]
        invcov_small = linalg.inv(cov_small)
        is_inverse_test_passed = matrix_utils.check_matrix_inverse(cov_small, \
                invcov_small, atol=1e-3, feedback_level=0)

    #print('cov', cov)
    #print('cov[10:13, 0:5]', cov[10:13, 0:5])

    check_block_inversion = False
    if check_block_inversion == True:
        rescale = 1.0
        ntot = 210
        
        #split = 32
        #A = get_small_cov(0, split, 0, split)/rescale
        #B = get_small_cov(0, split, split, ntot)/rescale
        #C = get_small_cov(split, ntot, 0, ntot)/rescale
        #D = get_small_cov(split, ntot, split, ntot)/rescale

        #print('max of A: ', np.max(np.abs(A)))
        #print('max of B: ', np.max(np.abs(B)))
        #print('max of C: ', np.max(np.abs(C)))
        #print('max of D: ', np.max(np.abs(D)))

        #Ainv = linalg.inv(A)
        #detA = linalg.det(A)
        #S = D - np.matmul(C, np.matmul(Ainv, B))
        #detS = linalg.det(S)

        #print('detA = {}, detS = {}, detA*detS = {}'.format(detA, detS, detA * detS))

        #cov_small = get_small_cov(0, 24, 0, 24)/rescale
        #print('det_cov = {}', linalg.det(cov_small))

        #start = 70
        #end = 210
        start = 0
        end = 60
        from lss_theory.math_utils.matrix_utils import delete_zero_cols_and_rows

        if do_one_term == True:
            cov_small, rows_null = delete_zero_cols_and_rows(cov, rows_null = range(62,66))
        else:
            #cov_small, rows_null = delete_zero_cols_and_rows(cov, rows_null = range(150, 210))
            cov_small, rows_null = delete_zero_cols_and_rows(cov, rows_null = range(152,156))
        # i = 60-72 coresponds to ib = 10, 11, ilm = 0-6
        # ib = 10, 11 corresponds to isample triplets (0, 2, 3) and (0, 2, 4)
        # range(62-66) works too
        # Don't know why it would make such a difference!! ??
        #cov_small = cov[start:end, start:end]
        # 
        
        invcov_small1 = get_inv_by_blocks(cov_small)
        invcov_small2 = linalg.inv(cov_small)

        print('inverse by block')
        is_inverse_test_passed = matrix_utils.check_matrix_inverse(cov_small, \
            invcov_small1, atol=1e-3, feedback_level=0)

        print('direct inverse')
        is_inverse_test_passed = matrix_utils.check_matrix_inverse(cov_small, \
            invcov_small2, atol=1e-3, feedback_level=0)

        print('cov_small.shape', cov_small.shape)
        print('rows_null', rows_null)

        conditional_number = get_conditional_number(cov_small)

    nb = 35
    nlm = 6
    i = 0
    for ib in range(nb):
        for ilm in range(nlm):
            #print('i = {}, ib = {}, ilm = {}'.format(i, ib, ilm))
            i = i+1

else:
    rescale = 1
    A = get_small_cov(0, 12, 0, 12)/rescale
    B = get_small_cov(0, 12, 12, 24)/rescale
    C = get_small_cov(12, 24, 0, 12)/rescale
    D = get_small_cov(12, 24, 12, 24)/rescale

    print('max of A: ', np.max(np.abs(A)))
    print('max of B: ', np.max(np.abs(B)))
    print('max of C: ', np.max(np.abs(C)))
    print('max of D: ', np.max(np.abs(D)))

    print('B = C transpose? --', np.allclose(B, np.transpose(C)))
    Ainv = linalg.inv(A)
    detA = linalg.det(A)
    S = D - np.matmul(C, np.matmul(Ainv, B))
    detS = linalg.det(S)

    print('detA = {}, detS = {}, detA*detS = {}'.format(detA, detS, detA * detS))

    cov_small = get_small_cov(0, 24, 0, 24)/rescale
    print('det_cov = {}', linalg.det(cov_small))

    print('cov is cov tranpsose?', np.allclose(cov_small/1e10, np.conj(np.transpose(cov_small/1e10))))
    import pdb
    #invcov_small1 = get_inv_by_blocks(cov_small)
    invcov_small2 = linalg.inv(cov_small)

    #w, v = linalg.eig(A)
    #print('eigenvalues = ', w)
    #w, v = linalg.eig(B)
    #print('eigenvalues = ', w)
    #w, v = linalg.eig(C)
    #print('eigenvalues = ', w)
    #w, v = linalg.eig(D)
    #print('eigenvalues = ', w)

    # Smallest eigenvalues drop by 9 orders of magnitude going from 
    # so matrix is either ill-conditioned, or it has too large of a range
    # between the eigenvalues. This can possibly be mitigated if it
    # is the later with some tricks (need to do some more research in this.)

    #cov_small = np.diag(np.arange(4)+1)
    #print(cov_small)
    #invcov_small = get_inv_by_blocks(cov_small)
    #print(invcov_small)

    #is_inverse_test_passed = matrix_utils.check_matrix_inverse(cov_small, \
    #        invcov_small1, atol=1e-3, feedback_level=1)
    #is_inverse_test_passed = matrix_utils.check_matrix_inverse(cov_small, \
    #        invcov_small2, atol=1e-3, feedback_level=0)
