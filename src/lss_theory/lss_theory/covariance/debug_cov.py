import numpy as np
from scipy import linalg
import os

import lss_theory.math_utils.matrix_utils as matrix_utils

do_unique_multitracer = True
do_one_term = False
do_split_signal_noise = True
do_five_samples = True

iz = 0

def load_cov_tot_and_cov_ori(subdir):

    output_dir_root = './results/bis_mult/covariance/cosmo_planck2018_fiducial/nk_11/lmax_2/do_folded_signal_False/theta_phi_2_4/'
    output_dir = os.path.join(output_dir_root, subdir)

    fn_cov = os.path.join(output_dir, 'cov.npy')
    fn_cov_signal = os.path.join(output_dir, 'cov_signal.npy')
    fn_cov_noise = os.path.join(output_dir, 'cov_noise.npy')
    fn_cov_ori = os.path.join(output_dir, 'cov_ori.npy')
            
    #fn_cov = './results/bis_mult/covariance/cosmo_planck2018_fiducial/nk_11/lmax_2/do_folded_signal_False/theta_phi_2_4/35x35/cov.npy'
    #fn_cov_signal = './results/bis_mult/covariance/cosmo_planck2018_fiducial/nk_11/lmax_2/do_folded_signal_False/theta_phi_2_4/35x35/cov_signal.npy'
    #fn_cov_noise = './results/bis_mult/covariance/cosmo_planck2018_fiducial/nk_11/lmax_2/do_folded_signal_False/theta_phi_2_4/35x35/cov_noise.npy'
    #fn_cov_ori = '/Users/chenhe/Research/My_Projects/SPHEREx/SPHEREx_forecasts/git/SphereLikes/results/bis_mult/covariance/cosmo_planck2018_fiducial/nk_11/lmax_2/do_folded_signal_False/theta_phi_2_4/35x35/cov_ori.npy'
    
    cov_tot = np.load(fn_cov)
    cov_ori = np.load(fn_cov_ori)
    
    cov_signal = np.load(fn_cov_signal)
    cov_noise_in = np.load(fn_cov_noise)

    print('Loaded cov_tot from fn_cov = {}'.format(fn_cov))

    return cov_tot, cov_ori, cov_signal, cov_noise_in

if do_unique_multitracer == True:
    
    fn_all_terms = './results/bis_mult/covariance/cosmo_planck2018_fiducial/nk_11/lmax_2/do_folded_signal_False/theta_phi_2_4/35x35/cov_all_terms.npy'
    fn_one_term = './results/bis_mult/covariance/cosmo_planck2018_fiducial/nk_11/lmax_2/do_folded_signal_False/theta_phi_2_4/35x35/cov_one_term.npy'
    if do_one_term == True:
        cov = np.load(fn_one_term)
    else:
        cov = np.load(fn_all_terms)

    if do_split_signal_noise == True:
        itri = 0
        if do_five_samples == True:
            subdir = 'five_samples/with_triangle_cases_debug_iz_%s_itri_%s/'%(iz, itri)
        else:
            subdir = '35x35_with_triangle_cases_iz_%s_itri_%s/'%(iz, itri)
        cov_tot, cov_ori, cov_signal, cov_noise_in = load_cov_tot_and_cov_ori(subdir)

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

def get_det(cov):
    det = np.linalg.det(cov)
    print('determinant = {:e}'.format(det))
    return det

def get_matrix_rank(cov):
    rank = np.linalg.matrix_rank(cov)
    print('matrix rank = {}'.format(int(rank)))
    return int(rank)

def get_inverse_of_sum_A_and_B(A, B):
    invA = linalg.inv(A)
    C = np.matmul(B, invA)
    trace = np.trace(C)
    inv_sum = invA - np.matmul(invA, C)/(1.+trace)
    return inv_sum

def get_mean(cov):
    mean = np.mean(cov)
    print('Mean =  {:e}'.format(mean))
    return mean

from lss_theory.math_utils.matrix_utils import delete_zero_cols_and_rows

if do_split_signal_noise == True:

    if do_five_samples == True:

        itris = [0]
        itris_equilateral = [0, 12, 23, 33, 42, 50, 57, 63, 68, 72, 75]

        is_inverse_test_passed_array = np.zeros(len(itris))

        for itri in itris:
            subdir = 'five_samples_lmax_1_bias_sample4_2p0/with_triangle_cases_debug_iz_%s_itri_%s/'%(iz, itri)
            cov_tot, cov_ori, cov_signal, cov_noise = load_cov_tot_and_cov_ori(subdir)

            #cov_tot = cov_tot[0:72,0:72]
            #print(cov_tot.shape)
            print('cov_tot.shape', cov_tot.shape)

            print('itri = {}'.format(itri))
            if itri in itris_equilateral:
                print('   is equilateral')
            get_matrix_rank(cov_tot)
            get_matrix_rank(cov_signal)
            get_matrix_rank(cov_noise)

            invcov_tot = linalg.inv(cov_tot)
            is_inverse_test_passed = matrix_utils.check_matrix_inverse(cov_tot, \
                invcov_tot, atol=2e-3, feedback_level=0)


            #assert is_inverse_test_passed
            #is_inverse_test_passed_array[itri] = is_inverse_test_passed
            #print('\n')

            #cov_small, rows_null = delete_zero_cols_and_rows(cov_tot, rows_null = range(14,20))
            #cov_small, rows_null = delete_zero_cols_and_rows(cov_tot, rows_null = range(12,15))
            #cov_small, rows_null = delete_zero_cols_and_rows(cov_small, rows_null = range(18,21))
            #cov_small, rows_null = delete_zero_cols_and_rows(cov_tot, rows_null = range(12,18))
            #cov_small, rows_null = delete_zero_cols_and_rows(cov_tot, rows_null = range(18,24))
            #cov_small, rows_null = delete_zero_cols_and_rows(cov_tot, rows_null = range(24,30))
            #print('\n number of rows removed:', len(rows_null))

            is_row_degenerate = np.zeros(105)
            is_inverse_test_passed_cov_small_array = np.zeros(105)
            for i in range(105):
                print('\nRemoving row i = ', i)
                cov_small, rows_null = delete_zero_cols_and_rows(cov_tot, rows_null = [i])
                print('number of rows removed:', len(rows_null))
            
                rank = get_matrix_rank(cov_small)
                invcov_small = linalg.inv(cov_small)
                is_inverse_test_passed_cov_small = matrix_utils.check_matrix_inverse(cov_small, \
                    invcov_small, atol=1e-3, feedback_level=0)
                print('is_inverse_test_passed_cov_small: ', is_inverse_test_passed_cov_small)
                
                if is_inverse_test_passed_cov_small == True:
                    is_inverse_test_passed_cov_small_array[i] = 1

                if rank == 104:
                    is_row_degenerate[i] = 1
                
            print('is_row_degenerate', is_row_degenerate)
            print('   which rows = ', np.where(is_row_degenerate == 1))
            ind_deg = np.where(is_row_degenerate == 1)

            print('is_inverse_test_passed_cov_small_array', is_inverse_test_passed_cov_small_array)
            print('   which rows = ', np.where(is_inverse_test_passed_cov_small_array == 1))


        cov_degenerate =cov_tot[ind_deg[0],:][:,ind_deg[0]]
        
        for i in range(9):
            print('\nRemoving row i = ', i)
            cov_small, rows_null = delete_zero_cols_and_rows(cov_degenerate, rows_null = [i])
            print('number of rows removed:', len(rows_null))
        
            rank = get_matrix_rank(cov_small)
            invcov_small = linalg.inv(cov_small)
            is_inverse_test_passed_cov_small = matrix_utils.check_matrix_inverse(cov_small, \
                invcov_small, atol=1e-3, feedback_level=0)
            print('is_inverse_test_passed_cov_small: ', is_inverse_test_passed_cov_small)
        
        print('cov_degenerate[0,:]')
        print(cov_degenerate[0,:])
        print('cov_degenerate[1,:]/cov_degenerate[1,0]*cov_degenerate[0,0]')
        print(cov_degenerate[1,:]/cov_degenerate[1,0]*cov_degenerate[0,0])

        print('cov_degenerate = ', cov_degenerate)

        invcov_degenerate = linalg.inv(cov_degenerate)
        is_inverse_test_passed_cov_degenerate = matrix_utils.check_matrix_inverse(cov_degenerate, \
            invcov_degenerate, atol=1e-3, feedback_level=0)
        print('is_inverse_test_passed_cov_degenerate: ', is_inverse_test_passed_cov_degenerate)

        import pdb; 
        pdb.set_trace()

        #from scipy.sparse.linalg import cg

        ##print('is_inverse_test_passed_array = {}'.format(is_inverse_test_passed_array))
        ##ind = np.where(is_inverse_test_passed_array == 0)
        ##print('ind = ', ind)

    else:
        
        itris = [0, 12, 23, 33, 42, 50, 57, 63, 68, 72, 75]
        is_inverse_test_passed_array = np.zeros(len(itris))

        iz = 0
        for itri in itris:
            subdir = '35x35_with_triangle_cases_iz_%s_itri_%s/'%(iz, itri)
            cov_tot, cov_ori, cov_signal, cov_noise = load_cov_tot_and_cov_ori(subdir)

            #get_conditional_number(cov_ori)
            #get_det(cov_ori)
            #get_matrix_rank(cov_ori)

            get_mean(cov_tot)
            get_mean(cov_signal)
            get_mean(cov_noise)
            
            rank = get_matrix_rank(cov_tot)
            rank = get_matrix_rank(cov_signal)
            rank = get_matrix_rank(cov_noise)

            do_test = False
            if do_test:
                end = 210
                start = 100
                for n in np.arange(18,20):
                    cov_small = cov_tot[n:end, n:end]
                    #cov_small = cov_tot[n:end, n:end]
                    print('For cov last block start at {}, {}'.format(n, n))
                    rank = get_matrix_rank(cov_small)
                    #assert n == rank, (n, rank)
                    assert n == end-rank, (n, end-rank)

            #sys.exit()
            #cov_small, rows_null = delete_zero_cols_and_rows(cov_tot, rows_null = range(120,195))
            #cov_small, rows_null = delete_zero_cols_and_rows(cov_noise, rows_null = range(125,195)) #156
            
            cov_small, rows_null = delete_zero_cols_and_rows(cov_tot, rows_null = range(14,20))
            print('number of rows removed:', len(rows_null))

            #cov_small = cov_tot
            rank = get_matrix_rank(cov_small)

            invcov_small = linalg.inv(cov_small)
            is_inverse_test_passed = matrix_utils.check_matrix_inverse(cov_small, \
                invcov_small, atol=1e-3, feedback_level=0)

            #is_inverse_test_passed_array[itri] = is_inverse_test_passed

        print('is_inverse_test_passed_array = {}'.format(is_inverse_test_passed_array))
        ind = np.where(is_inverse_test_passed_array == 0)
        print('ind = ', ind)

        sys.exit()

        rescale = 1.0

        get_conditional_number(cov_tot/rescale)
        get_conditional_number(cov_signal/rescale)
        get_conditional_number(cov_noise/rescale)

        get_det(cov_tot/rescale)
        get_det(cov_signal/rescale)
        get_det(cov_noise/rescale)

        get_mean(cov_tot)
        get_mean(cov_signal)
        get_mean(cov_noise)

        get_matrix_rank(cov_tot)
        get_matrix_rank(cov_signal)
        get_matrix_rank(cov_noise)

        print('cov_tot.shape', cov_tot.shape)

        invcov_tot = linalg.inv(cov_tot/rescale)
        is_inverse_test_passed = matrix_utils.check_matrix_inverse(cov_tot/rescale, \
            invcov_tot, atol=1e-3, feedback_level=0)

        A = cov_noise/rescale
        B = cov_signal/rescale
        invcov_tot2 = get_inverse_of_sum_A_and_B(A, B)
        is_inverse_test_passed = matrix_utils.check_matrix_inverse(cov_tot/rescale, \
            invcov_tot2, atol=1e-3, feedback_level=0)

        cov_tot2 = cov_noise + cov_signal
        frac_diff = np.abs((cov_tot2 - cov_tot)/cov_tot)
        max_frac_diff = np.nanmax(frac_diff)
        print('max_frac_diff = {}'.format(max_frac_diff))
        ind = np.where(frac_diff == max_frac_diff)
        print(cov_tot[ind], cov_tot2[ind])

else:
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