from operator import is_
import numpy as np
import os
import scipy.linalg as linalg

class RearrangeCov():

    def __init__(self, fn_cov):
        self.fn_cov = fn_cov
        self.cov = np.load(self.fn_cov)

    def rearrange_cov_full_to_diag_in_nb(self):
        """
        Given self.cov of shape (nbxnori, nbxnori, nz, ntri), 
        we return a covariance matrix of shape (nori, nori, nb, nz, ntri)
        throwing away all off-diagonal correlations between different 
        multi-tracer bispectrum indexed by ib. 
        """
        (nbxnori, nbxnori, nz, ntri) = self.cov.shape

        #HACK for now
        nori = 8
        nb = int(nbxnori/8)

        cov_new = np.zeros((nori, nori, nb, nz, ntri))

        for itri in range(ntri):
            for iz in range(nz):

                cov = self.cov[:, :, iz, itri]
                
                for iori in range(nori):
                    for jori in range(nori):
                        
                        start1 = iori * nb
                        end1 = (iori+1) * nb
                        start2 = jori * nb
                        end2 = (jori+1) * nb

                        # cov is nori^2 blocks of nbxnb matrices
                        # take the iori-th and jori-th nbxnb block
                        block_iori_jori = cov[start1:end1, start2:end2]
                        diag_in_b = np.diag(block_iori_jori)
                        
                        for ib in range(nb):
                            
                            cov_new[iori, jori, ib, iz, itri] = diag_in_b[ib]

        return cov_new

    def get_invcov_diag_in_nb(self, cov):

        if len(cov.shape) == 5:

            (nori, nori, nb, nz, ntri) = cov.shape
            invcov = np.zeros_like(cov)

            for itri in range(ntri):
                for iz in range(nz):
                    
                    for ib in range(nb):
                        print('iz = {}, itri = {}, ib = {}'.format(iz, itri, ib))
                        
                        cov_tmp = cov[:, :, ib, iz, itri]
                        print('cov_tmp = ', cov_tmp)

                        invcov_tmp = linalg.inv(cov_tmp)
                        print('invcov_tmp = ', invcov_tmp)
                        
                        id1 = np.matmul(invcov_tmp, cov_tmp)
                        id2 = np.matmul(cov_tmp, invcov_tmp)
                        
                        print('id1 = ', id1)
                        print('id2 = ', id2)
                        
                        invcov[:, :, ib, iz, itri] = invcov_tmp

        return invcov

    def rearrange_cov_full_to_diag_in_nori(self):
        """
        Given self.cov of shape (nbxnori, nbxnori, nz, ntri), 
        we return a covariance matrix of shape (nori, nori, nb, nz, ntri)
        throwing away all off-diagonal correlations between different 
        multi-tracer bispectrum indexed by ib. 
        """

        (nbxnori, nbxnori, nz, ntri) = self.cov.shape

        #HACK for now
        nori = 8
        nb = int(nbxnori/8)

        cov_new = np.zeros((nb, nb, nz, ntri, nori))

        for itri in range(ntri):
            for iz in range(nz):

                print('itri = {}, iz = {}'.format(itri, iz))

                cov = self.cov[:, :, iz, itri]
                for iori in range(nori):
                    cov_new[:, :, iz, itri, iori] = cov[iori*nb:(iori+1)*nb, iori*nb:(iori+1)*nb]

        return cov_new

    def get_invcov_diag_in_nori(self, cov, atol=1e-5):
        """Returns inverse covariance of shape (nb, nb, nz, ntri, nori)"""

        if len(cov.shape) == 5:

            (nb, nb, nz, ntri, nori) = cov.shape
            invcov = np.zeros_like(cov)
            
            for itri in range(ntri):
                for iz in range(nz):
                    
                    for iori in range(nori):

                        print('iz = {}, itri = {}, iori = {}'.format(iz, itri, iori))
                        
                        cov_tmp = cov[:, :, iz, itri, iori]
                        #print('cov_tmp = ', cov_tmp)

                        invcov_tmp = linalg.inv(cov_tmp)
                        #print('invcov_tmp = ', invcov_tmp)
                        
                        id1 = np.matmul(invcov_tmp, cov_tmp)
                        id2 = np.matmul(cov_tmp, invcov_tmp)
                        
                        #print('id1 = ', id1)
                        #print('id2 = ', id2)

                        id0 = np.identity(nb)
                        check1 = np.allclose(id1, id0, atol=atol)
                        check2 = np.allclose(id2, id0, atol=atol)
                        is_inverse_test_passed = (check1 and check2)
                        #print('Passed inverse test? - {}'.format(is_inverse_test_passed))
                        if is_inverse_test_passed == False:
                            max_diff1 = np.max(np.abs(id1 - id0))
                            max_diff2 = np.max(np.abs(id2 - id0))
                            print('max diff1 = {}, max_diff2 = {}'.format(max_diff1, max_diff2))
                        
                        invcov[:, :, iz, itri, iori] = invcov_tmp

        return invcov



if __name__ == '__main__':
    """Sample Usage: python3 -m galaxy_3d.theory.covariance.rearrange_cov"""

    cov_dir = '/Users/chenhe/Research/My_Projects/SPHEREx/SPHEREx_forecasts/git/SphereLikes/plots/theory/covariance/b3d_rsd_theta1_phi12_2_4/fnl_0/nk_11/test20210526/'
    fn_cov = os.path.join(cov_dir, 'cov_full.npy')
    rearr = RearrangeCov(fn_cov)
    
    # This doesn't work -- gives nearly degenerate (ill-conditioned) matrices
    # and they cannot be inverted accurately.
    #cov_new = rearr.rearrange_cov_full_to_diag_in_nb()
    #fn_cov_new = os.path.join(cov_dir, 'cov_diag_in_multitracer.npy')
    #np.save(fn_cov_new, cov_new)
    
    #invcov_new = rearr.get_invcov_diag_in_nb(cov_new)
    #fn_invcov = os.path.join(cov_dir, 'invcov_diag_in_multitracer.npy')
    #np.save(fn_invcov, invcov_new)

    cov_new = rearr.rearrange_cov_full_to_diag_in_nori()
    fn_cov_new = os.path.join(cov_dir, 'cov_diag_in_orientation.npy')
    np.save(fn_cov_new, cov_new)
    print('Saved rearranged covariance at {}'.format(fn_cov_new))
    
    atol = 1e-6 
    invcov_new = rearr.get_invcov_diag_in_nori(cov_new, atol=atol)
    fn_invcov = os.path.join(cov_dir, 'invcov_diag_in_orientation.npy')
    np.save(fn_invcov, invcov_new)
    print('Saved inverse of rearranged covariance at {}'.format(fn_invcov))
    
