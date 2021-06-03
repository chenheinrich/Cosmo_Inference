from operator import is_
import numpy as np
import argparse
import yaml
import os

from lss_theory.covariance import Bispectrum3DRSDCovarianceCalculator

class LikelihoodTest():

    def __init__(self, cov_calculator, fn0, fn1):
        self.cov_calculator = cov_calculator
        self.delta = self.load_delta_data_vector(fn0, fn1)
        (self.nb, self.nz, self.ntri, self.nori) = self.delta.shape
        
    def test(self, fn_invcov_full=None, fn_invcov_diag=None):
        
        if (fn_invcov_full is not None) and (fn_invcov_diag is not None):
            self.likelihood_accuracy_test(fn_invcov_full, fn_invcov_diag)
        
        self.likelihood_accuracy_test_for_iz_itri(iz=0, itri=0)
        chi2_check = self.get_chi2_diagonal_in_triangle_shape_for_iz_itri_check(iz=0, itri=0)
        
    def likelihood_accuracy_test(self, fn_invcov_full, fn_invcov_diag):

        invcov_full = np.load(fn_invcov_full)
        invcov_diag = np.load(fn_invcov_diag)
        print('invcov_diag.shape', invcov_diag.shape)

        chi2_full = self.get_chi2_full(invcov=invcov_full)
        chi2_diag = self.get_chi2_diagonal_in_triangle_shape(invcov=invcov_diag)

        print('Comparing results from:')
        print('    fn_invcov_full = {}'.format(fn_invcov_full))
        print('    fn_invcov_diag = {}'.format(fn_invcov_diag))
        print('    chi2_full = {}'.format(chi2_full))
        print('    chi2_diag = {}'.format(chi2_diag))
        print('    difference chi2_diag-chi2_full = {}'.format(chi2_diag-chi2_full))

    def likelihood_accuracy_test_for_iz_itri(self, iz, itri):

        self.cov_full, self.invcov_full = \
            self.get_cov_and_invcov_full_for_iz_itri(iz, itri)
        self.cov_diag, self.invcov_diag = \
            self.get_cov_and_invcov_diagonal_in_triangle_shape_for_iz_itri(iz, itri)

        chi2_full = self.get_chi2_full_for_iz_itri(iz, itri)
        chi2_diag = self.get_chi2_diagonal_in_triangle_shape_for_iz_itri(iz, itri)

        print('iz = {}, itri = {}'.format(iz, itri))
        print('    chi2_full = {}'.format(chi2_full))
        print('    chi2_diag = {}'.format(chi2_diag))
        print('    difference chi2_diag-chi2_full = {}'.format(chi2_diag-chi2_full))

    def get_cov_and_invcov_full_for_iz_itri(self, iz, itri):
        """Returns 2d numpy array of shape (nb*nori, nb*nori)"""
        cov_full = self.cov_calculator.get_cov_smallest_nondiagonal_block(iz, itri) \
                * self.cov_calculator._cov_rescale[iz, itri]
        invcov = np.linalg.inv(cov_full)
        return cov_full, invcov
    
    def get_cov_and_invcov_diagonal_in_triangle_shape_for_iz_itri(self, iz, itri):
        """Returns 3d numpy array of shape (nb, nb, nori)"""
        cov_diag = np.zeros((self.nb, self.nb, self.nori))
        invcov = np.zeros_like(cov_diag)
        for iori in range(self.nori):
            cov_diag[:,:,iori] = self.cov_calculator.get_cov_nb_x_nb_block(iz, itri, iori, iori) \
                    * self.cov_calculator._cov_rescale[iz, itri]
            invcov[:,:,iori] = np.linalg.inv(cov_diag[:,:,iori])
        return cov_diag, invcov

    def get_chi2_full(self, invcov=None):
        """Expects invcov to be a 4d numpy array of shape (nb*nori, nb*nori, nz, ntri)."""
        if invcov is None:
            invcov = self.invcov_full
        chi2 = 0.0
        for iz in range(self.nz):
            for itri in range(self.ntri):
                chi2 += self.get_chi2_full_for_iz_itri(iz, itri, invcov=invcov[:,:,iz,itri])
        return chi2

    # TODO need to think more carefully what's going on 
    # Transpose is needed 
    def get_chi2_full_for_iz_itri(self, iz, itri, invcov=None):
        """Expects invcov to be a 2d numpy array of shape (nb*nori, nb*nori)."""
        if invcov is None:
            invcov = self.invcov_full
        delta_tmp = (np.transpose(self.delta[:, iz, itri, :])).ravel()
        chi2 = np.matmul(delta_tmp, np.matmul(invcov, delta_tmp))
        return chi2

    def get_chi2_diagonal_in_triangle_shape(self, invcov=None):
        """Expects invcov to be a 3d numpy array of shape (nb, nb, nori)"""
        if invcov is None:
            invcov = self.invcov_diag
        print('get_chi2_diagonal_in_triangle_shape: invcov.shape', invcov.shape)
        chi2 = 0.0
        for iz in range(self.nz):
            for itri in range(self.ntri):
                chi2 += self.get_chi2_diagonal_in_triangle_shape_for_iz_itri(iz, itri, invcov=invcov[:,:,iz,itri,:])
        return chi2

    def get_chi2_diagonal_in_triangle_shape_for_iz_itri(self, iz, itri, invcov=None):
        """Expects invcov to be a 3d numpy array of shape (nb, nb, nori)"""
        if invcov is None:
            invcov = self.invcov_diag
        chi2 = 0.0
        for iori in range(self.nori):
            delta_tmp = self.delta[:, iz, itri, iori]
            chi2 += np.matmul(delta_tmp, np.matmul(invcov[:,:,iori], delta_tmp))
        return chi2

    def get_chi2_diagonal_in_triangle_shape_for_iz_itri_check(self, iz, itri):
        
        nb = self.nb
        nori = self.nori

        cov_diag = np.zeros_like(self.cov_full)

        for iori in range(nori):     
            Mstart = nb * iori
            Mend = nb * (iori + 1)
            Nstart = nb * iori
            Nend = nb * (iori + 1)

            cov_diag[Mstart:Mend, Nstart:Nend] = self.cov_full[Mstart:Mend, Nstart:Nend]
            assert np.allclose(cov_diag[Mstart:Mend, Nstart:Nend], self.cov_diag[:,:,iori]), \
                (cov_diag[Mstart:Mend, Nstart:Nend], self.cov_diag[:,:,iori])
    
        invcov_tmp = np.linalg.inv(cov_diag)
        chi2 = self.get_chi2_full(invcov=invcov_tmp)        
        print('chi2 diag check = {}'.format(chi2))

        return chi2

    @staticmethod
    def load_delta_data_vector(fn0, fn1):
        data1 = np.load(fn1)
        data0 = np.load(fn0)
        delta = data1 - data0
        return delta

def check_identity(cov, invcov, rtol=1e-3, atol=1e-8):

    (nbxnori, nbxnori, nz, ntri) = cov.shape
    I0 = np.identity(nbxnori)

    for iz in range(nz):
        for itri in range(ntri):

            print('iz = {}, itri = {}'.format(iz, itri))

            cov_tmp = cov[:, :, iz, itri] 
            invcov_tmp = invcov[:, :, iz, itri]

            check1 = np.matmul(cov_tmp, invcov_tmp) - I0
            check2 = np.matmul(invcov_tmp, cov_tmp) - I0
            
            print('check1 = ', check1)
            print('check2 = ', check2)
            #TODO why this part is so slow???
            #is_identity_1 = np.allclose(check1, tol=rtol)
            #is_identity_2 = np.allclose(I2, I0, rtol=rtol)

            #print('is_identity = {}, {}'.format(is_identity_1, is_identity_2))


def check_cov_symmetric(cov):

    from lss_theory.math_utils.matrix import check_matrix_symmetric

    (nbxnori, nbxnori, nz, ntri) = cov.shape

    is_symmetric = True
    for iz in range(nz):
        for itri in range(ntri):

            print('iz = {}, itri = {}'.format(iz, itri))

            cov_tmp = cov[:, :, iz, itri] 
            is_symmetric = check_matrix_symmetric(cov_tmp)

            print('is_symmetric = {}'.format(is_symmetric))
            if is_symmetric == False:
                return is_symmetric

    return is_symmetric

if __name__ == '__main__':
    """
    Example usage:
        python3 -m lss_theory.scripts.get_covariance_b3d_rsd ./lss_theory/inputs/get_covariance_b3d_rsd.yaml
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file", type=str, default=None,
        help="path to config file."
    )

    command_line_args = parser.parse_args()

    with open(command_line_args.config_file) as file:
        info = yaml.load(file, Loader=yaml.FullLoader)
    print('info = {}'.format(info))

    cov_calculator = Bispectrum3DRSDCovarianceCalculator(info)
    
    cov_type = 'full'

    fn_cov = os.path.join(info['plot_dir'], 'cov_%s.npy'%cov_type)
    fn_invcov = os.path.join(info['plot_dir'], 'invcov_%s.npy'%cov_type)

    #HACK
    iz = 0
    itri = 1
    iori = 0
    jori = 0
    block = cov_calculator.get_cov_nb_x_nb_block(iz, itri, iori, jori)

    cov_calculator.get_and_save_cov(fn_cov, cov_type=cov_type, do_invcov=True)
    #cov = np.load(fn_cov)
    #cov_calculator.cov = cov
    #cov_calculator.invcov = cov_calculator.get_invcov()
    cov_calculator.save_invcov(fn_invcov)
    
    cov = np.load(fn_cov)
    is_symmetric = check_cov_symmetric(cov)

    invcov = np.load(fn_invcov)
    is_symmetric = check_cov_symmetric(invcov)

    #TODO why this part is so slow???
    #check_identity(cov, invcov)    
    
    #cov_calculator.load_cov_from_fn(fn)

    #fn_invcov = './plots/theory/covariance/b3d_rsd_theta1_phi12_2_4/fnl_0/nk_11/invcov_full.npy'
    #cov_calculator.load_invcov_from_fn(fn_invcov)

    #cov_calculator.get_and_save_invcov(fn_invcov)

    # Tests:
    #fn1 = './plots/theory/bispectrum_oriented_theta1_phi12_2_4/fnl_1/nk_11/bis_rsd.npy'
    #fn0 = './plots/theory/bispectrum_oriented_theta1_phi12_2_4/fnl_0/nk_11/bis_rsd.npy'
        
    #like_test = LikelihoodTest(cov_calculator, fn0, fn1)
    
    #fn_cov_diag = './plots/theory/covariance/b3d_rsd_theta1_phi12_2_4/fnl_0/nk_11/cov_diag.npy'
    #fn_cov_full = './data/debug_grs_ingredients/nk_11/bis_rsd_v27/cov_full.npy'
    
    #TODO move this file invcov file to ./data once tests are done
    #fn_invcov_diag = './plots/theory/covariance/b3d_rsd_theta1_phi12_2_4/fnl_0/nk_11/invcov_diag.npy'
    #fn_invcov_full = './data/debug_grs_ingredients/nk_11/bis_rsd_v27/invcov_full.npy'

    #cov_diag = np.load(fn_cov_diag)
    #invcov_diag = np.load(fn_invcov_diag)
    #print('cov_diag.shape = {}'.format(cov_diag.shape))
    #print('invcov_diag.shape = {}'.format(invcov_diag.shape))
    #like_test.test(fn_invcov_full=fn_invcov_full, fn_invcov_diag=fn_invcov_diag)

    #TODO add test to make inverse went well
    
