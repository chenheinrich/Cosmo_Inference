import numpy as np
import argparse
import yaml
import os

from theory.covariance import Bispectrum3DBaseCovarianceCalculator
from theory.utils import file_tools

class CovTester():

    def __init__(self, cov_calculator, fn0, fn1):
        self.cov_calculator = cov_calculator
        self.delta = self.load_delta_data_vector(fn0, fn1)
        (self.nb, self.nz, self.ntri) = self.delta.shape
        
    def test(self, fn_invcov = None, fn_cov = None):
        
        if (fn_invcov is not None):
            self.chi2_test(fn_invcov)

            if (fn_cov is not None):
                self.inverse_test(fn_cov, fn_invcov)
        
    def inverse_test(self, fn_cov, fn_invcov):
        """Expect invcov and cov to be of shape (nb, nb, nz, ntri)"""
        print('Inverse test:')
        atol = 1e-6
        rtol = 1e-5
        invcov = np.load(fn_invcov)
        cov = np.load(fn_cov)
        id0 = np.diag(np.ones(self.nb))
        for iz in range(self.nz):
            for itri in range(self.ntri):
                print('... iz = {}, itri = {}'.format(iz, itri))
                id1 = np.matmul(invcov[:, :, iz, itri], cov[:, :, iz, itri])
                id2 = np.matmul(cov[:, :, iz, itri], invcov[:, :, iz, itri])
                assert np.allclose(id1, id0, rtol=rtol, atol=atol), (id1-id0)
                assert np.allclose(id2, id0, rtol=rtol, atol=atol), (id2-id0)
        print('... inverse test successful!')

    def chi2_test(self, fn_invcov):

        invcov = np.load(fn_invcov)
        print('invcov.shape', invcov.shape)

        chi2_full = self.get_chi2_full(invcov)

        print('Comparing results from:')
        print('    fn_invcov = {}'.format(fn_invcov))
        print('    chi2_full = {}'.format(chi2_full))

        sigma_fnl = chi2_full**(-0.5)
        print('    sigma_fnl = {}'.format(sigma_fnl))

    def get_chi2_full(self, invcov):
        """Expects invcov to be a 4d numpy array of shape (nb, nb, nz, ntri)."""
        chi2 = 0.0
        for iz in range(self.nz):
            for itri in range(self.ntri):
                chi2 += self.get_chi2_full_for_iz_itri(iz, itri, invcov[:,:,iz,itri])
        return chi2

    def get_chi2_full_for_iz_itri(self, iz, itri, invcov):
        """Expects invcov to be a 2d numpy array of shape (nb, nb)."""
        delta_tmp = self.delta[:, iz, itri]
        chi2 = np.matmul(delta_tmp, np.matmul(invcov, delta_tmp))
        return chi2

    @staticmethod
    def load_delta_data_vector(fn0, fn1):
        data0 = np.load(fn0)
        data1 = np.load(fn1)
        delta = data1 - data0
        return delta


if __name__ == '__main__':
    """
    Example usage:
        python3 -m theory.scripts.get_covariance_b3d_base ./inputs_theory/get_covariance_b3d_base.yaml
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

    cov_calculator = Bispectrum3DBaseCovarianceCalculator(info)
    
    fn_cov = os.path.join(info['plot_dir'], 'cov.npy')
    fn_invcov = os.path.join(info['plot_dir'], 'invcov.npy')

    # Calculations
    cov_calculator.get_and_save_cov(fn_cov, do_invcov=True)
    cov_calculator.save_invcov(fn_invcov)
    #cov_calculator.get_and_save_invcov(fn_invcov)

    # Tests:
    fn1 = './plots/theory/bispectrum_base/fnl_1/nk_11/bis_base.npy'
    fn0 = './plots/theory/bispectrum_base/fnl_0/nk_11/bis_base.npy'
        
    cov_tester = CovTester(cov_calculator, fn0, fn1)
    
    fn_cov = './plots/theory/covariance/b3d_base/fnl_0/nk_11/cov.npy'
    fn_invcov = './plots/theory/covariance/b3d_base/fnl_0/nk_11/invcov.npy'
    cov_tester.chi2_test(fn_invcov=fn_invcov)
    #cov_tester.inverse_test(fn_cov=fn_cov, fn_invcov=fn_invcov)

    
