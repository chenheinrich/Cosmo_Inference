import yaml
import argparse
import numpy as np
import copy
import sys

from theory.fisher.derivative_generic import Derivatives
from theory.fisher.derivative_generic import DerivativeConvergence
from theory.fisher.fisher_generic import Fisher

class Bispectrum3DRSDDerivatives(Derivatives):
    
    def __init__(self, info, ignore_cache=False, do_save=False, \
            parent_dir="./results/b3d_rsd/derivatives"):
        super().__init__(info, ignore_cache, do_save, parent_dir)
    
    def _setup_metadata(self):
        self._metadata = self._info

    def _get_signal_for_info(self, info): 
        from theory.scripts.get_bis_rsd import get_galaxy_bis
        return get_galaxy_bis(info)


class Bispectrum3DRSDFisher(Fisher):

    def __init__(self, info):
        self._cov_type = info['fisher']['cov_type'] 
        super().__init__(info)

    def _setup_dims(self):
        (self._nparam, self._nb, self._nz, self._ntri, self._nori) = \
            self._derivatives.shape

    def _load_invcov(self):
        #TODO temporary, we need to change this to 
        if self._cov_type == 'full':
            #invcov_path = '/Users/chenhe/Research/My_Projects/SPHEREx/SPHEREx_forecasts/git/SphereLikes/plots/theory/covariance/b3d_rsd_theta1_phi12_2_4/fnl_0/nk_11/invcov_full.npy'
            invcov_path = '/Users/chenhe/Research/My_Projects/SPHEREx/SPHEREx_forecasts/git/SphereLikes/data/debug_grs_ingredients/nk_11/bis_rsd_v27/invcov_full.npy'
        elif self._cov_type == 'diagonal_in_triangle_orientation':
            invcov_path = '/Users/chenhe/Research/My_Projects/SPHEREx/SPHEREx_forecasts/git/SphereLikes/plots/theory/covariance/b3d_rsd_theta1_phi12_2_4/fnl_0/nk_11/invcov_diag.npy'
        
        invcov = np.load(invcov_path)
        print('invcov.shape = {}'.format(invcov.shape))
        
        expected_ndim = 4 if self._cov_type == 'full' else 5
        assert len(invcov.shape) == expected_ndim
        
        return invcov

    #TODO temporary solution, could do better in DerivativeConvergence
    def _setup_module_and_class_names(self):
        self._module_name = 'theory.fisher.fisher_b3d_rsd'
        self._class_name = 'Bispectrum3DRSDDerivatives'
        self._derivative_dir = './results/b3d_rsd/derivatives/'
    
    def _get_fisher_matrix_element(self, iparam, jparam):

        if self._cov_type == "full":

            f = 0
            for iz in range(self._nz):
                for itri in range(self._ntri):
                    der_i = (np.transpose(self._derivatives[iparam, :, iz, itri, :])).ravel()
                    der_j = (np.transpose(self._derivatives[jparam, :, iz, itri, :])).ravel()
                    invcov_tmp = self._invcov[:, :, iz, itri]
                    is_symmetric = self._check_matrix_symmetric(invcov_tmp)
                    if is_symmetric == False:
                        print('invcov_tmp = ', invcov_tmp)
                    print('is_symmetric', is_symmetric)
                    # invcov is NOT SYMMETRIC!!! was cov symmetric?
                    # need to regenerate??
                    tmp = np.matmul(invcov_tmp, der_j)
                    f += np.matmul(der_i, tmp)

            print('iparam, jparam, f', iparam, jparam, f)
            return f

        elif self._cov_type == "diagonal_in_triangle_orientation":

            f = 0
            for iz in range(self._nz):
                for itri in range(self._ntri):
                    for iori in range(self._nori):
                        der_i = self._derivatives[iparam, :, iz, itri, iori]
                        der_j = self._derivatives[jparam, :, iz, itri, iori]
                        invcov_tmp = self._invcov[:, :, iz, itri, iori]
                        tmp = np.matmul(invcov_tmp, der_j)
                        f += np.matmul(der_i, tmp)

            return f

        else:
            msg = "You specified cov_type = %s, but needs to be\
                'full' or 'diagonal_in_triangle_orientation'."%(self._cov_type)
            print(msg)
            sys.exit() #TODO do proper error handling
        
    def _check_matrix_symmetric(self, a, rtol=1e-05, atol=1e-08):
        return np.allclose(a, a.T, rtol=rtol, atol=atol)

#TODO merge fisher and derivatives so that they use the same save metadata functions etc.


def check_for_convergence(info):
    module_name = 'theory.fisher.fisher_b3d_rsd'
    class_name = 'Bispectrum3DRSDDerivatives'
    der_conv = DerivativeConvergence(info, module_name, class_name, \
        ignore_cache=False, do_save=True,\
        parent_dir = './results/b3d_rsd/derivatives/')
    
if __name__ == '__main__':
    """
    Example usage:
        python3 -m galaxy_3d.theory.fisher.fisher_b3d_rsd ./galaxy_3d/inputs_theory/get_fisher_b3d_rsd.yaml 
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

    info_input = copy.deepcopy(info)
    #convergence_results = check_for_convergence(info_input)

    b3d_rsd_fisher = Bispectrum3DRSDFisher(info_input)
    

    
    