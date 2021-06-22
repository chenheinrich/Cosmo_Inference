import yaml
import argparse
import numpy as np
import copy
import sys

from lss_theory.fisher.derivative_generic import Derivatives
from lss_theory.fisher.derivative_generic import DerivativeConvergence
from lss_theory.fisher.fisher_generic import Fisher

class BispectrumMultipoleDerivatives(Derivatives):
    
    def __init__(self, info, ignore_cache=False, do_save=False, \
            parent_dir="./results/bis_mult/derivatives"):
        super().__init__(info, ignore_cache, do_save, parent_dir)
    
    def _setup_metadata(self):
        self._metadata = self._info

    def _get_signal_for_info(self, info): 
        from lss_theory.scripts.get_bis_mult import get_galaxy_bis_mult
        return get_galaxy_bis_mult(info)
 
class BispectrumMultipoleFisher(Fisher):

    def __init__(self, info):
        self._cov_type = info['fisher']['cov_type'] 
        super().__init__(info)

    def _setup_dims(self):
        # nori is really nlm here
        (self._nparam, self._nb, self._nz, self._ntri, self._nori) = \
            self._derivatives.shape

    def _load_invcov(self):
        #TODO need to put in the right covariance path
        #TODO temporary, we need to change this to 
        if self._cov_type == 'full':
            invcov_path = '/Users/chenhe/Research/My_Projects/SPHEREx/SPHEREx_forecasts/git/spherex_cobaya/plots/theory/covariance/b3d_rsd_theta1_phi12_2_4/fnl_0/nk_11/invcov_full.npy'
        elif self._cov_type == 'diagonal_in_lm':
            invcov_path = '/Users/chenhe/Research/My_Projects/SPHEREx/SPHEREx_forecasts/git/spherex_cobaya/plots/theory/covariance/b3d_rsd_theta1_phi12_2_4/fnl_0/nk_11/invcov_diag.npy'
        
        invcov = np.load(invcov_path)
        print('invcov.shape = {}'.format(invcov.shape))
        
        expected_ndim = 4 if self._cov_type == 'full' else 5
        assert len(invcov.shape) == expected_ndim
        
        return invcov

    #TODO temporary solution, could do better in DerivativeConvergence
    def _setup_module_and_class_names(self):
        self._module_name = 'lss_theory.fisher.fisher_bis_mult'
        self._class_name = 'BispectrumMultipoleDerivatives'
        self._derivative_dir = './results/bis_mult/derivatives/'
    
    def _get_fisher_matrix_element(self, iparam, jparam):

        if self._cov_type == "full":

            f = 0
            for iz in range(self._nz):
                for itri in range(self._ntri):
                    der_i = (np.transpose(self._derivatives[iparam, :, iz, itri, :])).ravel()
                    der_j = (np.transpose(self._derivatives[jparam, :, iz, itri, :])).ravel()
                    invcov_tmp = self._invcov[:, :, iz, itri]
                    tmp = np.matmul(invcov_tmp, der_j)
                    f += np.matmul(der_i, tmp)

        elif self._cov_type == "diagonal_in_lm":

            f = 0
            for iz in range(self._nz):
                for itri in range(self._ntri):
                    for iori in range(self._nori):
                        der_i = self._derivatives[iparam, :, iz, itri, iori]
                        der_j = self._derivatives[jparam, :, iz, itri, iori]
                        invcov_tmp = self._invcov[:, :, iz, itri, iori]
                        tmp = np.matmul(invcov_tmp, der_j)
                        f += np.matmul(der_i, tmp)

        else:
            msg = "You specified cov_type = %s, but needs to be\
                'full' or 'diagonal_in_lm'."%(self._cov_type)
            print(msg)
            sys.exit() #TODO do proper error handling
        

def check_for_convergence(info):
    module_name = 'lss_theory.fisher.fisher_bis_mult'
    class_name = 'BispectrumMultipoleDerivatives'
    der_conv = DerivativeConvergence(info, module_name, class_name, \
        ignore_cache=False, do_save=True,\
        parent_dir = './results/bis_mult/derivatives/')
    
if __name__ == '__main__':
    """
    Example usage:
        python3 -m lss_theory.fisher.fisher_bis_mult ./lss_theory/inputs/get_fisher_bis_mult.yaml 
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

    bis_mult_fisher = BispectrumMultipoleFisher(info_input)
    
    
    