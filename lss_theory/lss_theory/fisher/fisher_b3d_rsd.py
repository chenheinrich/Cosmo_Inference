import yaml
import argparse
import numpy as np
import copy
import sys

from lss_theory.fisher.derivative_generic import Derivative
from lss_theory.fisher.derivative_generic import DerivativeConvergence
from lss_theory.fisher.fisher_generic import Fisher
from lss_theory.params.survey_par import SurveyPar
from lss_theory.params.other_par import OtherPar


def get_param_set_definition(info):
    #HACK
    other_par = get_other_par(info)
    param_set_def = {"*gaussian_biases": other_par.params_list}
    #param_set_def = {"*gaussian_biases": other_par.params_list[0:2]}
    return param_set_def

def get_other_par(info):

    survey_par = SurveyPar(info['survey_par_file'])

    gaussian_bias = survey_par.get_galaxy_bias_array()
    (nsample, nz) = gaussian_bias.shape

    params_dict = {}
    for isample in range(nsample):
        for iz in range(nz):
            params_dict['gaussian_bias_s%i_z%i'%(isample+1, iz+1)] = gaussian_bias[isample, iz]
    
    other_par = OtherPar(params_dict)

    return other_par
class Bispectrum3DRSDDerivative(Derivative):

    def __init__(self, info, ignore_cache=False, do_save=False, \
            parent_dir="./results/b3d_rsd/derivatives"):
        
        other_par = self._get_other_par(info)

        super().__init__(info, ignore_cache, do_save, parent_dir, other_par)
    
    def _get_param_set_definition(self, info):
        return get_param_set_definition(info)

    def _get_other_par(self, info):
        return get_other_par(info)

    def _setup_metadata(self):
        self._metadata = self._info

    def _get_signal_for_info(self, info): 
        from lss_theory.scripts.get_bis_rsd import get_galaxy_bis
        return get_galaxy_bis(info)


class Bispectrum3DRSDFisher(Fisher):

    def __init__(self, info, inverse_atol):
        self._cov_type = info['fisher']['cov_type'] 
        super().__init__(info, inverse_atol=inverse_atol)

    def _setup_dims(self):
        (self._nparam, self._nb, self._nz, self._ntri, self._nori) = \
            self._derivatives.shape

    def _load_invcov(self):
        
        #TODO temporary, we need to eliminate cov_type=full, inverse not stable
        if self._cov_type == 'full': # Inverse not stable!! #HACK
            invcov_path = './plots/theory/covariance/b3d_rsd_theta1_phi12_2_4/fnl_0/nk_11/test20210513/invcov_full.npy'
        
        elif self._cov_type == 'diagonal_in_triangle_orientation': 
            invcov_path = '/Users/chenhe/Research/My_Projects/SPHEREx/SPHEREx_forecasts/git/SphereLikes/plots/theory/covariance/b3d_rsd_theta1_phi12_2_4/fnl_0/nk_11/test20210526/invcov_diag_in_orientation.npy'
        
        invcov = np.load(invcov_path)
        print('invcov.shape = {}'.format(invcov.shape))
        
        expected_ndim = 4 if self._cov_type == 'full' else 5
        assert len(invcov.shape) == expected_ndim
        
        return invcov

    #TODO temporary solution, could do better in DerivativeConvergence
    def _setup_module_and_class_names(self):
        self._module_name = 'lss_theory.fisher.fisher_b3d_rsd'
        self._class_name = 'Bispectrum3DRSDDerivative'
        self._derivative_dir = './results/b3d_rsd/derivatives/'
    
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
                'full' or 'diagonal_in_triangle_orientation' \
                'diagonal_in_triangle_orientation2'."%(self._cov_type)
            print(msg)
            sys.exit() #TODO do proper error handling
        
    def _check_matrix_symmetric(self, a, rtol=1e-05, atol=1e-08):
        return np.allclose(a, a.T, rtol=rtol, atol=atol)

#TODO merge fisher and derivatives so that they use the same save metadata functions etc.


def check_for_convergence(info):
    module_name = 'lss_theory.fisher.fisher_b3d_rsd'
    class_name = 'Bispectrum3DRSDDerivative'
    der_conv = DerivativeConvergence(info, module_name, class_name, \
        ignore_cache=False, do_save=True,\
        parent_dir = './results/b3d_rsd/derivatives/')
    return der_conv
    
if __name__ == '__main__':
    """
    Example usage:
        python3 -m lss_theory.fisher.fisher_b3d_rsd ./lss_theory/inputs/get_fisher_b3d_rsd.yaml 
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
    
    do_derivative_convergence = True
    do_plot_derivative = False
    do_fisher = False
    
    # Get converged derivatives
    if do_derivative_convergence == True:
        deriv_converged = check_for_convergence(info_input)

    # Plot converged derivatives
    if do_plot_derivative == True:
        from lss_theory.scripts.get_bis_rsd import get_data_spec
        from lss_theory.plotting.bis_plotter import Bispectrum3DRSDDerivativePlotter
        
        data_spec = get_data_spec(info)
        deriv_converged = check_for_convergence(info_input)
        deriv_plotter = Bispectrum3DRSDDerivativePlotter(deriv_converged, data_spec)
        deriv_plotter.make_plots()

    # Get Fisher
    if do_fisher == True:
        b3d_rsd_fisher = Bispectrum3DRSDFisher(info_input, inverse_atol=1e-3)

    

    
    