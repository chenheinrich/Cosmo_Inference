import yaml
import argparse
import numpy as np
import copy
import sys

from lss_theory.fisher.derivative_generic import AllDerivatives
from lss_theory.fisher.derivative_generic import AllDerivativesConvergence
from lss_theory.fisher.fisher_generic import Fisher
from lss_theory.params.survey_par import SurveyPar
from lss_theory.params.other_par import OtherPar

class BispectrumMultipole_AllDerivatives(AllDerivatives):
    
    def __init__(self, info, ignore_cache=False, do_save=False, \
            parent_dir="./results/bis_mult/derivatives"):

        super().__init__(info, ignore_cache, do_save, parent_dir)
    
    def get_other_par(self, info):

        survey_par = SurveyPar(info['survey_par_file'])

        gaussian_bias = survey_par.get_galaxy_bias_array()
        (nsample, nz) = gaussian_bias.shape

        params_dict = {}
        for isample in range(nsample):
            for iz in range(nz):
                params_dict['gaussian_bias_s%i_z%i'%(isample+1, iz+1)] = gaussian_bias[isample, iz]
        
        other_par = OtherPar(params_dict)

        return other_par
    
    def get_param_set_definition(self, info):
        other_par = self.get_other_par(info)
        param_set_def = {\
            "*gaussian_biases": other_par.params_list, 
            "*gaussian_biases_first_two": [other_par.params_list[0], other_par.params_list[1]]}
        return param_set_def

    def _get_signal_for_info(self, info): 
        from lss_theory.scripts.get_bis_mult import get_galaxy_bis_mult
        return get_galaxy_bis_mult(info)
 
class BispectrumMultipoleFisher(Fisher):

    def __init__(self, info, 
            inverse_atol, 
            der_conv_eps,
            der_conv_std_threshold,
            der_conv_axis_to_vary
            ):
        self._cov_type = info['fisher']['cov_type'] 
        super().__init__(info, 
            inverse_atol=inverse_atol, \
            der_conv_eps=der_conv_eps, \
            der_conv_std_threshold=1e-2, \
            der_conv_axis_to_vary=2
        )

    def _setup_dims(self):
        # nori is really nlm here
        (self._nparam, self._nb, self._nz, self._ntri, self._nlm) = \
            self._derivatives.shape

    def _load_invcov(self):
        #TODO need to put in the right covariance path
        #TODO temporary, we need to change this to 
        if self._cov_type == 'full':
            invcov_path = '/Users/chenhe/Research/My_Projects/SPHEREx/SPHEREx_forecasts/git/spherex_cobaya/plots/theory/covariance/b3d_rsd_theta1_phi12_2_4/fnl_0/nk_11/invcov_full.npy'
        elif self._cov_type == 'diagonal_in_lm':
            invcov_path = '/Users/chenhe/Research/My_Projects/SPHEREx/SPHEREx_forecasts/git/spherex_cobaya/plots/theory/covariance/b3d_rsd_theta1_phi12_2_4/fnl_0/nk_11/invcov_diag.npy'
        
        invcov = np.load(invcov_path)
        expected_ndim = 4 if self._cov_type == 'full' else 5

        print('invcov.shape = {}'.format(invcov.shape))
        assert len(invcov.shape) == expected_ndim, (len(invcov.shape), expected_ndim)
        
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
            
            print('iparam, jparam, f', iparam, jparam, f)
            return f

        elif self._cov_type == "diagonal_in_lm":

            f = 0
            for iz in range(self._nz):
                for itri in range(self._ntri):
                    for ilm in range(self._nlm):
                        der_i = self._derivatives[iparam, :, iz, itri, ilm]
                        der_j = self._derivatives[jparam, :, iz, itri, ilm]
                        invcov_tmp = self._invcov[:, :, iz, itri, ilm]
                        tmp = np.matmul(invcov_tmp, der_j)
                        f += np.matmul(der_i, tmp)

        else:
            msg = "You specified cov_type = %s, but needs to be\
                'full' or 'diagonal_in_lm'."%(self._cov_type)
            print(msg)
            sys.exit() #TODO do proper error handling
    
    def _check_matrix_symmetric(self, a, rtol=1e-05, atol=1e-08):
        return np.allclose(a, a.T, rtol=rtol, atol=atol)

def check_for_convergence(info):
    module_name = 'lss_theory.fisher.fisher_bis_mult'
    class_name = 'BispectrumMultipole_AllDerivatives'
    der_conv = AllDerivativesConvergence(info, module_name, class_name, \
        ignore_cache=False, do_save=True,\
        parent_dir = './results/bis_mult/derivatives/',\
        eps = 0.001)
    return der_conv

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
    
    do_all_derivatives = False
    do_derivative_convergence = True
    do_plot_derivative = False
    do_fisher = False
    
    if do_all_derivatives == True:
        all_derivatives = BispectrumMultipole_AllDerivatives(
            info_input,\
            ignore_cache=False, \
            do_save=True,\
            parent_dir = './results/bis_mult/derivatives/'\
        )
        print('all_derivatives.data.shape', all_derivatives.data.shape)
        print(all_derivatives.metadata)

    # Get converged derivatives
    if do_derivative_convergence == True:
        deriv_converged = check_for_convergence(info_input)

    # Plot converged derivatives
    if do_plot_derivative == True:
        from lss_theory.scripts.get_b3d_rsd import get_data_spec
        from lss_theory.plotting.bis_plotter import Bispectrum3DRSDDerivativePlotter
        
        data_spec = get_data_spec(info)
        deriv_converged = check_for_convergence(info_input)
        deriv_plotter = Bispectrum3DRSDDerivativePlotter(deriv_converged, data_spec)
        deriv_plotter.make_plots()

    # Get Fisher
    if do_fisher == True:
        b3d_rsd_fisher = BispectrumMultipoleFisher(info_input, \
            inverse_atol=1e-3,
            der_conv_eps=1e-3,
            der_conv_std_threshold=1e-2,
            der_conv_axis_to_vary=2
        )

    

    
    