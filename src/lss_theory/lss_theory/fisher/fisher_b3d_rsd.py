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

class Bispectrum3DRSD_AllDerivatives(AllDerivatives):

    def __init__(self, info, ignore_cache=False, do_save=False, \
            parent_dir="./results/b3d_rsd/derivatives"):
        
        super().__init__(info, ignore_cache, do_save, parent_dir)
    
    def get_other_par(self, info):
        params_dict = self._get_gaussian_bias_default(info)
        params_dict.update(self._get_b2_default(info))
        other_par = OtherPar(params_dict)
        return other_par
    
    def _get_gaussian_bias_default(self, info):

        survey_par = SurveyPar(info['survey_par_file'])

        gaussian_bias = survey_par.get_galaxy_bias_array()
        (nsample, nz) = gaussian_bias.shape

        params_dict = {}
        for isample in range(nsample):
            for iz in range(nz):
                name = self._get_key_from_name_isample_iz('gaussian_bias', isample, iz)
                params_dict[name] = gaussian_bias[isample, iz]

        return params_dict
    
    def _get_b2_default(self, info):
        """Returns a dictionary for the second-order galaxy 
        bias, a key for each galaxy sample and redshift.
        """

        survey_par = SurveyPar(info['survey_par_file'])
        
        nsample = survey_par.get_nsample() #TODO change to property!
        nz = survey_par.get_nz()

        b1_dict = self._get_gaussian_bias_default(info)

        params_dict = {}
        for isample in range(nsample):
            for iz in range(nz):
                key_b1 = self._get_key_from_name_isample_iz('gaussian_bias', isample, iz)
                key_b2 = self._get_key_from_name_isample_iz('b2', isample, iz)
                b2 = self._get_b2_Lezeyras_2016_for_b1(b1_dict[key_b1])
                params_dict[key_b2] = b2

        return params_dict

    @staticmethod
    def _get_key_from_name_isample_iz(name, isample, iz):
        return '%s_s%i_z%i'%(name, isample+1, iz+1)

    @staticmethod
    def _get_b2_Lezeyras_2016_for_b1(b1):
        """Returns b2 given b1 using GR fit from Lezeyras et al. 2016
        https://arxiv.org/pdf/1511.01096.pdf Eq. 5.2 
        (Also used in SPHEREx forecast) defined with 
        delta_g(x) = b1 * delta_m(x) + 0.5 * b2 * delta_m^2(x).
        """
        b2 = 0.412 - 2.143 * b1 + 0.929 * b1 * b1 + 0.008 * b1 * b1 * b1
        return b2

    def get_param_set_definition(self, info):
        """Returns a dictionary that map a name denoting a set of parameters,
        to the list of names for all the parameters it contains."""
        
        b1_param_list = list(self._get_gaussian_bias_default(info).keys())
        b2_param_list = list(self._get_b2_default(info).keys())
        param_set_def = {\
            "*gaussian_biases": b1_param_list,
            "*gaussian_biases_first_two": [b1_param_list[0], b1_param_list[1]],
            "*b2": b2_param_list,
            "*b2_first_two": [b2_param_list[0], b2_param_list[1]],
        }

        return param_set_def

    def _get_signal_for_info(self, info): 
        from lss_theory.scripts.get_b3d_rsd import get_galaxy_bis
        return get_galaxy_bis(info)

class Bispectrum3DRSDFisher(Fisher):

    def __init__(self, info, 
            inverse_atol, 
            der_conv_eps,
            der_conv_std_threshold,
            der_conv_axis_to_vary
            ):
        self._cov_type = info['fisher']['cov_type'] 
        self._invcov_path = info['fisher']['invcov_path'] 
        super().__init__(info, 
            inverse_atol=inverse_atol, \
            der_conv_eps=der_conv_eps, \
            der_conv_std_threshold=1e-2, \
            der_conv_axis_to_vary=2
        )

    def _setup_dims(self):
        (self._nparam, self._nb, self._nz, self._ntri, self._nori) = \
            self._derivatives.shape

    def _load_invcov(self):

        invcov = np.load(self._invcov_path)
        print('invcov.shape = {}'.format(invcov.shape))
        
        expected_ndim = 4 if self._cov_type == 'full' else 5
        assert len(invcov.shape) == expected_ndim, (len(invcov.shape), expected_ndim)

        #HACK
        #raise error properly here with a good messages
        
        return invcov

    #TODO temporary solution, could do better in DerivativeConvergence
    def _setup_module_and_class_names(self):
        self._module_name = 'lss_theory.fisher.fisher_b3d_rsd'
        self._class_name = 'Bispectrum3DRSD_AllDerivatives'
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

                        #HACK
                        #is_all_zero = np.all((der_j == 0))
                        #print('iparam, jparam', iparam, jparam)
                        #print('is_all_zero = {}'.format(is_all_zero))

                        invcov_tmp = self._invcov[:, :, iz, itri, iori] 
                        tmp = np.matmul(invcov_tmp, der_j)
                        f += np.matmul(der_i, tmp)
            
            print('iparam, jparam, f', iparam, jparam, f)
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
    class_name = 'Bispectrum3DRSD_AllDerivatives'
    der_conv = AllDerivativesConvergence(info, module_name, class_name, \
        ignore_cache=False, do_save=True,\
        parent_dir = './results/b3d_rsd/derivatives/', \
        eps = 0.001)
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
    
    do_all_derivatives = False
    do_derivative_convergence = False
    do_plot_derivative = False
    do_fisher = True
    
    if do_all_derivatives == True:
        all_derivatives = Bispectrum3DRSD_AllDerivatives(
            info_input,\
            ignore_cache=False, \
            do_save=True,\
            parent_dir = './results/b3d_rsd/derivatives/'\
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
        b3d_rsd_fisher = Bispectrum3DRSDFisher(info_input, \
            inverse_atol=1e-3,
            der_conv_eps=1e-3,
            der_conv_std_threshold=1e-2,
            der_conv_axis_to_vary=2
        )

    

    
    