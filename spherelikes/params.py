import numpy as np

from cobaya.yaml import yaml_load_file

# TODO: Might want to make this a class that deals with intefacing with the survey par file
# so we can change survey par file format in the future, and don't need to change
# how the information is being served to the other classes.

def get_bias_params_for_survey_file(survey_par_file, fix_to_default=False, include_latex=True):
    
    bias_default = get_bias_default_for_survey_file(survey_par_file)
    nsample, nz = get_nsample_and_nz_for_survey_file(survey_par_file)
    
    bias_params = {}
    fractional_delta = 0.03 # about 2x delta As/As in Planck 2018
         
    for isample in range(nsample):
        for iz in range(nz):
            
            key = 'gaussian_bias_sample_%s_z_%s' % (isample + 1, iz + 1)
            
            latex = 'b_g^{%i}(z_{%i})' % (isample + 1, iz + 1)
            default_value = bias_default[isample, iz]
            
            scale = default_value * fractional_delta
            scale_ref = scale/10.0
            
            if fix_to_default is True:
                if include_latex is True:
                    value = {'value': default_value, 
                            'latex': latex
                            }
                else: 
                    value = default_value
            else:
                value = {'prior': {'dist': 'norm', 'loc': default_value, 'scale': scale},
                        'ref': {'dist': 'norm', 'loc': default_value, 'scale': scale_ref},
                        'propose': scale_ref,
                        'latex': latex,
                        }

            bias_params[key] = value

    return bias_params


def get_nsample_and_nz_for_survey_file(survey_par_file):
    survey_pars = yaml_load_file(survey_par_file)
    
    nz = len(survey_pars['zbin_lo'])
    nsample = len(survey_pars['sigz_over_one_plus_z'])
    return nsample, nz

def get_bias_default_for_survey_file(survey_par_file):
    """Returns a tuple (bias, nsample, nz) where bias is a 2-d numpy
    array of shape (nsample, nz) for the default bias values
    given by the input survey_par_file, and nsample and nz are
    the number of galaxy samples and redshift bins respectively."""

    nsample, nz = get_nsample_and_nz_for_survey_file(survey_par_file)

    survey_pars = yaml_load_file(survey_par_file)
    
    bias_default_values = np.zeros((nsample, nz)) 
    for isample in range(nsample):
        b = np.array(survey_pars['galaxy_bias%s'%(isample+1)])
        assert nz == b.size
        bias_default_values[isample, :] = b
    
    return bias_default_values