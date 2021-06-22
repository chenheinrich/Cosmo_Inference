import numpy as numpy
import matplotlib.pyplot as plt 
import os
import sys
import copy

from getdist import plots as gplots
import getdist
from cobaya.yaml import yaml_load_file

from spherex_cobaya.params import SurveyPar, CobayaPar
#TODO need to make this flexible for different classes
from analysis import log


def list_intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3 

class ChainPlotter():
    
    def __init__(self, args):

        self.logger = log.class_logger(self)

        self._analysis_settings = args['analysis_settings']
        #TODO ideally write out all the parameter values into a yaml file to be loaded anytime
        self._data_cosmo_par_file = args['data_cosmo_par_file']
        self._chain_dir = self._remove_trailing_slash(args['chain_dir'])
        self._roots = args['roots']
        
        base_name = os.path.basename(self._chain_dir)
        self._plot_dir = os.path.join(args['plot_dir'], base_name, self._roots[0])
        self._nsample = args['nsample']
        self._nz = args['nz']

        info = self.get_chain_updated_info()
        self._params_processor = ParamsProcessor(info)

        self.params_lcdm = self._params_processor.get_camb_params()
        self.params_lcdm.append('fnl')

        sampled_params = self._params_processor.get_sampled_params()
        self.params_lcdm = list_intersection(self.params_lcdm, sampled_params)
        print('self.params_lcdm', self.params_lcdm)

        self.params_base = ['fnl', 'logA']
        self.params_base = list_intersection(self.params_base, sampled_params)
        
        self.params_bias = self.get_params_bias(range(self._nsample), range(self._nz))
        self.params_bias = list_intersection(self.params_bias, sampled_params)
        
        self.params_sys = self._params_processor.get_sys_params()
        self.params_sys = list_intersection(self.params_sys, sampled_params)
        
        self.params = self.params_lcdm + self.params_bias + self.params_sys
        print('self.params = {}'.format(self.params))

        chain_updated_yaml = os.path.join(self._chain_dir, self._roots[0]+'.updated.yaml')
        self._cobaya_par = CobayaPar(chain_updated_yaml)
        self._survey_par = self._cobaya_par.get_survey_par()

        self.bias_default_values = self.get_bias_default_values()
        self.lcdm_sim_values = yaml_load_file(self._data_cosmo_par_file) 
        self.sim_values = dict(self.bias_default_values, **self.lcdm_sim_values) #TODO add sys params in the future

        self.priors = self.get_priors()

    def _remove_trailing_slash(self, string):
        result = string[:-1] if string.endswith('/') else string
        return result

    def get_chain_updated_info(self):
        path = os.path.join(self._chain_dir, self._roots[0]+'.updated.yaml')
        info = yaml_load_file(path)
        return info 

    def get_params_bias(self, isamples, izs):
        names = []
        for isample in isamples:
            for iz in izs:
                name = 'gaussian_bias_sample_%s_z_%s' % (isample+1, iz+1)
                names.append(name)
        return names

    def plot_all(self):
        self.plot_1d()
        self.plot_triangles()

    def plot_1d(self):
        if len(self.params_lcdm) > 0:
            self.plot_params_lcdm(plot_type='1d')
        if len(self.params_bias) > 0:
            self.plot_params_bias(plot_type='1d')

    def plot_triangles(self):
        if len(self.params_lcdm) > 0:
            self.plot_params_lcdm(plot_type='triangle')
        if len(self.params_bias) > 0:
            self.plot_params_bias(plot_type='triangle')

    def plot_params_lcdm(self, plot_type = '1d'):
        """Make 1d posterior plot per galaxy sample for bias parameters at all z."""
        params = self.params_lcdm
        plot_name = os.path.join(self._plot_dir, 'plot_%s_lcdm.png'%(plot_type))
        self.plot_params(plot_name, params=params, plot_type=plot_type)

    def plot_params_bias(self, plot_type = '1d'):
        """Make 1d posterior plot per galaxy sample for bias parameters at all z."""
        for isample in range(self._nsample):
            params = self.get_params_bias([isample], range(self._nz))
            params.extend(self.params_base)
            plot_name = os.path.join(self._plot_dir, \
                'plot_%s_bias_sample_%s.png'%(plot_type, isample+1))
            self.plot_params(plot_name, params=params, plot_type=plot_type)

    def plot_params(self, plot_name, params=None, plot_type='1d'):
        """Make 1d posterior plot for all parameters or those specified through a list params."""

        params = (params or self.params)

        g = gplots.get_subplot_plotter(
            chain_dir=self._chain_dir, 
            analysis_settings=self._analysis_settings)

        mcsamples = self.get_mcsamples(g, params)

        plot_func = {'1d': 'plots_1d', 'triangle': 'triangle_plot'}[plot_type]
        getattr(g, plot_func)([mcsamples], params=params, filled=True)

        for param in params:
            ax = g.get_axes_for_params(param)
            self.add_input_line_for_param(g, ax, param)
            self.add_prior_bands_for_param(g, ax, param)

        g.export(plot_name)
        self.logger.info('Saved plot: {}'.format(plot_name))

    def get_margestat(self, params=None):

        params = (params or self.params)

        g = gplots.get_subplot_plotter(
            chain_dir=self._chain_dir, 
            analysis_settings=self._analysis_settings)

        mcsamples = self.get_mcsamples(g, params)

        marge_stats = mcsamples.getMargeStats()
        print(marge_stats)

        return marge_stats

    def get_mcsamples(self, g, params):
        samples = g.sampleAnalyser.samplesForRoot(self._roots[0])
        print('Number of points in chain = {}'.format(samples.weights.size))

        p = samples.getParams()

        #HACK did not vary gaussian_bias_sample_1_z_1 unfortunately
    
        print('params = {}'.format(params))
        if 'gaussian_bias_sample_1_z_1' in params:
            params.remove('gaussian_bias_sample_1_z_1')
            print('params = {}'.format(params))
        
        sample_values = [getattr(p, param) for param in params]

        ranges = self.get_mcsamples_ranges()
        ranges['w'] = [-1, None] #HACK

        MCSamples = getdist.MCSamples(
            samples = sample_values, \
            weights = samples.weights,\
            loglikes = samples.loglikes, \
            names = params,\
            settings = self._analysis_settings, \
            ranges = ranges, 
        )
        return MCSamples

    def add_input_line_for_param(self, g, ax, param):
        g.add_x_marker(marker=self.sim_values[param], \
                ax=ax, ls='-', color = 'red')

    def add_prior_bands_for_param(self, g, ax, param):
        prior_dict = self.priors[param]['prior']
        prior = Prior(prior_dict)
        center, sigma = prior.get_center_and_sigma()
        if prior.is_uniform():
            g.add_x_bands(center, sigma, ax=ax, alpha1 = 0.5, alpha2 = 0)
        else:
            g.add_x_bands(center, sigma, ax=ax, alpha1 = 0.25, alpha2 = 0.2)

    def get_bias_default_values(self):
        """Returns a dictionary with bias name and default values"""
        from spherex_cobaya.theory.PowerSpectrum3D import make_dictionary_for_bias_params
        default_values = make_dictionary_for_bias_params(
            self._survey_par, fix_to_default=True, include_latex=False)
        return default_values

    def get_priors(self):
        """Get priors applied when running chains"""
        updated_yaml_file = os.path.join(self._chain_dir, self._roots[0] + '.updated.yaml')
        info = yaml_load_file(updated_yaml_file)
        params_info = info['params']
        return params_info

    def get_mcsamples_ranges(self):
        """Return a dictionary for parameter ranges in the GetDist format
        given priors loaded from the cobaya updated yaml file."""
        ranges = {}
        priors = copy.deepcopy(self.priors)
        for paramname in priors.keys():
            try: 
                pmin = priors[paramname]['prior']['min']
                pmax = priors[paramname]['prior']['max']
                ranges[paramname] = [pmin, pmax]
            except Exception as e:
                pass
        return ranges

class ParamsProcessor():

    """Input: entire dictionary stored in the updated yaml file of cobaya chains."""

    def __init__(self, info):
        self._info = info
        self._params_dict = self._info['params']

    def get_all_params(self):
        return self._params_dict.keys()

    def get_derived_params(self):
        derived_params = []
        for key in self._params_dict.keys():
            if isinstance(self._params_dict['key'], dict):
                if 'derived' in self._params_dict['key'].keys():
                    name = self._params_dict[key]
                    derived_params.append(name)    
        return derived_params

    def get_sampled_params(self):
        sampled_params = []
        for key in self._params_dict.keys():
            if 'prior' in self._params_dict[key].keys():
                sampled_params.append(key)
        print('sampled_params = {}'.format(sampled_params))
        return sampled_params
    
    # Theory params
    def get_camb_params(self):
        params = self._info['theory']['camb']['input_params']
        replace = {'As': 'logA', 'cosmomc_theta': 'theta_MC_100'}
        for param_to_replace in replace:
            if param_to_replace in params:
                params.remove(param_to_replace)
                params.append(replace[param_to_replace])
        return params

    def get_other_theory_params(self):
        params = []
        for key in self._info['theory'].keys():
            if key is not 'camb':
                theory_params = self._info['theory'][key]['input_params']
                params.extend(theory_params)
        return params

    # Likelihood params
    def get_sys_params(self):
        """Returns all systematic parameters in updated yaml 
        (those under 'likelihood')."""
        params = []
        for key in self._info['likelihood'].keys():
            input_params = self._info['likelihood'][key]['input_params']
            params.append(input_params)
        return params
    
    def get_latex_for_paramname(self, paramname):
        latex = self._params_dict[paramname]['latex']
        return latex

    def get_prior_dict_for_paramname(self, paramname):
        prior_dict = self._params_dict[paramname]['prior']
        return prior_dict


class Prior():

    """Input: dictionary of parameter prior as formatted by Cobaya chain yaml file"""

    def __init__(self, prior_dict):
        self._prior_dict = prior_dict

    def is_uniform(self):
        is_min_given = 'min' in self._prior_dict.keys()
        is_max_given = 'max' in self._prior_dict.keys()
        is_uniform = (is_min_given and is_max_given)
        return is_uniform

    def get_center_and_sigma(self):
        if self.is_uniform():
            xmin = self._prior_dict['min'] 
            xmax = self._prior_dict['max'] 
            center = (xmin+xmax) / 2.0
            sigma = (xmax-xmin) / 2.0
        else:
            if self._prior_dict['dist'] == 'norm':
                center = self._prior_dict['loc'] 
                sigma = self._prior_dict['scale'] 
            else:
                raise NotImplementedError
        return center, sigma

    def get_ranges(self, nstd=3.0):
        """Get ranges based on prior; for normal 
        distribution, default is 3 sigma bounds."""
        if self.is_uniform():
            lo = self._prior_dict['min']
            hi = self._prior_dict['max']
        else:
            if self._prior_dict['dist'] == 'norm':
                lo = self._prior_dict['loc'] - nstd * self._prior_dict['scale'] 
                hi = self._prior_dict['loc'] + nstd * self._prior_dict['scale'] 
            else:
                raise NotImplementedError
        return [lo, hi]

if __name__ == '__main__':
    """Example usage: 
    python3 -m analysis.plot ./analysis/inputs/ps_base.yaml
    """

    config_file = sys.argv[1]

    args = yaml_load_file(config_file)

    plotter = ChainPlotter(args)
    plotter.plot_1d()
    #plotter.plot_triangles()
    plotter.get_margestat()

