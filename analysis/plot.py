import numpy as numpy
import matplotlib.pyplot as plt 
import os
import sys
import copy

from getdist import plots as gplots
import getdist
from cobaya.yaml import yaml_load_file

#TODO need to make this flexible for different classes
from spherelikes.params import get_bias_params_for_survey_file
from spherelikes.theories.base_classes.ps_base.ps_base import PowerSpectrumBase
import spherelikes.theories as theories
from analysis import log


class ChainPlotter():
    
    def __init__(self, args):

        self.logger = log.class_logger(self)
        self.args = args

        self.analysis_settings = self.args['analysis_settings']
        #TODO ideally write out all the parameter values into a yaml file to be loaded anytime
        self.data_cosmo_par_file = self.args['data_cosmo_par_file']
        self.chain_dir = self.args['chain_dir']
        self.roots = self.args['roots']

        if self.chain_dir[-1] == '/':
            chain_dir = self.chain_dir[:-1]
        base_name = os.path.basename(chain_dir)
        self.plot_dir = os.path.join(self.args['plot_dir'], base_name, self.roots[0])
        self.nsample = self.args['nsample']
        self.nz = self.args['nz']

        self.params_base = ['fnl', 'logA']
        # TODO might want to put this somewhere central
        self.params_lcdm = ['fnl', 'logA', 'ns', 'nrun', 'theta_MC_100', \
            'ombh2', 'omch2', 'omegak', 'mnu', 'w', 'wa', 'tau'] 
        self.params_bias = self.get_params_bias(range(self.nsample), range(self.nz))
        self.params_sys = []
        self.params = self.params_lcdm + self.params_bias + self.params_sys
        #TODO ini the fuure could check this against input_params in root + 'updated.yaml', input
        # params_sys for those under likelihood; and params_bias for those under spherelikes.theories

        self.survey_par_file = getattr(PowerSpectrumBase, 'survey_par_file')
        self.bias_default_values = self.get_bias_default_values()
        self.lcdm_sim_values = yaml_load_file(self.data_cosmo_par_file) 
        self.sim_values = dict(self.bias_default_values, **self.lcdm_sim_values) #TODO add sys params in the future

        self.priors = self.get_priors()

    def get_params_bias(self, isamples, izs):
        names = []
        for isample in isamples:
            for iz in izs:
                name = 'gaussian_bias_sample_%s_z_%s'%(isample+1, iz+1)
                names.append(name)
        return names

    def plot_all(self):
        self.plot_1d()
        self.plot_triangles()

    def plot_1d(self):
        self.plot_params_lcdm(plot_type='1d')
        self.plot_params_bias(plot_type='1d')

    def plot_triangles(self):
        self.plot_params_lcdm(plot_type='triangle')
        self.plot_params_bias(plot_type='triangle')

    def plot_params_lcdm(self, plot_type = '1d'):
        """Make 1d posterior plot per galaxy sample for bias parameters at all z."""
        params = self.params_lcdm
        plot_name = os.path.join(self.plot_dir, 'plot_%s_lcdm.png'%(plot_type))
        self.plot_params(plot_name, params=params, plot_type=plot_type)

    def plot_params_bias(self, plot_type = '1d'):
        """Make 1d posterior plot per galaxy sample for bias parameters at all z."""
        for isample in range(self.nsample):
            params = self.get_params_bias([isample], range(self.nz))
            params.extend(self.params_base)
            plot_name = os.path.join(self.plot_dir, \
                'plot_%s_bias_sample_%s.png'%(plot_type, isample+1))
            self.plot_params(plot_name, params=params, plot_type=plot_type)

    def plot_params(self, plot_name, params=None, plot_type='1d'):
        """Make 1d posterior plot for all parameters or those specified through a list params."""

        params = (params or self.params)

        g = gplots.get_subplot_plotter(
            chain_dir=self.chain_dir, 
            analysis_settings=self.analysis_settings)

        mcsamples = self.get_mcsamples(g, params)

        plot_func = {'1d': 'plots_1d', 'triangle': 'triangle_plot'}[plot_type]
        getattr(g, plot_func)([mcsamples], params=params, filled=True)

        for param in params:
            ax = g.get_axes_for_params(param)
            self.add_input_line_for_param(g, ax, param)
            self.add_prior_bands_for_param(g, ax, param)

        g.export(plot_name)
        self.logger.info('Saved plot: {}'.format(plot_name))

    def get_mcsamples(self, g, params):
        samples = g.sampleAnalyser.samplesForRoot(self.roots[0])
        print('Number of points in chain = {}'.format(samples.weights.size))

        p = samples.getParams()
        sample_values = [getattr(p, param) for param in params]

        ranges = self.get_mcsamples_ranges()
        ranges['w'] = [-1, None] #HACK

        MCSamples = getdist.MCSamples(
            samples = sample_values, \
            weights = samples.weights,\
            loglikes = samples.loglikes, \
            names = params,\
            settings = self.analysis_settings, \
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
        default_values = get_bias_params_for_survey_file(\
            self.survey_par_file, fix_to_default=True, include_latex=False)
        return default_values

    def get_priors(self):
        """Get priors applied when running chains"""
        updated_yaml_file = os.path.join(self.chain_dir, self.roots[0] + '.updated.yaml')
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


class Prior():

    def __init__(self, prior_dict):
        self.prior_dict = prior_dict

    def get_center_and_sigma(self):
        if self.is_uniform():
            xmin = self.prior_dict['min'] 
            xmax = self.prior_dict['max'] 
            center = (xmin+xmax) / 2.0
            sigma = (xmax-xmin) / 2.0
        else:
            if self.prior_dict['dist'] == 'norm':
                center = self.prior_dict['loc'] 
                sigma = self.prior_dict['scale'] 
            else:
                raise NotImplementedError
        return center, sigma
    
    def is_uniform(self):
        is_min_given = 'min' in self.prior_dict.keys()
        is_max_given = 'max' in self.prior_dict.keys()
        is_uniform = (is_min_given and is_max_given)
        return is_uniform


if __name__ == '__main__':
    """Example usage: python3 -m analysis.plot ./analysis/inputs/ps_base.yaml"""

    config_file = sys.argv[1]

    args = yaml_load_file(config_file)

    plotter = ChainPlotter(args)
    plotter.plot_1d()

