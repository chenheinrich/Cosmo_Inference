import numpy as numpy
import matplotlib.pyplot as plt 
import os
import sys

from getdist import plots
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
        self.plot_dir = self.args['plot_dir']
        self.nsample = self.args['nsample']
        self.nz = self.args['nz']

        self.params_base = ['fnl', 'logA']
        # TODO might want to put this somewhere central
        self.params_lcdm = ['fnl', 'logA', 'ns', 'theta_MC_100', 'ombh2', 'omch2', 'tau'] 
        self.params_bias = self.get_params_bias(range(self.nsample), range(self.nz))
        self.params_sys = []
        self.params = self.params_lcdm + self.params_bias + self.params_sys

        self.survey_par_file = getattr(PowerSpectrumBase, 'survey_par_file')
        self.bias_default_values = self.get_bias_default_values()
        self.lcdm_sim_values = yaml_load_file(self.data_cosmo_par_file) 
        self.sim_values = dict(self.bias_default_values, **self.lcdm_sim_values) #TODO add sys params in the future

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
            plot_name = os.path.join(self.plot_dir, 'plot_%s_bias_sample_%s.png'%(plot_type, isample+1))
            self.plot_params(plot_name, params=params, plot_type=plot_type)

    def plot_params(self, plot_name, params=None, plot_type='1d'):
        """Make 1d posterior plot for all parameters or those specified through a list params."""

        params = (params or self.params)

        g = plots.get_subplot_plotter(
            chain_dir=self.chain_dir, 
            analysis_settings=self.analysis_settings)
        plot_func = {'1d': 'plots_1d', 'triangle': 'triangle_plot'}[plot_type]
        getattr(g, plot_func)(self.roots, params=params, filled=True)

        for param in params:
            ax = g.get_axes_for_params(param)
            g.add_x_marker(marker=self.sim_values[param], ax=ax, ls='--')

        g.export(plot_name)
        self.logger.info('Saved plot: {}'.format(plot_name))

    def get_bias_default_values(self):
        """Returns a dictionary with bias name and default values"""
        default_values = get_bias_params_for_survey_file(\
            self.survey_par_file, fix_to_default=True, include_latex=False)
        return default_values

if __name__ == '__main__':
    """Example usage: python3 -m analysis.plot ./analysis/inputs/ps_base.yaml"""

    config_file = sys.argv[1]

    args = yaml_load_file(config_file)

    plotter = ChainPlotter(args)
    plotter.plot_triangles()