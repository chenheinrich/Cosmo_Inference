import numpy as np
import argparse
import yaml
import os
import pickle
import matplotlib.pyplot as plt

from theory.params.cosmo_par import CosmoPar
from theory.params.survey_par import SurveyPar
from theory.data_vector.data_spec import DataSpecPowerSpectrum
from theory.data_vector.data_vector import DataVector, P3D, B3D
from theory.utils import file_tools

def get_data_vec_ps(info):
    cosmo_par_file = info['cosmo_par_file']
    cosmo_par_fid_file = info['cosmo_par_fid_file']
    survey_par_file = info['survey_par_file']
    data_spec_dict = info['PowerSpectrum3D'] 

    cosmo_par = CosmoPar(cosmo_par_file)
    cosmo_par_fid = CosmoPar(cosmo_par_fid_file)

    survey_par = SurveyPar(survey_par_file)
    data_spec = DataSpecPowerSpectrum(survey_par, data_spec_dict)
    #TODO need to clean up DataSpecPowerSpectrum vs DataSpecBispectrum and refactor

    data_vec = P3D(cosmo_par, cosmo_par_fid, survey_par, data_spec)
    
    return data_vec

def get_ps(info):
    data_vec = get_data_vec_ps(info)
    galaxy_ps = data_vec.get('galaxy_ps')
    return galaxy_ps

def get_fn(info):
    file_tools.mkdir_p(info['plot_dir'])
    return os.path.join(info['plot_dir'], info['run_name'] + '.npy')

#TODO to move to a different file
def compare_galaxy_ps(d1, d2):
    if np.allclose(d1, d2):
        print('Passed!')
    else:
        diff = d2 - d1
        frac_diff = diff/d1
        max_diff = np.max(np.abs(diff))
        ind = np.where(np.abs(diff)==max_diff)
        print('ind', ind)
        print(d1[ind], d2[ind])
        print(max_diff/d1[ind])
        max_frac_diff = np.nanmax(np.abs(frac_diff))
        print('max_diff', max_diff)
        print('max_frac_diff', max_frac_diff)


def get_indices(n_tot, n_wanted=5):
    delta = 1 if n_tot <= n_wanted else int(np.floor(
        n_tot / np.float(n_wanted)))
    indices = np.arange(0, n_tot, delta)
    return indices

def plot_galaxy_ps(d1, d2, info):
    survey_par_file = info['survey_par_file']
    data_spec_dict = info['PowerSpectrum3D'] 
    survey_par = SurveyPar(survey_par_file)
    data_spec = DataSpecPowerSpectrum(survey_par, data_spec_dict)

    axis_names = ['ps', 'z', 'k', 'mu']
    yname = 'galaxy_ps'
    plot_type = 'loglog'

    ips_list = get_indices(data_spec.nps, n_wanted=2)
    izs = get_indices(data_spec.nz, n_wanted=2)
    for ips in ips_list:
        for iz in izs:
            axis_names_in = ['k', 'mu']
            yname_in = yname + '_ips%s_iz%s' % (ips, iz)
            data_in = d1[ips, iz, :, :]
            data_in2 = d2[ips, iz, :, :]
            ylatex = 'galaxy power spectrum $P_g(k, \mu)$'
            plot_2D(axis_names_in, data_in, data_spec, ylatex,
                            yname_in, plot_type=plot_type, data2=data_in2)

def plot_2D(*args, **kwargs):
    make_plot_2D_fixed_axis('col', * args, **kwargs)
    make_plot_2D_fixed_axis('row', * args, **kwargs)

def make_plot_2D_fixed_axis(fixed_axis_name, axis_names, data, data_spec, ylatex, yname,
                            plot_type='plot', k=None, z=None, ylim=None, data2=None):
    """
    Make 1D plot of selected rows of the input 2-d numpy array data.
    If data has more than 5 rows, 5 indices equally spaced are selected.

    [More doc needed here on input args.]
    """

    FIXED, VARIED = get_fixed_and_varied_axes(fixed_axis_name)

    shape = data.shape
    indices = get_indices(shape[FIXED], n_wanted=5)

    if fixed_axis_name == 'row':
        y_list = [data[i, :] for i in indices]
        if data2 is not None:
            y_list2 = [data2[i, :] for i in indices]
    elif fixed_axis_name == 'col':
        y_list = [data[:, i] for i in indices]
        if data2 is not None:
            y_list2 = [data2[:, i] for i in indices]

    x = get_xarray(axis_names[FIXED], data_spec)
    legend = ['%s = %.2e' % (axis_names[FIXED], x[i])
                for i in indices]

    plot_1D(axis_names[VARIED], y_list, ylatex, yname, data_spec,
                    legend=legend, plot_type=plot_type, k=k, z=z, 
                    ylim=ylim, y_list2=y_list2)

def get_fixed_and_varied_axes(fixed_axis):
    if fixed_axis == 'row':
        FIXED = 0
        VARIED = 1
    elif fixed_axis == 'col':
        FIXED = 1
        VARIED = 0
    else:
        raise ValueError(
            "_get_fixed_and_varied_axes: fixed_axis can only be: row or col.")
    return FIXED, VARIED

def get_xarray(dim, data_spec, k=None, z=None):
    if dim == 'z':
        x = data_spec.z if z is None else z
    elif dim == 'k':
        x = data_spec.k if k is None else k
    elif dim == 'mu':
        x = data_spec.mu
    elif dim == 'sample':
        x = np.arange(data_spec.nsample)
    elif dim == 'a':
        x = data_spec.z if z is None else z
        x = 1.0 / (1.0 + x)
    else:
        msg = "get_xarray can only take input values for dim: 'z', 'k', 'mu', 'sample' or 'a'."
        raise ValueError(msg)
    return x

def get_xlabel(dim):
    if dim == 'z':
        xlabel = '$z$'
    elif dim == 'k':
        xlabel = '$k$ [1/Mpc]'
    elif dim == 'mu':
        xlabel = '$\mu$'
    elif dim == 'sample':
        xlabel = 'galaxy sample number'
    elif dim == 'a':
        xlabel = '$a$'
    else:
        msg = "get_xarray can only take input values for dim: 'z', 'k', 'mu', 'sample' or 'a'."
        raise ValueError(msg)
    return xlabel

def plot_1D(dimension, y_list, ylatex, yname, data_spec,
                legend='', plot_type='plot', k=None, z=None,
                ylim=None, xlim=None, y_list2=None):

    plot_dir = info['plot_dir']

    allowed_plot_types = ['plot', 'loglog', 'semilogx', 'semilogy']

    x = get_xarray(dimension, data_spec)
    xlabel = get_xlabel(dimension)

    fig, ax = plt.subplots()

    for i, y in enumerate(y_list):
        if plot_type in allowed_plot_types:
            getattr(ax, plot_type)(x, y)
            if y_list2 is not None:
                y2 = y_list2[i]
                getattr(ax, plot_type)(x, y2, ls = '--')
        else:
            msg = "plot_type can only be one of the following: {}".format(
                allowed_plot_types)
            raise ValueError(msg)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylatex)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.legend(legend)

    plot_name = 'plot_%s_vs_%s.png' % (yname, dimension)
    plot_name = os.path.join(plot_dir, plot_name)
    fig.savefig(plot_name)
    print('Saved plot = {}'.format(plot_name))
    plt.close()


if __name__ == '__main__':
    """
    Example usage:
        python3 -m theory.get_ps ./inputs_theory/ps.yaml
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file", type=str, default=None,
        help="path to config file."
    )

    command_line_args = parser.parse_args()

    with open(command_line_args.config_file) as file:
        info = yaml.load(file, Loader=yaml.FullLoader)
    print('info', info)

    ps = get_ps(info)
    fn = get_fn(info)
    file_tools.save_file_npy(fn, ps)

    do_comparison = True #TODO put all this code somewhere else (plotting and testing code)
    if do_comparison is True:
        
        d1 = np.load(fn)

        fid_cosmology_file = './inputs/cosmo_pars/planck2018_fiducial.yaml'
        sim_cosmology_file = './inputs/cosmo_pars/planck2018_fnl_1p0.yaml'

        if info['cosmo_par_file'] == fid_cosmology_file:
            fn2 = './data/ps_base_v27/ref.pickle'
        elif info['cosmo_par_file'] == sim_cosmology_file:
            fn2 = './data/ps_base_v27/sim_data.pickle'
        else:
            raise NotImplementedError

        results = pickle.load(open(fn2, "rb"))
        d2 = results['galaxy_ps']

        compare_galaxy_ps(d1, d2)
        plot_galaxy_ps(d1, d2, info)

    #TODO NEXT: need to add plotting routines and unit tests (from old module, that can be tested now)
    