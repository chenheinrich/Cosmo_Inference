import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os

from lss_theory.utils import file_tools
from lss_theory.data_vector.data_vector import DataVector, Bispectrum3DBase, Bispectrum3DRSD
from lss_theory.plotting.triangle_plotter import TrianglePlotter


class BisPlotter(TrianglePlotter):
    
    def __init__(self, data_vec, data_spec, d2=None, plot_dir='./plots/theory/bispectrum/', do_run_checks=True):
    
        super().__init__(data_spec.triangle_spec, plot_dir)

        self._data_vec = data_vec
        self._data_spec = data_spec

        self._do_run_checks = do_run_checks

        if self._do_run_checks is True:
            self._run_checks()
    
    def make_plots(self):

        nb = self._data_spec.nsample ** 3

        for ib in range(nb):
            self._plot_galaxy_bis(ib)  

    def _run_checks(self):
        pass

    def _plot_galaxy_bis(self, ib):
        raise NotImplementedError
    
    def _plot_1D_with_orientation(self, dimension, y_list, ylatex, yname, data_spec,\
                legend='', plot_type='plot', k=None, z=None, \
                ylim=None, xlim=None,
                plot_dir='', izs=None, title='',
                y2_list=None, y3_list=None, ylim_clip=None):#TODO need to pass izs idfferently

        allowed_plot_types = ['plot', 'loglog', 'semilogx', 'semilogy']

        ntri = y_list[0].shape[0]
        nori = y_list[0].shape[1]

        x = np.arange(ntri) 
        xlabel = 'Triangles'

        for i, y in enumerate(y_list): # different redshifts, each being a plot
            
            fig = plt.figure(figsize=(12,12), constrained_layout=True)
            gs = gridspec.GridSpec(ncols=1, nrows=nori+1, hspace=0, wspace=0, figure=fig)

            kwargs_equilateral = {'color': 'grey', 'alpha': 0.8}

            if plot_type in allowed_plot_types:
                
                for iori in range(nori):

                    ax = fig.add_subplot(gs[iori, 0])

                    if iori%2 == 1:
                        ax.set_facecolor('cornsilk')
                        ax.set_alpha(0.2)

                    line_style = '-'
                    line, = getattr(ax, plot_type)(x, y[:, iori], ls=line_style, marker='.', markersize=4)

                    if y2_list is not None:
                        y2 = y2_list[i]
                        line, = getattr(ax, plot_type)(x, y2[:, iori], ls='--', marker='.', markersize=4)

                    if y3_list is not None:
                        y3 = y3_list[i]
                        line, = getattr(ax, plot_type)(x, y3[:, iori], ls=':', marker='.', markersize=4)

                    self._add_vertical_lines_at_equilateral(ax, **kwargs_equilateral)

                    if iori == 0:
                        ax.set_title(title)
                        ax.legend(legend, bbox_to_anchor=(1.01, 0.5))

                    self._add_zero_line(ax, color='black', ls='--', alpha=0.5)

                    self._set_ylim_clipped(ax, ylim=ylim, ylim_clip=ylim_clip)
                    
                    ax.set_ylabel(ylatex)
                    
                    self._turn_off_xaxis_ticklabels(ax)
                    self._turn_off_yaxis_first_ticklabel(ax)

                    if xlim is not None:
                        ax.set_xlim(xlim)

            else:
                msg = "plot_type can only be one of the following: {}".format(
                    allowed_plot_types)
                raise ValueError(msg)

            plot_name = 'plot_%s_vs_%s_iz_%i.png' % (yname, dimension, izs[i])
            plot_name = os.path.join(plot_dir, plot_name)

            fig.savefig(plot_name)
            print('Saved plot = {}'.format(plot_name))
            plt.close()

    def _plot_1D(self, dimension, y_list, ylatex, yname, data_spec,\
                legend='', plot_type='plot', k=None, z=None, \
                ylim=None, xlim=None, y_list2=None, y_list3=None,\
                plot_dir='', izs=None, title=''):#TODO need to pass izs idfferently
        
        allowed_plot_types = ['plot', 'loglog', 'semilogx', 'semilogy']

        x = np.arange(y_list[0].size) #TODO to refine
        xlabel = 'Triangles'

        for i, y in enumerate(y_list):

            fig, ax = plt.subplots(figsize=(18,6))

            if plot_type in allowed_plot_types:
                getattr(ax, plot_type)(x, y, marker='.', markersize=4)
                if y_list2 is not None:
                    y2 = y_list2[i]
                    getattr(ax, plot_type)(x, y2, ls = '--')
                if y_list3 is not None:
                    y3 = y_list3[i]
                    getattr(ax, plot_type)(x, y3, ls = ':')
                self._add_markers_at_k2_equal_k3(ax, plot_type, x, y, marker='.')
            else:
                msg = "plot_type can only be one of the following: {}".format(
                    allowed_plot_types)
                raise ValueError(msg)

            if ylim is not None:
                ax.set_ylim(ylim)
            if xlim is not None:
                ax.set_xlim(xlim)

            kwargs_equilateral = {'color': 'grey', 'alpha': 0.8}
            self._add_vertical_lines_at_equilateral(ax, **kwargs_equilateral)
            self._add_k_labels(ax, **kwargs_equilateral)

            ax.legend(legend)
        
            #kwargs_zero_line = {'color': 'black', 'ls': '--', 'alpha': 0.5}
            self._add_zero_line(ax, color='black', ls='--', alpha=0.5)

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylatex)
            ax.set_title(title)

            plot_name = 'plot_%s_vs_%s_iz_%i.png' % (yname, dimension, izs[i])
            plot_name = os.path.join(plot_dir, plot_name)

            fig.savefig(plot_name)
            print('Saved plot = {}'.format(plot_name))
            plt.close()

class Bispectrum3DBasePlotter(BisPlotter):
    
    def __init__(self, data_vec, data_spec, d2=None, plot_dir='./plots/theory/bispectrum/', do_run_checks=True):
        
        super().__init__(data_vec, data_spec, d2=d2, plot_dir=plot_dir, do_run_checks=do_run_checks)

        self._setup_d()

    def _setup_d(self):
        self._d = self._data_vec.get('galaxy_bis')
        self._d1 = self._data_vec.get('Bggg_b10')
        self._d2 = self._data_vec.get('Bggg_b20')

        self._d1_primordial = self._data_vec.get('Bggg_b10_primordial')
        self._d1_gravitational = self._data_vec.get('Bggg_b10_gravitational')

    def _plot_galaxy_bis(self, ib):

        izs = np.arange(0, self._data_spec.nz, 10)

        y_list = [self._d1[ib, iz, :] for iz in izs]
        y_list2 = [self._d2[ib, iz, :] for iz in izs]
        y_list3 = [self._d[ib, iz, :] for iz in izs]
        
        dimension = 'tri'
        (isample1, isample2, isample3) = self._data_spec.dict_ib_to_isamples['%i'%ib]
        yname = 'galaxy_bis_ABC_%i_%i_%i'%(isample1, isample2, isample3)
        ylatex = r'$B_{g_{%s}g_{%s}g_{%s}}$'%(isample1, isample2, isample3)

        fnl = self._data_vec._grs_ingredients.get('fnl')
        
        title = r'$B_{g_{%s}g_{%s}g_{%s}}$, $f_{\rm NL} = %s$'%(isample1, isample2, isample3, fnl)

        legend = [r'$b_{10}$ terms', r'$b_{20}$ terms', 'total',\
            r'$k_2 = k_3$', r'$k_1 = k_2 = k_3 = k_{eq}$']

        xlim = [0, self._data_spec.triangle_spec.ntri]

        self._plot_1D(dimension, y_list, ylatex, yname, self._data_spec, \
            xlim=xlim, plot_dir=self._plot_dir, izs=izs, \
            y_list2=y_list2, y_list3=y_list3, legend=legend, title=title)


class Bispectrum3DRSDPlotter(BisPlotter):

    def __init__(self, data_vec, data_spec, d2=None, plot_dir='./plots/theory/bispectrum/', do_run_checks=True):
        
        super().__init__(data_vec, data_spec, d2=d2, plot_dir=plot_dir, do_run_checks=do_run_checks)

        self._setup_d()

    def _setup_d(self):
        self._d = self._data_vec.get('galaxy_bis')
        self._d1 = self._data_vec.get('Bggg_b10')
        self._d2 = self._data_vec.get('Bggg_b20')

        self._d1_primordial = self._data_vec.get('Bggg_b10_primordial')
        self._d1_gravitational = self._data_vec.get('Bggg_b10_gravitational')

    def _plot_galaxy_bis(self, ib):

        izs = np.arange(0, self._data_spec.nz, 10)
        y_list = [self._d[ib, iz, :, :] for iz in izs]

        dimension = 'tri'
        (isample1, isample2, isample3) = self._data_spec.dict_ib_to_isamples['%i'%ib]
        yname = 'galaxy_bis_oriented_ABC_%i_%i_%i'%(isample1, isample2, isample3)
        ylatex = r'$B_{g_{%s}g_{%s}g_{%s}}$'%(isample1, isample2, isample3)

        fnl = self._data_vec._grs_ingredients.get('fnl')
        title = r'$B_{g_{%s}g_{%s}g_{%s}}$, $f_{\rm NL} = %s$'%(isample1, isample2, isample3, fnl)

        nori = self._data_spec._triangle_spec.nori
        
        xlim = [0, self._data_spec.triangle_spec.ntri]

        #TODO set this flag somewhere else
        do_plot_primordial_vs_gravitational_bispectrum = True

        if do_plot_primordial_vs_gravitational_bispectrum is True:

            y2_list = [self._d1_primordial[ib, iz, :, :] for iz in izs]
            y3_list = [self._d1_gravitational[ib, iz, :, :] for iz in izs]

            legend = ['total', r'$b_1^3$ prim.', r'$b_1^3$ grav.']
            legend.extend([\
            #r'$k_2 = k_3$', \
            r'$k_1 = k_2 = k_3 = k_{eq}$'])

            self._plot_1D_with_orientation(dimension, y_list, ylatex, yname, self._data_spec, \
                xlim=xlim, plot_dir=self._plot_dir, izs=izs, legend=legend, title=title, \
                y2_list=y2_list, y3_list=y3_list)

        else:
            legend = ['total']
            legend.extend([\
            #r'$k_2 = k_3$', \
            r'$k_1 = k_2 = k_3 = k_{eq}$'])
            self._plot_1D_with_orientation(dimension, y_list, ylatex, yname, self._data_spec, \
                xlim=xlim, plot_dir=self._plot_dir, izs=izs, legend=legend, title=title)

        #Plot frac difference wrt mu's = 0
        do_plot_frac_diff = False
        
        if do_plot_frac_diff is True:
            fname_set_mu_to_zero = self._get_fname_set_mu_to_zero(self._plot_dir)
            print('Loading bispectrum with set_mu_to_zero from {}'.format(fname_set_mu_to_zero))
            bis_set_mu_to_zero = os.path.join(fname_set_mu_to_zero)
            #bis_set_mu_to_zero = './plots/theory/bispectrum_oriented_theta1_phi12_2_4/set_mu_to_zero/fnl_0/multi_tracer_bis.npy'
            y2 = np.load(bis_set_mu_to_zero)
            frac_diff = (self._d - y2)/y2
            y2_list = [frac_diff[ib, iz, :, :] for iz in izs]

            ylatex = ''
            yname = 'galaxy_bis_oriented_ABC_%i_%i_%i_frac_diff_wrt_mu_set_to_zero'%(isample1, isample2, isample3)
            ylim_clip = [-50, 50]
            title = r'Fractional Difference of $B_{g_{%s}g_{%s}g_{%s}}$ wrt $\mu_i = 0$ ($f_{\rm NL} = %s)$'%(isample1, isample2, isample3, fnl)

            self._plot_1D_with_orientation(dimension, y2_list, ylatex, yname, self._data_spec, \
                xlim=xlim, plot_dir=self._plot_dir, izs=izs, legend=legend, title=title, 
                ylim_clip=ylim_clip)

    @staticmethod
    def _get_fname_set_mu_to_zero(plot_dir):
        string = plot_dir
        ind_fnl = string.find('fnl')
        ind_replace = string[ind_fnl:].find('/')
        string_to_replace = string[(ind_fnl + ind_replace):]
        print(string_to_replace)
        fname_set_mu_to_zero = string.replace(string_to_replace, '/set_mu_to_zero/multi_tracer_bis.npy')
        return fname_set_mu_to_zero
        
    def _run_checks(self): #TODO are these for bis_base or bis_rsd?

        self._check_cyc_perm_of_tracers_are_different(samples=(0,0,1), iz=0)
        self._check_cyc_perm_of_tracers_are_different(samples=(0,0,4), iz=0)
        
        k_peak = self._get_k_at_peak_of_Bggg_b10_equilateral_triangles()
        print('k_peak = {}'.format(k_peak))
        
        print('k = {}'.format(self._data_spec.k))

        self._check_Bggg_b10_equilateral_triangles()

    def _check_cyc_perm_of_tracers_are_different(self, samples=(0,0,1), iz=0): #TODO are these for bis_base or bis_rsd?
        
        print('Checking cyclic permutations of bispectrum with samples {}'.format(samples))
        perm1 = '%s_%s_%s'%(samples[0], samples[1], samples[2])
        perm2 = '%s_%s_%s'%(samples[1], samples[2], samples[0])
        perm3 = '%s_%s_%s'%(samples[2], samples[0], samples[1])

        ib1 =  self._data_spec.dict_isamples_to_ib[perm1]
        ib2 =  self._data_spec.dict_isamples_to_ib[perm2]
        ib3 =  self._data_spec.dict_isamples_to_ib[perm3]

        print('B %s =? B %s:'%(perm1, perm2), np.allclose(self._d[ib1, iz, :], self._d[ib2, iz, :]))
        print('B %s =? B %s:'%(perm1, perm3), np.allclose(self._d[ib1, iz, :], self._d[ib3, iz, :]))
        
        diff = self._d[ib1, iz, :] - self._d[ib2, iz, :]
        max_diff = np.max(np.abs(diff))
        print('max abs diff B %s - B %s:'%(perm1, perm2), max_diff)
        ind = np.where(np.abs(diff) == max_diff)[0]
        print('ind', ind)
        frac_diff = diff[ind]/self._d[ib1, iz, ind]
        print('frac diff at max abs diff for B %s - B %s:'%(perm1, perm2), frac_diff)
        assert np.abs(frac_diff) > 1e-10

        diff = self._d[ib1, iz, :] - self._d[ib3, iz, :]
        max_diff = np.max(np.abs(diff))
        print('max abs diff B %s - B %s:'%(perm1, perm3), max_diff)
        ind = np.where(np.abs(diff) == max_diff)[0]
        print('ind', ind)
        frac_diff = diff[ind]/self._d[ib1, iz, ind]
        print('frac diff at max abs diff for B %s - B %s:'%(perm1, perm3), frac_diff)
        assert np.abs(frac_diff) > 1e-10

        print('')

    def _get_k_at_peak_of_Bggg_b10_equilateral_triangles(self, iz=0): #TODO are these for bis_base or bis_rsd?
        
        ib1 = self._data_spec.dict_isamples_to_ib['0_0_0']
        indices_equilateral = self._data_spec.triangle_spec.indices_equilateral
        Bggg_b10 = self._d1[ib1, iz, indices_equilateral]
        
        ind = np.where(Bggg_b10 == np.max(Bggg_b10))[0]
        k_peak = self._data_spec.k[ind]
        
        return k_peak

    def _check_Bggg_b10_equilateral_triangles(self, isample=0, iz=0, imu=0): #TODO are these for bis_base or bis_rsd?
        
        expected = self._data_vec.get_expected_Bggg_b10_equilateral_triangles_single_tracer(
            isample=0, iz=0, imu=0)

        ibis = self._data_spec.dict_isamples_to_ib['%i_%i_%i'%(isample, isample, isample)]
        indices_equilateral = self._data_spec.triangle_spec.indices_equilateral
        Bggg_b10_equilateral = self._d1[ibis, iz, indices_equilateral]
        
        print((expected - Bggg_b10_equilateral)/Bggg_b10_equilateral)
        assert np.allclose(Bggg_b10_equilateral, expected)
        

class Bispectrum3DRSDDerivativePlotter(BisPlotter):
    
    def __init__(self, deriv_converged, data_spec, d2=None, plot_dir='./plots/theory/bispectrum/', do_run_checks=True):
        
        """
        Args:
            deriv_converged: An instance of the DerivativeConvergence class, 
                we can access converged derivative and metadata via
                deriv_converged.data and metadata.
            data_spec: An instance of Bispectrum3DRSDSpec class, 
                where we can get specifications for Bispectrum3DRSD such as
                nsample, nz, etc. via data_spec.nsample, data_spec.nz, etc.
            d2 (optinoal): An instance of the same class deriv_converged, 
                if want to compare the two, not supported at the moment.
            plot_dir: A string for the directory to save plots
            do_run_checks: A boolean for whether to run checks.
        """

        super().__init__(deriv_converged, data_spec, plot_dir=plot_dir, do_run_checks=do_run_checks)

        self._deriv_converged = self._data_vec

        self._setup_d()

        self._metadata = self._deriv_converged.metadata
        self._params_list = self._metadata['derivatives']['params']

        self._cosmo = self._deriv_converged.cosmo
        self._fnl = self._cosmo.fnl

    def make_plots(self):

        nb = self._data_spec.nsample ** 3

        for ib in range(nb):
            self._plot_galaxy_bis(ib)  

    def _setup_d(self):

        self._d = self._data_vec.data

    def _plot_galaxy_bis(self, ib):

        #HACK during debugging
        #izs = np.arange(0, self._data_spec.nz, 10)
        izs = np.array([0])

        for ip, param in enumerate(self._params_list):

            y_list = [self._d[ip, ib, iz, :, :] for iz in izs]

            dimension = 'tri'
            (isample1, isample2, isample3) = self._data_spec.dict_ib_to_isamples['%i'%ib]
            
            yname = 'b3d_rsd_derivative_param_%s_ABC_%i_%i_%i'%(
                param, isample1, isample2, isample3)
            ylatex = r'$\partial B_{g_{%s}g_{%s}g_{%s}}}/\partial (%s)$'%(
                isample1, isample2, isample3, param)

            fnl = self._fnl 
            title = r'$\partial B_{g_{%s}g_{%s}g_{%s}}/ \partial (%s)$, $f_{\rm NL} = %s$'%(\
                isample1, isample2, isample3, param, fnl)

            xlim = [0, self._data_spec.triangle_spec.ntri]

            legend = ['total']
            legend.extend([\
            #r'$k_2 = k_3$', \
            r'$k_1 = k_2 = k_3 = k_{eq}$'])
            self._plot_1D_with_orientation(dimension, y_list, ylatex, yname, self._data_spec, \
                xlim=xlim, plot_dir=self._plot_dir, izs=izs, legend=legend, title=title)

    # TODO plotting out derivatives:
    # order of magnitude seems too big, this may be where problem is from (not from covariance)
    # check what order of magnitude is expected?