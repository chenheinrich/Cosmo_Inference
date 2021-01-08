import matplotlib.pyplot as plt
import numpy as np
import os

from theory.utils import file_tools

class BisPlotter():
    
    def __init__(self, data_vec, data_spec, d2=None, plot_dir='./plots/theory/bispectrum/'):
        self._d = data_vec.get('galaxy_bis')
        self._d1 = data_vec.get('Bggg_b10')
        self._d2 = data_vec.get('Bggg_b20')
        self._data_spec = data_spec
        self._data_vec = data_vec

        self._plot_dir = plot_dir
        file_tools.mkdir_p(self._plot_dir)

    def make_plots(self):
        
        self._run_checks()

        nb = self._data_spec.nsample ** 3

        for ib in range(nb):
            self._plot_galaxy_bis(ib)       
            

    def _run_checks(self):

        self._check_cyc_perm_of_tracers_are_different(samples=(0,0,1), iz=0)
        self._check_cyc_perm_of_tracers_are_different(samples=(0,0,4), iz=0)
        
        k_peak = self._get_k_at_peak_of_Bggg_b10_equilateral_triangles()
        print('k_peak = {}'.format(k_peak))
        
        print('k = {}'.format(self._data_spec.k))

        self._check_Bggg_b10_equilateral_triangles()
        

    def _check_cyc_perm_of_tracers_are_different(self, samples=(0,0,1), iz=0):
        
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

    def _get_k_at_peak_of_Bggg_b10_equilateral_triangles(self, iz=0):
        
        ib1 = self._data_spec.dict_isamples_to_ib['0_0_0']
        indices_equilateral = self._data_spec.triangle_spec.indices_equilateral
        Bggg_b10 = self._d1[ib1, iz, indices_equilateral]
        
        ind = np.where(Bggg_b10 == np.max(Bggg_b10))[0]
        k_peak = self._data_spec.k[ind]
        
        return k_peak

    def _check_Bggg_b10_equilateral_triangles(self, isample=0, iz=0, imu=0):
        
        expected = self._data_vec.get_expected_Bggg_b10_equilateral_triangles_single_tracer(
            isample=0, iz=0, imu=0)

        ibis = self._data_spec.dict_isamples_to_ib['%i_%i_%i'%(isample, isample, isample)]
        indices_equilateral = self._data_spec.triangle_spec.indices_equilateral
        Bggg_b10_equilateral = self._d1[ibis, iz, indices_equilateral]
        
        print((expected - Bggg_b10_equilateral)/Bggg_b10_equilateral)
        assert np.allclose(Bggg_b10_equilateral, expected)
        

    def _plot_galaxy_bis(self, ib):

        #izs = np.arange(0, self._data_spec.nz, 10)
        izs = [3,4]

        y_list = [self._d1[ib, iz, :] for iz in izs]
        y_list2 = [self._d2[ib, iz, :] for iz in izs]
        y_list3 = [self._d[ib, iz, :] for iz in izs]
        
        dimension = 'tri'
        (isample1, isample2, isample3) = self._data_spec.dict_ib_to_isamples['%i'%ib]
        yname = 'galaxy_bis_ABC_%i_%i_%i'%(isample1, isample2, isample3)
        ylatex = r'$B_{g_{%s}g_{%s}g_{%s}}$'%(isample1, isample2, isample3)

        fnl = self._data_vec._cosmo_par.fnl
        title = r'$B_{g_{%s}g_{%s}g_{%s}}$, $f_{\rm NL} = %s$'%(isample1, isample2, isample3, fnl)

        legend = [r'$b_{10}$ terms', r'$b_{20}$ terms', 'total',\
            r'$k_2 = k_3$', r'$k_1 = k_2 = k_3 = k_{eq}$']

        xlim = [0, self._data_spec.triangle_spec.ntri]

        self._plot_1D(dimension, y_list, ylatex, yname, self._data_spec, \
            xlim=xlim, plot_dir=self._plot_dir, izs=izs, \
            y_list2=y_list2, y_list3=y_list3, legend=legend, title=title)

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

    def _add_markers_at_k2_equal_k3(self, ax, plot_type, x, y, marker='.'):
        indices_k2_equal_k3 = self._data_spec.triangle_spec.indices_k2_equal_k3
        x_k2 = x[indices_k2_equal_k3]
        y_k2 = y[indices_k2_equal_k3]
        getattr(ax, plot_type)(x_k2, y_k2, marker, markersize = 4)

    def _add_vertical_lines_at_equilateral(self, ax, **kwargs):
        indices_equilateral = self._data_spec.triangle_spec.indices_equilateral
        self._add_vertical_lines_at_xs(ax, indices_equilateral, **kwargs)

    def _add_zero_line(self, ax, **kwargs):
        ax.axhline(y=0, **kwargs)

    def _add_k_labels(self, ax, y_margins=0.3, **kwargs):

        indices_equilateral = self._data_spec.triangle_spec.indices_equilateral
        k = self._data_spec.k

        ax.margins(y=y_margins)
        y = ax.get_ylim()[0]

        for i in np.arange(0, self._data_spec.nk, 2):
            textstr = '%.1e'%k[i]
            if i == 0:
                textstr = r'$k_{eq}=$'+textstr
            x = indices_equilateral[i] + 1
            plt.annotate(textstr, xy=(x,y),  xytext=(0,15), \
                textcoords="offset points", **kwargs)

        for i in np.arange(1, self._data_spec.nk, 2):
            textstr = '%.1e'%k[i]
            x = indices_equilateral[i]
            plt.annotate(textstr, xy=(x,y), xytext=(0,6), \
                textcoords="offset points", **kwargs)

    @staticmethod
    def _add_vertical_lines_at_xs(ax, xs, **kwargs):
        for x in xs:
            ax.axvline(x = x, **kwargs)
        

            

        
            

