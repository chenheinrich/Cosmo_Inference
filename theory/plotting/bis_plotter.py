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
        
        #self._run_checks()

        nb = self._data_spec.nsample ** 3

        for ib in range(nb):
            self._plot_galaxy_bis(ib)       
            

    def _run_checks(self):
        
        self._check_cyclic_permutation_of_tracers_are_different()
        
        k_peak = self._get_k_at_peak_of_Bggg_b10_equilateral_triangles()
        print('k_peak = {}'.format(k_peak))

        self._check_Bggg_b10_equilateral_triangles()

    def _check_cyclic_permutation_of_tracers_are_different(self):
        iz = 0
        ib1 =  self._data_spec.dict_isamples_to_ib['0_0_1']
        ib2 =  self._data_spec.dict_isamples_to_ib['0_1_0']
        ib3 =  self._data_spec.dict_isamples_to_ib['1_0_0']
        print('B001 =? B010:', np.allclose(self._d[ib1, iz, :], self._d[ib2, iz, :]))
        print('B001 =? B100:', np.allclose(self._d[ib1, iz, :], self._d[ib3, iz, :]))
        diff = self._d[ib1, iz, :] - self._d[ib2, iz, :]
        max_diff = np.max(np.abs(diff))
        print('max abs diff B001 - B010:', max_diff)
        ind = np.where(np.abs(diff) == max_diff)[0]
        print('ind', ind)
        frac_diff = diff[ind]/self._d[ib1, iz, ind]
        print('frac diff at max abs diff for B001 - B010:', frac_diff)

    def _get_k_at_peak_of_Bggg_b10_equilateral_triangles(self, iz=0):
        
        ib1 = self._data_spec.dict_isamples_to_ib['0_0_0']
        indices_equilateral = self._data_spec.triangle_specs.indices_equilateral
        Bggg_b10 = self._d1[ib1, iz, indices_equilateral]
        
        ind = np.where(Bggg_b10 == np.max(Bggg_b10))[0]
        k_peak = self._data_spec.k[ind]
        
        return k_peak

    def _check_Bggg_b10_equilateral_triangles(self, isample=0, iz=0, imu=0):
        
        expected = self._data_vec.get_expected_Bggg_b10_equilateral_triangles_single_tracer(
            isample=0, iz=0, imu=0)

        ibis = self._data_spec.dict_isamples_to_ib['%i_%i_%i'%(isample, isample, isample)]
        indices_equilateral = self._data_spec.triangle_specs.indices_equilateral
        Bggg_b10_equilateral = self._d1[ibis, iz, indices_equilateral]
        
        print((expected - Bggg_b10_equilateral)/Bggg_b10_equilateral)
        assert np.allclose(Bggg_b10_equilateral, expected)
        

    def _plot_galaxy_bis(self, ib):

        izs = np.arange(0, self._data_spec.nz, 10)

        y_list = [self._d1[ib, iz, :] for iz in izs]
        y_list2 = [self._d2[ib, iz, :] for iz in izs]
        y_list3 = [self._d[ib, iz, :] for iz in izs]
        
        dimension = 'tri'
        ylatex = r'$B_{ggg}^{ABC}$ singler tracer'

        (isample1, isample2, isample3) = self._data_spec.dict_ib_to_isamples['%i'%ib]
        yname = 'galaxy_bis_ABC_%i_%i_%i'%(isample1, isample2, isample3)

        legend = [r'$b_{10}$ terms', r'$b_{20}$ terms', 'total',\
            r'$k_2 = k_3$', r'$k_1 = k_2 = k_3$']

        xlim = [0, self._data_spec.triangle_specs.ntri]

        self._plot_1D(dimension, y_list, ylatex, yname, self._data_spec, \
            xlim=xlim, plot_dir=self._plot_dir, izs=izs, \
            y_list2=y_list2, y_list3=y_list3, legend=legend)

    def _plot_1D(self, dimension, y_list, ylatex, yname, data_spec,\
                legend='', plot_type='plot', k=None, z=None, \
                ylim=None, xlim=None, y_list2=None, y_list3=None,\
                plot_dir='', izs=None):#TODO need to pass izs idfferently
        
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

            self._add_vertical_lines_at_equilateral(ax)
        
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylatex)

            if ylim is not None:
                ax.set_ylim(ylim)
            if xlim is not None:
                ax.set_xlim(xlim)

            ax.legend(legend)

            plot_name = 'plot_%s_vs_%s_iz_%i.png' % (yname, dimension, izs[i])
            plot_name = os.path.join(plot_dir, plot_name)

            fig.savefig(plot_name)
            print('Saved plot = {}'.format(plot_name))
            plt.close()

    def _add_markers_at_k2_equal_k3(self, ax, plot_type, x, y, marker='.'):
        indices_k2_equal_k3 = self._data_spec.triangle_specs.indices_k2_equal_k3
        x_k2 = x[indices_k2_equal_k3]
        y_k2 = y[indices_k2_equal_k3]
        getattr(ax, plot_type)(x_k2, y_k2, marker, markersize = 4)

    def _add_vertical_lines_at_equilateral(self, ax):
        indices_equilateral = self._data_spec.triangle_specs.indices_equilateral
        self._add_vertical_lines_at_xs(ax, indices_equilateral)

    @staticmethod
    def _add_vertical_lines_at_xs(ax, xs):
        for x in xs:
            ax.axvline(x = x, color = 'grey')
        

            

        
            

