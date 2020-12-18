import matplotlib.pyplot as plt
import numpy as np
import os

from theory.utils import file_tools

class BisPlotter():
    
    def __init__(self, d1, data_spec, d2=None):
        self._d1 = d1
        self._data_spec = data_spec

        self._plot_dir = './plots/theory/bispectrum/'
        file_tools.mkdir_p(self._plot_dir)

    def make_plots(self):
        
        print('self._d1.shape', self._d1.shape)

        #isamples = np.arange(0, self._data_spec.nsample, 2)
        isample = 0
        izs = np.arange(0, self._data_spec.nz, 5)

        y_list = [self._d1[isample, iz, :] for iz in izs]
        
        print(len(y_list))
        dimension = 'tri'
        ylatex = r'$B_{ggg}$ singler tracer isample=%s'%isample
        yname = 'galaxy_bis'

        xlim = [0, self._data_spec.triangle_specs.ntri]

        self.plot_1D(dimension, y_list, ylatex, yname, self._data_spec, \
            xlim=xlim, plot_dir=self._plot_dir, izs=izs)

    def plot_1D(self, dimension, y_list, ylatex, yname, data_spec,
                legend='', plot_type='plot', k=None, z=None,
                ylim=None, xlim=None, y_list2=None, plot_dir='', 
                izs=None):#TODO need to pass izs idfferently
        
        allowed_plot_types = ['plot', 'loglog', 'semilogx', 'semilogy']

        x = np.arange(y_list[0].size) #TODO to refine
        xlabel = 'Triangles'

        for i, y in enumerate(y_list):

            fig, ax = plt.subplots(figsize=(18,6))

            if plot_type in allowed_plot_types:
                getattr(ax, plot_type)(x, y, marker='.', markersize=4)
                self._add_markers_at_k2_equal_k3(ax, plot_type, x, y, marker='.')
                if y_list2 is not None:
                    y2 = y_list2[i]
                    getattr(ax, plot_type)(x, y2, ls = '--')
            else:
                msg = "plot_type can only be one of the following: {}".format(
                    allowed_plot_types)
                raise ValueError(msg)

            self.add_vertical_lines_at_k(ax)
        
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylatex)

            if ylim is not None:
                ax.set_ylim(ylim)
            if xlim is not None:
                ax.set_xlim(xlim)

            ax.legend(legend)

            plot_name = 'plot_%s_vs_%s_isample_0_iz_%i.png' % (yname, dimension, izs[i])
            plot_name = os.path.join(plot_dir, plot_name)

            fig.savefig(plot_name)
            print('Saved plot = {}'.format(plot_name))
            plt.close()

    def _add_markers_at_k2_equal_k3(self, ax, plot_type, x, y, marker='.'):
        indices_k2_equal_k3 = self._data_spec.triangle_specs.indices_k2_equal_k3
        x_k2 = x[indices_k2_equal_k3]
        y_k2 = y[indices_k2_equal_k3]
        getattr(ax, plot_type)(x_k2, y_k2, marker, markersize = 4)

    def add_vertical_lines_at_k(self, ax):
        indices_equilateral = self._data_spec.triangle_specs.indices_equilateral
        print(indices_equilateral)
        self.add_vertical_lines_at_xs(ax, indices_equilateral)

    @staticmethod
    def add_vertical_lines_at_xs(ax, xs):
        for x in xs:
            ax.axvline(x = x, color = 'grey')
        

            

        
            

