import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os

from theory.utils import file_tools
from theory.data_vector.multipole_data_vector import BispectrumMultipole
from theory.plotting.triangle_plotter import TrianglePlotter

class BisMultPlotter(TrianglePlotter):
    
    def __init__(self, data_vec, data_spec, d2=None, plot_dir='./plots/theory/bispectrum_multipoles/', do_run_checks=True):
        
        super().__init__(data_spec.triangle_spec, plot_dir)

        self._data_vec = data_vec
        self._data_spec = data_spec

        self._do_run_checks = do_run_checks

        self._d = data_vec.get('galaxy_bis')

    def make_plots(self):
        
        if self._do_run_checks is True:
            self._run_checks()

        nb = self._data_spec.nsample ** 3

        #TODO modified here 
        if isinstance(self._data_vec, BispectrumMultipole):
            #HACK
            #for ib in range(nb):
            for ib in range(1):
                #TODO decide what to do here
                self._plot_galaxy_bis_mult(ib, real_or_imag_or_abs='real')  
                self._plot_galaxy_bis_mult(ib, real_or_imag_or_abs='imag')  
                self._plot_galaxy_bis_mult(ib, real_or_imag_or_abs='abs')  

    def _run_checks(self):
        pass

    #TODO may be able to combine with bis_plotter.
    def _plot_galaxy_bis_mult(self, ib, real_or_imag_or_abs='real'):

        izs = np.arange(0, self._data_spec.nz, 10)
        #TODO changing this
        if real_or_imag_or_abs == 'real':
            y_list = [self._d[ib, iz, :, :].real for iz in izs]
        elif real_or_imag_or_abs == 'imag':
            y_list = [self._d[ib, iz, :, :].imag for iz in izs]
        elif real_or_imag_or_abs == 'abs':
            y_list = [abs(self._d[ib, iz, :, :]) for iz in izs]
        else:
            print('_plot_galaxy_bis_mult(): real_or_imag_or_abs must be real, imag or abs.')
            raise NotImplementedError

        dimension = 'tri'
        (isample1, isample2, isample3) = self._data_spec.dict_ib_to_isamples['%i'%ib]
        yname = 'galaxy_bis_mult_ABC_%i_%i_%i_%s'%(isample1, isample2, isample3, real_or_imag_or_abs)
        ylatex = r'$B_{lm}^{g_{%s}g_{%s}g_{%s}}$'%(isample1, isample2, isample3)

        fnl = self._data_vec._grs_ingredients.get('fnl')
        title = r'$%s(B_{lm}^{g_{%s}g_{%s}g_{%s}})$, $f_{\rm NL} = %s$'%(real_or_imag_or_abs, isample1, isample2, isample3, fnl)

        #TODO modified here
        nlm = self._data_spec.nlm
        
        xlim = [0, self._data_spec.triangle_spec.ntri]

        legend = ['total']
        legend.extend([\
        #r'$k_2 = k_3$', \
        r'$k_1 = k_2 = k_3 = k_{eq}$'])
        self._plot_1D_with_orientation(dimension, y_list, ylatex, yname, self._data_spec, \
            xlim=xlim, plot_dir=self._plot_dir, izs=izs, legend=legend, title=title)

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

        lm_list = self._data_spec.lm_list

        for i, y in enumerate(y_list): # different redshifts, each being a plot
            
            fig = plt.figure(figsize=(12,12.0/8.0*nori), constrained_layout=True)
            gs = gridspec.GridSpec(ncols=1, nrows=nori+1, hspace=0, wspace=0, figure=fig)

            kwargs_equilateral = {'color': 'grey', 'alpha': 0.8}

            if plot_type in allowed_plot_types:
                
                for iori in range(nori):

                    ax = fig.add_subplot(gs[iori, 0])
                    
                    lm = lm_list[iori]

                    #TODO changed
                    if lm[0]%2 == 1:
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

                    #TODO changed
                    textstr = r'$(l,m)=(%s,%s)$'%(lm[0], lm[1])
                    ax.annotate(textstr, xy=(1.02,0.8), xycoords='axes fraction',\
                        bbox=dict(boxstyle="round", fc="w", lw=1))

            else:
                msg = "plot_type can only be one of the following: {}".format(
                    allowed_plot_types)
                raise ValueError(msg)

            plot_name = 'plot_%s_vs_%s_iz_%i.png' % (yname, dimension, izs[i])
            plot_name = os.path.join(plot_dir, plot_name)

            plt.tight_layout = True
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

        

            

