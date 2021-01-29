import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os

from theory.utils import file_tools
from theory.data_vector.data_vector import DataVector, B3D, B3D_RSD
from theory.plotting.triangle_plotter import TrianglePlotter

class BisSNPlotter(TrianglePlotter):
    
    def __init__(self, data_vec, data_spec, \
            data_vec2=None, \
            bis_var=None, \
            plot_dir='./plots/theory/bispectrum/', \
            do_run_checks=True):
        
        super().__init__(data_spec.triangle_spec, plot_dir)

        self._data_vec = data_vec
        self._data_spec = data_spec
        self._data_vec_ref = data_vec2 
        self._bis_var = bis_var
        self._do_run_checks = do_run_checks

    def make_plots(self):
        
        if self._do_run_checks is True:
            self._run_checks()

        nb = self._data_spec.nsample ** 3

        kwargs = {'plot_type': 'semilogy'}
        if isinstance(self._data_vec, B3D_RSD):
            for ib in range(nb):
                self._plot_sn_bis_rsd(ib, **kwargs)  

        #TODO might want to tighten this logic
        elif isinstance(self._data_vec, B3D):
            for ib in range(nb):
                self._plot_galaxy_bis(ib)   


    def _plot_galaxy_bis(self, ib):

        izs = np.arange(0, self._data_spec.nz, 10)

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

    def _plot_sn_bis_rsd(self, ib, **kwargs):

        """Support two kinds of plot:
            1) Difference: data_vec2 - data_vec 
            2) Difference/error: (data_vec2 - data_vec)/bis_var if bis_var is not None
        """

        # plot B (fnl = 1) - B (fnl = 0)
        #TODO eventually (plot B (fnl = 1) - B (fnl = 0)) / error
        
        #fname_ref = self._get_fname_fnl_ref(self._plot_dir)
        #print('Loading bispectrum with fnl = 0 from {}'.format(fname_ref))
        #bis_fnl_ref = os.path.join(fname_ref)
        #y2 = np.load(bis_fnl_ref)

        if self._bis_var is None:
            bis_error = np.ones_like(self._data_vec.get('Bggg_b10_primordial'))
            yname_tag = 'diff_wrt_fnl_0'
        else:
            bis_error = self._bis_var.bis_error[:,:,:,np.newaxis]
            yname_tag = 'diff_over_error_wrt_fnl_0'

        izs = np.arange(0, self._data_spec.nz, 10)
        dimension = 'tri'
        fnl = self._data_vec._cosmo_par.fnl

        diff = self._data_vec.get('Bggg_b10_primordial') - self._data_vec_ref.get('Bggg_b10_primordial')
        diff = diff/bis_error
        y1_list = [diff[ib, iz, :, :] for iz in izs]

        diff = self._data_vec.get('Bggg_b10_gravitational') - self._data_vec_ref.get('Bggg_b10_gravitational')
        diff = diff/bis_error
        y2_list = [diff[ib, iz, :, :] for iz in izs]

        diff = self._data_vec.get('Bggg_b20') - self._data_vec_ref.get('Bggg_b20')
        diff = diff/bis_error
        y3_list = [diff[ib, iz, :, :] for iz in izs]

        diff = self._data_vec.get('galaxy_bis') - self._data_vec_ref.get('galaxy_bis')
        diff = diff/bis_error
        y4_list = [diff[ib, iz, :, :] for iz in izs]

        legend = [ r'$b_1^3$ prim.', r'$b_1^3$ grav.', r'$b_2 b_1^2$', 'total']

        (isample1, isample2, isample3) = self._data_spec.dict_ib_to_isamples['%i'%ib]
        ylatex = ''
        yname = 'galaxy_bis_oriented_ABC_%i_%i_%i_%s'%(isample1, isample2, isample3, yname_tag)
        ylim_clip = None

        if self._bis_var is None:
            ylim = [1e5, 1e12]
        else:
            ylim = [1e-5, 1e5]
        
        title = r'$\Delta B_{g_{%s}g_{%s}g_{%s}}$ ($f_{\rm NL} = %s$ vs 0)'%(isample1, isample2, isample3, fnl)
        xlim = [0, self._data_spec.triangle_spec.ntri]

        line_styles = ['-', '-', '-', '-']

        self._plot_1D_with_orientation(dimension, y1_list, ylatex, yname, self._data_spec, \
            xlim=xlim, plot_dir=self._plot_dir, izs=izs, legend=legend, title=title, \
            ylim=ylim, ylim_clip=ylim_clip, y2_list=y2_list, y3_list=y3_list, y4_list=y4_list, \
            line_styles=line_styles, **kwargs)


    def _plot_1D_with_orientation(self, dimension, y_list, ylatex, yname, data_spec,\
                legend='', plot_type='plot', k=None, z=None, \
                ylim=None, xlim=None,\
                plot_dir='', izs=None, title='',\
                y2_list=None, y3_list=None, y4_list=None, ylim_clip=None, \
                line_styles=['-', '--', ':', '-.']):#TODO need to pass izs idfferently

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

                    line, = self._draw_line(ax, plot_type, x, y[:, iori])

                    if y2_list is not None:
                        y2 = y2_list[i]
                        line, = self._draw_line(ax, plot_type, x, y2[:, iori])

                    if y3_list is not None:
                        y3 = y3_list[i]
                        line, = self._draw_line(ax, plot_type, x, y3[:, iori])

                    if y4_list is not None:
                        y4 = y4_list[i]
                        line, = self._draw_line(ax, plot_type, x, y4[:, iori])

                    self._add_vertical_lines_at_equilateral(ax, **kwargs_equilateral)

                    if iori == 0:
                        ax.set_title(title)
                        ax.legend(legend, bbox_to_anchor=(1.05, 1.0), ncol=2)

                    self._add_zero_line(ax, color='black', ls='--', alpha=0.5)

                    self._set_ylim_clipped(ax, ylim=ylim, ylim_clip=ylim_clip)
                    
                    ax.set_ylabel(ylatex)

                    if xlim is not None:
                        ax.set_xlim(xlim)

                    if iori == (nori-1):
                        ax.set_xlabel(xlabel)
                    else:
                        self._turn_off_xaxis_ticklabels(ax)
                        self._turn_off_yaxis_first_ticklabel(ax)

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
        
            self._add_zero_line(ax, color='black', ls='--', alpha=0.5)

            ax.set_title(title)

            plot_name = 'plot_%s_vs_%s_iz_%i.png' % (yname, dimension, izs[i])
            plot_name = os.path.join(plot_dir, plot_name)

            fig.savefig(plot_name)
            print('Saved plot = {}'.format(plot_name))
            plt.close()


    @staticmethod #TODO need to combine with other bis plotters
    def _draw_line_neg(ax, plot_type, x, y, **kwargs):
        pos_signal = np.copy(y)
        neg_signal = np.copy(y)

        pos_signal[np.where(pos_signal <= 0.0)] = np.nan
        neg_signal[np.where(neg_signal > 0.0)] = np.nan

        if plot_type in ['semilogy', 'loglog']:
            neg_signal = -neg_signal

        line, = getattr(ax, plot_type)(x, pos_signal, **kwargs)
        
        lw = plt.getp(line, 'linewidth')
        color = line.get_color()

        if 'color' in kwargs.keys():
            del kwargs['color']

        return getattr(ax, plot_type)(x, neg_signal, \
            ls='--', color=color, \
            label='_nolegend_', **kwargs)


    def _draw_line(self, ax, plot_type, x, y, **kwargs):
        if plot_type in ['semilogy', 'loglog']:
            return self._draw_line_neg(ax, plot_type, x, y, **kwargs)
        else:
            return getattr(ax, plot_type)(x, y, **kwargs)
