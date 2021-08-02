import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from lss_theory.data_vector.triangle_spec import TriangleSpec
from lss_theory.utils.file_tools import mkdir_p


class TrianglePlotter():

    def __init__(self, triangle_spec, plot_dir):
        assert isinstance(triangle_spec, TriangleSpec)
        self._triangle_spec = triangle_spec
        self._plot_dir = plot_dir
        mkdir_p(self._plot_dir)

    def _add_markers_at_k2_equal_k3(self, ax, plot_type, x, y, marker='.', **kwargs):
        indices_k2_equal_k3 = self._triangle_spec.indices_k2_equal_k3
        x_k2 = x[indices_k2_equal_k3]
        y_k2 = y[indices_k2_equal_k3]
        getattr(ax, plot_type)(x_k2, y_k2, marker, markersize = 4, **kwargs)

    def _add_vertical_lines_at_equilateral(self, ax, **kwargs):
        indices_equilateral = self._triangle_spec.indices_equilateral
        self._add_vertical_lines_at_xs(ax, indices_equilateral, **kwargs)

    def _add_zero_line(self, ax, **kwargs):
        ax.axhline(y=0, **kwargs)

    def _add_k_labels(self, ax, y_margins=0.3, **kwargs):

        indices_equilateral = self._triangle_spec.indices_equilateral
        k = self._data_spec.k

        ax.margins(y=y_margins)
        y = ax.get_ylim()[0]

        for i in np.arange(0, self._triangle_spec.nk, 2):
            textstr = '%.1e'%k[i]
            if i == 0:
                textstr = r'$k_{eq}=$'+textstr
            x = indices_equilateral[i] + 1
            plt.annotate(textstr, xy=(x,y),  xytext=(0,15), \
                textcoords="offset points", **kwargs)

        for i in np.arange(1, self._triangle_spec.nk, 2):
            textstr = '%.1e'%k[i]
            x = indices_equilateral[i]
            plt.annotate(textstr, xy=(x,y), xytext=(0,6), \
                textcoords="offset points", **kwargs)

    @staticmethod
    def _add_vertical_lines_at_xs(ax, xs, **kwargs):
        for x in xs:
            ax.axvline(x = x, **kwargs)

    @staticmethod
    def _turn_off_xaxis_ticklabels(ax):
        ax.xaxis.set_visible(False)
        ax.xaxis.set_ticklabels([])

    @staticmethod
    def _turn_off_yaxis_first_ticklabel(ax):
        plt.setp(ax.get_yticklabels()[0], visible=False)    

    @staticmethod
    def _set_ylim_clipped(ax, ylim=None, ylim_clip=None):

        if ylim is not None:
            ax.set_ylim(ylim)
            return 
        else:   
            if ylim_clip is None:
                ax.set_ylim(ylim)
                return 
            else:
                ylim = ax.get_ylim()

                if ylim_clip[0] is not None:
                    clipped_ylim_lo = max(ylim[0], ylim_clip[0])
                else:
                    clipped_ylim_lo = ylim[0]

                if ylim_clip[1] is not None:
                    clipped_ylim_hi = min(ylim[1], ylim_clip[1])
                else:
                    clipped_ylim_hi = ylim[1]
                    
                ylim = [clipped_ylim_lo, clipped_ylim_hi]
                ax.set_ylim(ylim)
                return 

