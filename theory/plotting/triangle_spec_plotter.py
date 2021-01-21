
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from theory.data_vector.triangle_spec import TriangleSpecTheta1Phi12
from theory.data_vector.triangle_spec import TriangleSpec
from theory.plotting.triangle_plotter import TrianglePlotter 

class TriangleSpecPlotter(TrianglePlotter):

    def __init__(self, triangle_spec, plot_dir):
        super().__init__(triangle_spec, plot_dir)

    def make_plots(self):
        self._plot_theta12_and_k()

    def _plot_theta12_and_k(self, plot_type='plot', plot_name=None, plot_type2='semilogy', plot_type3='semilogy'):

        ntri = self._triangle_spec.ntri

        fig = plt.figure(figsize=(12,6))
        gs = gridspec.GridSpec(3, 1, hspace=0, wspace=0)

        allowed_plot_types = ['plot', 'loglog', 'semilogx', 'semilogy']

        if plot_type in allowed_plot_types:

            (ik1, ik2, ik3) = self._triangle_spec.get_ik1_ik2_ik3()
            k1 = self._triangle_spec.k[ik1]
            k2 = self._triangle_spec.k[ik2]
            k3 = self._triangle_spec.k[ik3]
            theta12 = self._triangle_spec.get_theta12(k1, k2, k3)
            theta12_in_deg = theta12/np.pi*180
            
            itriangles = np.arange(ntri)

        # Top panel

            ax = fig.add_subplot(gs[0, 0])

            line, = getattr(ax, plot_type)(itriangles, theta12_in_deg)

            self._add_markers_at_k2_equal_k3(ax, plot_type, itriangles, theta12_in_deg, marker='.')

            ax.axhline(y=90, color='red', ls='-', lw=1)
            ax.axhline(y=120, color='red', ls='--', lw=1)
            ax.axhline(y=30, color='red', ls=':', lw=1)

            ax.legend([r'$\theta_{12}$', r'$k_2 = k_3$', r'$90^{\circ}$', r'$120^{\circ}$', r'$30^{\circ}$'], bbox_to_anchor=(1.01, 0.9))

            ax.set_ylim([0, 180])
            ax.set_xlim([0, ntri-1])

            ax.yaxis.set_ticks([0, 30, 60, 90, 120, 150, 180])

            ax.xaxis.set_visible(False)
            ax.xaxis.set_ticklabels([])
            plt.setp(ax.get_yticklabels()[0], visible=False)    

        # Middle panel

            ax2 = fig.add_subplot(gs[1, 0])

            line1, = getattr(ax2, plot_type2)(itriangles, k1, lw = 2)
            line2, = getattr(ax2, plot_type2)(itriangles, k2, lw = 1.5)
            line3, = getattr(ax2, plot_type2)(itriangles, k3, lw = 1.0, color='k')

            self._add_markers_at_k2_equal_k3(ax2, plot_type, itriangles, k2, marker='.', color=line2.get_color())

            ax2.legend([r'$k_1$', r'$k_2$', r'$k_3$', r'$k_2 = k_3$',], bbox_to_anchor=(1.01, 0.9))

            ax2.set_xlim([0, ntri-1])

            ax2.xaxis.set_visible(False)
            ax2.xaxis.set_ticklabels([])
            plt.setp(ax2.get_yticklabels()[0], visible=False)    

            kwargs_equilateral = {'color': 'grey', 'alpha': 0.8}
            self._add_vertical_lines_at_equilateral(ax2, **kwargs_equilateral)

        # Bottom panel

            ax3 = fig.add_subplot(gs[2, 0])

            line1, = getattr(ax3, plot_type3)(itriangles, k1/k1, lw = 2)
            line2, = getattr(ax3, plot_type3)(itriangles, k2/k1, lw = 1.5)
            line3, = getattr(ax3, plot_type3)(itriangles, k3/k1, lw = 1.0, color='k')

            self._add_markers_at_k2_equal_k3(ax3, plot_type, itriangles, k2/k1, marker='.', color=line2.get_color())

            ax3.legend([r'$1$', r'$k_2/k_1$', r'$k_3/k_1$', r'$k_2 = k_3$',], bbox_to_anchor=(1.01, 0.9))

            ax3.set_xlim([0, ntri-1])

            ax3.set_xlabel('Triangles')

            kwargs_equilateral = {'color': 'grey', 'alpha': 0.8}
            self._add_vertical_lines_at_equilateral(ax3, **kwargs_equilateral)

        if plot_name is None:
            plot_name = os.path.join(self._plot_dir, 'plot_triangle_theta12.pdf')

        plt.savefig(plot_name)
        print('Saved plot: {}'.format(plot_name))
        

class TriangleSpecTheta1Phi12Plotter(TriangleSpecPlotter):

    def __init__(self, triangle_spec_theta1_phi12, plot_dir):
        assert isinstance(triangle_spec_theta1_phi12, TriangleSpecTheta1Phi12)
        self._triangle_spec = triangle_spec_theta1_phi12
        self._plot_dir = plot_dir

    def make_plots(self):
        self._plot_theta12_and_k()
        self._plot_mu()

    def _plot_mu(self, plot_type='plot', plot_name=None):

        nori = self._triangle_spec.nori
        ntri = self._triangle_spec.ntri

        fig = plt.figure(figsize=(12,12))
        gs = gridspec.GridSpec(nori, 1, hspace=0, wspace=0)

        allowed_plot_types = ['plot', 'loglog', 'semilogx', 'semilogy']

        if plot_type in allowed_plot_types:

            for iori in range(nori):

                ax = fig.add_subplot(gs[iori, 0])
                
                if iori%2 == 1:
                    ax.set_facecolor('cornsilk')
                    ax.set_alpha(0.2)

                itriangles = range(ntri)
                (mu1, mu2, mu3) = (self._triangle_spec.mu_array[:,iori,0],\
                    self._triangle_spec.mu_array[:,iori,1],\
                    self._triangle_spec.mu_array[:,iori,2])
                
                line, = getattr(ax, plot_type)(itriangles, mu1)
                line, = getattr(ax, plot_type)(itriangles, mu2)
                line, = getattr(ax, plot_type)(itriangles, mu3)

                if iori == 0:
                    ax.legend(['$\mu_1$', '$\mu_2$', '$\mu_3$'], bbox_to_anchor=(1.01, 0.9))

                ax.set_ylim([-1.5,1.5])
                ax.set_xlim([0, ntri-1])
                if iori == nori-1:
                    ax.set_xlabel('Triangles')
                else:
                    ax.xaxis.set_visible(False)
                    ax.xaxis.set_ticklabels([])

                theta1 = self._triangle_spec.angle_array[iori, 0]
                phi12 = self._triangle_spec.angle_array[iori, 1]
                textstr = r'$\theta_1=%3.1f^{\circ}$'%(theta1/np.pi*180) + '\n' + '$\phi_{12}=%3.1f^{\circ}$'%(phi12/np.pi*180)
                ax.annotate(textstr, xy=(-0.15,0.5), xycoords='axes fraction',\
                    bbox=dict(boxstyle="round", fc="w", lw=1))

                kwargs_equilateral = {'color': 'grey', 'alpha': 0.8}
                self._add_vertical_lines_at_equilateral(ax, **kwargs_equilateral)

        if plot_name is None:
            plot_name = os.path.join(self._plot_dir, 'plot_triangle_orientation_mu.pdf')

        plt.savefig(plot_name)
        print('Saved plot: {}'.format(plot_name))


