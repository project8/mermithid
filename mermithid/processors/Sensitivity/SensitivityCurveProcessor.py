'''
Calculate sensitivity curve and plot vs. pressure
function.
Author: R. Reimann, C. Claessens
Date:11/17/2020

More description
'''

from __future__ import absolute_import


import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# Numericalunits is a package to handle units and some natural constants
# natural constants
from numericalunits import e, me, c0, eps0, kB, hbar
from numericalunits import meV, eV, keV, MeV, cm, m, ns, s, Hz, kHz, MHz, GHz
from numericalunits import nT, uT, mT, T, mK, K,  C, F, g, W
from numericalunits import hour, year, day
ppm = 1e-6
ppb = 1e-9

# morpho imports
from morpho.utilities import morphologging, reader
from morpho.processors import BaseProcessor
from mermithid.misc.SensitivityFormulas import Sensitivity


logger = morphologging.getLogger(__name__)



__all__ = []
__all__.append(__name__)

class SensitivityCurveProcessor(BaseProcessor):
    '''
    Description
    Args:

    Inputs:

    Output:

    '''
    def InternalConfigure(self, params):
        '''
        Configure
        '''
        # file paths
        self.config_file_path = reader.read_param(params, 'config_file_path', "required")
        self.comparison_config_file_path = reader.read_param(params, 'comparison_config_file_path', '')
        self.plot_path = reader.read_param(params, 'plot_path', "required")


        # labels
        self.main_curve_upper_label = reader.read_param(params, 'main_curve_upper_label', r"molecular"+"\n"+r"$V_\mathrm{eff} = 2\, \mathrm{cm}^3$"+"\n"+r"$\sigma_B = 7\,\mathrm{ppm}$")
        self.main_curve_lower_label = reader.read_param(params, 'main_curve_lower_label', r"$\sigma_B = 1\,\mathrm{ppm}$")
        self.comparison_curve_label = reader.read_param(params, 'comparison_curve_label', r"atomic"+"\n"+r"$V_\mathrm{eff} = 5\, \mathrm{m}^3$"+"\n"+r"$\sigma_B = 0.13\,\mathrm{ppm}$")


        # options
        self.comparison_curve = reader.read_param(params, 'comparison_curve', False)
        self.B_error = reader.read_param(params, 'B_inhomogeneity', 7e-6)
        self.B_error_uncertainty = reader.read_param(params, 'B_inhom_uncertainty', 0.05)


        # plot configurations
        self.figsize = reader.read_param(params, 'figsize', (6,6))
        self.density_range = reader.read_param(params, 'density_range', [1e14,1e21])
        self.ylim = reader.read_param(params, 'y_limits', [1e-2, 1e2])
        self.track_length_axis = reader.read_param(params, 'track_length_axis', True)
        self.atomic_axis = reader.read_param(params, 'atomic_axis', False)
        self.molecular_axis = reader.read_param(params, 'molecular_axis', False)
        self.label_x_position = reader.read_param(params, 'label_x_position', 5e19)
        self.upper_label_y_position = reader.read_param(params, 'upper_label_y_position', 5)
        self.lower_label_y_position = reader.read_param(params, 'lower_label_y_position', 2.3)


        # goals
        self.goals = reader.read_param(params, "goals", {})


        # setup sensitivities
        self.sens_main = Sensitivity(self.config_file_path)
        self.sens_main_is_atomic = self.sens_main.Experiment.atomic

        if self.comparison_curve:
            self.sens_ref = Sensitivity(self.comparison_config_file_path)
            self.sens_ref_is_atomic = self.sens_ref.Experiment.atomic

        # check atomic and molecular
        if self.molecular_axis:
            if not self.sens_main_is_atomic:
                self.molecular_sens = self.sens_main
                logger.info("Main curve is molecular")
            elif not self.sens_ref_is_atomic:
                self.molecular_sens = self.sens_ref
                logger.info("Comparison curve is molecular")
            else:
                raise ValueError("No experiment is configured to be molecular")

        if self.atomic_axis:
            if self.sens_main_is_atomic:
                self.atomic_sens = self.sens_main
                logger.info("Main curve is atomic")
            elif self.sens_ref_is_atomic:
                self.atomic_sens = self.sens_ref
                logger.info("Comparison curve is atomic")
            else:
                raise ValueError("No experiment is configured to be atomic")

        # densities
        self.rhos = np.logspace(np.log10(self.density_range[0]), np.log10(self.density_range[1]), 100)/m**3

        return True



    def InternalRun(self):

        self.create_plot()

        # add second and third x axis for track lengths
        if self.track_length_axis:
            self.add_track_length_axis()

        # add line for comparison using second config
        if self.comparison_curve:
            self.add_comparison_curve(label=self.comparison_curve_label)
            #self.add_arrow(self.sens_main)

        for key, value in self.goals.items():
            logger.info('Adding goal: {}'.format(key))
            self.add_goal(value*eV, key)

        # if B is list plot line for each B
        print(self.B_error)
        if isinstance(self.B_error, list) or isinstance(self.B_error, np.ndarray):
            N = len(self.B_error)
            for a, color in self.range(0, N):
                sig = self.sens_main.BToKeErr(self.sens_main.MagneticField.nominal_field*self.B_error[a], self.sens_main.MagneticField.nominal_field)
                self.sens_main.MagneticField.usefixedvalue = True
                self.sens_main.MagneticField.default_systematic_smearing = sig
                self.sens_main.MagneticField.default_systematic_uncertainty = 0.05*sig
                self.add_sens_line(self.sens_main, color=color)
            self.add_text(self.label_x_position, self.upper_label_y_position, self.main_curve_upper_label)
            self.add_text(self.label_x_position, self.lower_label_y_position, self.main_curve_lower_label)

        else:
            sig = self.sens_main.BToKeErr(self.sens_main.MagneticField.nominal_field*self.B_error[a], self.sens_main.MagneticField.nominal_field)
            self.sens_main.MagneticField.usefixedvalue = True
            self.sens_main.MagneticField.default_systematic_smearing = sig
            self.sens_main.MagneticField.default_systematic_uncertainty = 0.05*sig
            self.add_sens_line(self.sens_main, color='blue')
            self.add_text(self.label_x_position, self.upper_label_y_position, self.main_curve_upper_label)


        # save plot
        self.save(self.plot_path)

        # print number of events
        limit = [self.sens_main.CL90(Experiment={"number_density": rho})/eV for rho in self.rhos]
        self.opt_ref = np.argmin(limit)

        rho_opt = self.rhos[self.opt_ref]


        logger.info('Main curve (veff = {} cm**3, rho = {} /m**3):'.format(self.sens_main.Experiment.v_eff/(cm**3), rho_opt*(m**3)))
        logger.info('CL90 limit: {}'.format(self.sens_main.CL90(Experiment={"number_density": rho_opt})/eV))
        logger.info('T2 in Veff: {}'.format(rho_opt*self.sens_main.Experiment.v_eff))
        logger.info('Total signal: {}'.format(rho_opt*self.sens_main.Experiment.v_eff*
                                                   self.sens_main.Experiment.LiveTime/
                                                   self.sens_main.tau_tritium*2))
        logger.info('Signal in last eV: {}'.format(self.sens_main.last_1ev_fraction*eV**3*
                                                   rho_opt*self.sens_main.Experiment.v_eff*
                                                   self.sens_main.Experiment.LiveTime/
                                                   self.sens_main.tau_tritium*2))

        self.sens_main.print_statistics()
        self.sens_main.print_systematics()

        return True


    def create_plot(self):
        # setup axis
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        ax = self.ax
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(self.rhos[0]*m**3, self.rhos[-1]*m**3)
        ax.set_ylim(self.ylim)
        #ax.grid()

        if self.atomic_axis and self.molecular_axis:
            axis_label = r"(atomic / molecular) number density $\rho\, /\, \mathrm{m}^{-3}$"
        elif self.atomic_axis:
            axis_label = r"(atomic) number density $\rho\, /\, \mathrm{m}^{-3}$"
        elif self.molecular_axis:
            axis_label = r"(molecular) number density $\rho\, /\, \mathrm{m}^{-3}$"
        else:
            axis_label = r"number density $\rho\, /\, \mathrm{m}^{-3}$"

        ax.set_xlabel(axis_label)
        ax.set_ylabel(r"90% CL $m_\beta$ / eV")

    def add_track_length_axis(self):
        if self.atomic_axis:
            ax2 = self.ax.twiny()
            ax2.set_xscale("log")
            ax2.set_xlabel("(atomic) track length / s")
            ax2.set_xlim(self.atomic_sens.track_length(self.rhos[0])/s,
                         self.atomic_sens.track_length(self.rhos[-1])/s)

        if self.molecular_axis:
            ax3 = self.ax.twiny()

            if self.atomic_axis:
                ax3.spines["top"].set_position(("axes", 1.2))
                ax3.set_frame_on(True)
                ax3.patch.set_visible(False)
                for sp in ax3.spines.values():
                    sp.set_visible(False)
                ax3.spines["top"].set_visible(True)

            ax3.set_xscale("log")
            ax3.set_xlabel("(molecular) track length / s")
            ax3.set_xlim(self.molecular_sens.track_length(self.rhos[0])/s,
                         self.molecular_sens.track_length(self.rhos[-1])/s)

        else:
            logger.warning("No track length axis added since neither atomic nor molecular was requested")
        self.fig.tight_layout()

    def add_comparison_curve(self, label, color='k'):
        limit = [self.sens_ref.CL90(Experiment={"number_density": rho})/eV for rho in self.rhos]
        self.opt_ref = np.argmin(limit)

        rho_opt = self.rhos[self.opt_ref]
        logger.info('Ref. curve (veff = {} m**3):'.format(self.sens_ref.Experiment.v_eff/(m**3)))
        logger.info('T in Veff: {}'.format(rho_opt*self.sens_ref.Experiment.v_eff))
        logger.info('Ref. total signal: {}'.format(rho_opt*self.sens_ref.Experiment.v_eff*
                                                   self.sens_ref.Experiment.LiveTime/
                                                   self.sens_ref.tau_tritium))

        self.ax.plot(self.rhos*m**3, limit, color=color)
        self.ax.axvline(self.rhos[self.opt_ref]*m**3, ls=":", color="gray", alpha=0.4)

        #self.ax.axhline(0.04, color="gray", ls="--")
        #self.ax.text(3e14, 0.044, "Phase IV (40 meV)")
        self.ax.text(1.5e14, 0.11, label)

    def add_arrow(self, sens):
        if not hasattr(self, "opt_ref"):
            self.add_comparison_curve()

        def get_relative(val, axis):
            xmin, xmax = self.ax.get_xlim() if axis == "x" else self.ax.get_ylim()
            return (np.log10(val)-np.log10(xmin))/(np.log10(xmax)-np.log10(xmin))

        rho_IV = self.rhos[self.opt_ref]
        track_length_IV = self.sens_ref.track_length(rho_IV)
        track_length_III = self.sens_main.track_length(rho_IV)
        rho_III = rho_IV*track_length_III/track_length_IV
        limit_III  = sens.CL90(Experiment={"number_density": rho_III})

        x_start = get_relative(rho_IV*m**3, "x")
        y_start = get_relative(2*limit_III/eV, "y")
        x_stop = get_relative(rho_III*m**3, "x")
        y_stop = get_relative(limit_III/eV, "y")
        plt.arrow(x_start, y_start, x_stop-x_start, y_stop-y_start,
                      transform = self.ax.transAxes,
                      facecolor = 'black',
                      edgecolor='k',
                      length_includes_head=True,
                      head_width=0.02,
                      head_length=0.03,
                      )

    def add_goal(self, value, label):
        self.ax.axhline(value/eV, color="gray", ls="--")
        self.ax.text(3e18, 0.8*value/eV, label)

    def add_sens_line(self, sens, **kwargs):
        limits = [sens.CL90(Experiment={"number_density": rho})/eV for rho in self.rhos]
        self.ax.plot(self.rhos*m**3, limits, **kwargs)
        logger.info('Minimum limit at {}: {}'.format(self.rhos[np.argmin(limits)]*m**3, np.min(limits)))

    def add_text(self, x, y, text):
        self.ax.text(x, y, text)

    def range(self, start, stop):
        cmap = matplotlib.cm.get_cmap('Spectral')
        norm = matplotlib.colors.Normalize(vmin=start, vmax=stop-1)
        return [(idx, cmap(norm(idx))) for idx in range(start, stop)]

    def save(self, savepath, **kwargs):
        self.fig.tight_layout()
        #keywords = ", ".join(["%s=%s"%(key, value) for key, value in kwargs.items()])
        metadata = {"Author": "p8/mermithid",
                    "Title": "Neutrino mass sensitivity",
                    "Subject":"90% CL upper limit on neutrino mass assuming true mass is zero."
                    }
                    #"Keywords": keywords}
        if savepath is not None:
            self.fig.savefig(savepath.replace(".pdf", ".png"), dpi=300, metadata=metadata)
            self.fig.savefig(savepath.replace(".png", ".pdf"), bbox_inches="tight", metadata=metadata)


