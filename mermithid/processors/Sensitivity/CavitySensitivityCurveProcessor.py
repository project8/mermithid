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
from numericalunits import meV, eV, keV, MeV, cm, m
from numericalunits import nT, uT, mT, T, mK, K,  C, F, g, W
from numericalunits import hour, year, day, ms, ns, s, Hz, kHz, MHz, GHz
ppm = 1e-6
ppb = 1e-9

# morpho imports
from morpho.utilities import morphologging, reader
from morpho.processors import BaseProcessor
from mermithid.misc.SensitivityFormulas import Sensitivity
from mermithid.misc.SensitivityCavityFormulas import CavitySensitivity


logger = morphologging.getLogger(__name__)



__all__ = []
__all__.append(__name__)

class CavitySensitivityCurveProcessor(BaseProcessor):
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
        self.sigmae_theta_r = reader.read_param(params, 'sigmae_theta_r', np.array([0.03])) #eV


        # main plot configurations
        self.figsize = reader.read_param(params, 'figsize', (6,6))
        self.density_range = reader.read_param(params, 'density_range', [1e14,1e21])
        self.efficiency_range = reader.read_param(params, 'efficiency_range', [0.001,0.1])
        self.year_range = reader.read_param(params, "years_range", [0.1, 10])
        self.density_axis = reader.read_param(params, "density_axis", True)
        self.ylim = reader.read_param(params, 'y_limits', [1e-2, 1e2])
        self.track_length_axis = reader.read_param(params, 'track_length_axis', True)
        self.atomic_axis = reader.read_param(params, 'atomic_axis', False)
        self.molecular_axis = reader.read_param(params, 'molecular_axis', False)
        self.label_x_position = reader.read_param(params, 'label_x_position', 5e19)
        self.upper_label_y_position = reader.read_param(params, 'upper_label_y_position', 5)
        self.lower_label_y_position = reader.read_param(params, 'lower_label_y_position', 2.3)
        self.goal_x_pos = reader.read_param(params, "goals_x_position", 1e14)
        
        # other plot
        self.make_key_parameter_plots = reader.read_param(params, 'plot_key_parameters', False)
        
        if self.density_axis:
            self.add_sens_line = self.add_density_sens_line
            logger.info("Doing density lines")
        else:
            self.add_sens_line = self.add_exposure_sens_line
            logger.info("Doing exposure lines")


        # goals
        self.goals = reader.read_param(params, "goals", {})


        # setup sensitivities
        self.cavity = reader.read_param(params, 'cavity', True)
        
        if self.cavity:
            self.sens_main = CavitySensitivity(self.config_file_path)
        else:
            self.sens_main = Sensitivity(self.config_file_path)
        self.sens_main_is_atomic = self.sens_main.Experiment.atomic

        if self.comparison_curve:
            if self.cavity:
                self.sens_ref = CavitySensitivity(self.comparison_config_file_path)
            else:
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
        self.effs = np.logspace(np.log10(self.efficiency_range[0]), np.log10(self.efficiency_range[1]), 10)
        self.years = np.logspace(np.log10(self.year_range[0]), np.log10(self.year_range[1]), 10)*year
        
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
            
        # first optimize density
        limit = [self.sens_main.CL90(Experiment={"number_density": rho})/eV for rho in self.rhos]
        opt_ref = np.argmin(limit)
        rho_opt = self.rhos[opt_ref]
        self.sens_main.Experiment.number_density = rho_opt

        # if B is list plot line for each B
        if isinstance(self.sigmae_theta_r, list) or isinstance(self.sigmae_theta_r, np.ndarray):
            N = len(self.sigmae_theta_r)
            for a, color in self.range(0, N):
                #sig = self.sens_main.BToKeErr(self.sens_main.MagneticField.nominal_field*self.B_error[a], self.sens_main.MagneticField.nominal_field)
                #self.sens_main.MagneticField.usefixedvalue = True
                #self.sens_main.MagneticField.default_systematic_smearing = sig
                #self.sens_main.MagneticField.default_systematic_uncertainty = 0.05*sig
                self.sens_main.MagneticField.sigmae_r = self.sigmae_theta_r[a] * eV
                self.sens_main.MagneticField.sigmae_theta = 0 * eV
                self.add_sens_line(self.sens_main, color=color)
                #print("sigmae_theta_r:", self.sens_main.MagneticField.sigmae_r/eV)
                self.sens_main.print_systematics()
            self.add_text(self.label_x_position, self.upper_label_y_position, self.main_curve_upper_label, color="darkblue")
            self.add_text(self.label_x_position, self.lower_label_y_position, self.main_curve_lower_label, color="darkred")

        else:
            #sig = self.sens_main.BToKeErr(self.sens_main.MagneticField.nominal_field*self.B_error[a], self.sens_main.MagneticField.nominal_field)
            #self.sens_main.MagneticField.usefixedvalue = True
            #self.sens_main.MagneticField.default_systematic_smearing = sig
            #self.sens_main.MagneticField.default_systematic_uncertainty = 0.05*sig
            self.sens_main.MagneticField.sigmaer = self.sigmae_theta_r * eV
            self.sens_main.MagneticField.sigmae_theta = 0 * eV
            self.add_sens_line(self.sens_main, color='blue')
            self.add_text(self.label_x_position, self.upper_label_y_position, self.main_curve_upper_label)


        # save plot
        self.save(self.plot_path)
        
        
        # PRINT OPTIMUM RESULTS

        # print number of events
        # for minimum field smearing
        #sig = self.sens_main.BToKeErr(self.sens_main.MagneticField.nominal_field*np.min(self.B_error), self.sens_main.MagneticField.nominal_field)
        #self.sens_main.MagneticField.usefixedvalue = True
        #self.sens_main.MagneticField.default_systematic_smearing = sig
        #self.sens_main.MagneticField.default_systematic_uncertainty = 0.05*sig
                
        if isinstance(self.sigmae_theta_r, list) or isinstance(self.sigmae_theta_r, np.ndarray):
            self.sens_main.MagneticField.sigmae_r = self.sigmae_theta_r[0] * eV
            self.sens_main.MagneticField.sigmae_theta = 0 * eV
        else:
            self.sens_main.MagneticField.sigmaer = self.sigmae_theta_r * eV
            self.sens_main.MagneticField.sigmae_theta = 0 * eV

        logger.info('Main curve:')
        logger.info('veff = {} m**3, rho = {} /m**3:'.format(self.sens_main.effective_volume/(m**3), rho_opt*(m**3)))
        logger.info('Larmor power = {} W, Hanneke power = {} W'.format(self.sens_main.larmor_power/W, self.sens_main.signal_power/W))
        logger.info('Hanneke / Larmor power = {}'.format(self.sens_main.signal_power/self.sens_main.larmor_power))
        logger.info('CL90 limit: {}'.format(self.sens_main.CL90(Experiment={"number_density": rho_opt})/eV))
        logger.info('T2 in Veff: {}'.format(rho_opt*self.sens_main.effective_volume))
        logger.info('Total signal: {}'.format(rho_opt*self.sens_main.effective_volume*
                                                   self.sens_main.Experiment.LiveTime/
                                                   self.sens_main.tau_tritium*2))
        logger.info('Signal in last eV: {}'.format(self.sens_main.last_1ev_fraction*eV**3*
                                                   rho_opt*self.sens_main.effective_volume*
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
        if self.density_axis:
            logger.info("Adding density axis")
            ax.set_xlim(self.rhos[0]*m**3, self.rhos[-1]*m**3)

            if self.atomic_axis and self.molecular_axis:
                axis_label = r"(Atomic / molecular) number density $\rho\, \, (\mathrm{m}^{-3})$"
            elif self.atomic_axis:
                axis_label = r"(Atomic) number density $\rho\, \, (\mathrm{m}^{-3})$"
            elif self.molecular_axis:
                axis_label = r"(Molecular) number density $\rho\, \, (\mathrm{m}^{-3})$"
            else:
                axis_label = r"Number density $\rho\, \, (\mathrm{m}^{-3})$"

            
            
        else:
            logger.info("Adding efficiency axis")
            ax.set_xlim(self.effs[0], self.effs[-1])
            axis_label = r"Efficiency"
            
        ax.set_xlabel(axis_label)
        ax.set_ylim(self.ylim)
        ax.set_ylabel(r"90% CL $m_\beta$ (eV)")
        
        if self.make_key_parameter_plots:
            self.kp_fig, self.kp_ax = plt.subplots(1,2, figsize=(10,5))
            self.kp_ax[0].set_ylabel('Resolution (meV)')
            self.kp_ax[1].set_ylabel('Track analysis length (ms)')
            
            if self.density_axis:
            
                for ax in self.kp_ax:
                    ax.set_xlim(self.rhos[0]*m**3, self.rhos[-1]*m**3)
                    ax.set_xscale("log")
                    ax.set_yscale("log")
                    axis_label = r"Number density $\rho\, \, (\mathrm{m}^{-3})$"
                    ax.set_xlabel(axis_label)
                    

    def add_track_length_axis(self):
        if self.atomic_axis:
            ax2 = self.ax.twiny()
            ax2.set_xscale("log")
            ax2.set_xlabel("(Atomic) track length (s)")
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
            ax3.set_xlabel("(Molecular) track length (s)")
            ax3.set_xlim(self.molecular_sens.track_length(self.rhos[0])/s,
                         self.molecular_sens.track_length(self.rhos[-1])/s)

        else:
            logger.warning("No track length axis added since neither atomic nor molecular was requested")
        self.fig.tight_layout()

    """def add_density_comparison_curve(self, label, color='k'):
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
        self.ax.text(1.5e14, 0.11, label)"""
        
    def add_comparison_curve(self, label, color='k'):
        
            
        limit = [self.sens_ref.CL90(Experiment={"number_density": rho})/eV for rho in self.rhos]
        opt_ref = np.argmin(limit)
        rho_opt = self.rhos[opt_ref]
        
        logger.info('Ref. optimum density: {} /m**3'.format(rho_opt*m**3))
        logger.info('Ref. sigmaE_r: {} eV'.format(self.sens_ref.MagneticField.sigmae_r/eV))
        logger.info('Ref. curve (veff = {} m**3):'.format(self.sens_ref.effective_volume/(m**3)))
        logger.info('Larmor power = {} W, Hanneke power = {} W'.format(self.sens_ref.larmor_power/W, self.sens_ref.signal_power/W))
        logger.info('Ref. T in Veff: {}'.format(rho_opt*self.sens_ref.effective_volume))
        logger.info('Ref. total signal: {}'.format(rho_opt*self.sens_ref.effective_volume*
                                                   self.sens_ref.Experiment.LiveTime/
                                                   self.sens_ref.tau_tritium))
        logger.info('Ref. CL90 limit: {}'.format(self.sens_ref.CL90(Experiment={"number_density": rho_opt})/eV))
        
        limits = self.add_sens_line(self.sens_ref, plot_key_params=True, color=color)

        #self.ax.axhline(0.04, color="gray", ls="--")
        #self.ax.text(3e14, 0.044, "Phase IV (40 meV)")
        self.ax.text(self.label_x_position, 0.042, label)

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
        self.ax.text(self.goal_x_pos, 0.75*value/eV, label)

    def add_density_sens_line(self, sens, plot_key_params=False, **kwargs):
        limits = []
        resolutions = []
        crlb_window = []
        crlb_max_window = []
        crlb_slope_zero_window = []
        
        for rho in self.rhos:
            limits.append(sens.CL90(Experiment={"number_density": rho})/eV)
            resolutions.append(sens.sigma_K_f_CRLB/meV)
            crlb_window.append(sens.best_time_window/ms)
            crlb_max_window.append(sens.time_window/ms)
            crlb_slope_zero_window.append(sens.time_window_slope_zero/ms)
            
        
        self.ax.plot(self.rhos*m**3, limits, **kwargs)
        logger.info('Minimum limit at {}: {}'.format(self.rhos[np.argmin(limits)]*m**3, np.min(limits)))
        
        if self.make_key_parameter_plots and plot_key_params:
            self.kp_ax[0].plot(self.rhos*m**3, resolutions, **kwargs)
            
            self.kp_ax[1].plot(self.rhos*m**3, crlb_max_window, color='red', marker='.')
            self.kp_ax[1].plot(self.rhos*m**3, crlb_slope_zero_window, color='green', marker='.')
            self.kp_ax[1].plot(self.rhos*m**3, crlb_window, linestyle="--", marker='.', **kwargs)
        return limits
        
    def add_exposure_sens_line(self, sens, **kwargs):
        limits = []
        for eff in self.effs:
            sens.Experiment.efficiency = eff
            sens.CavityVolume()
            limit = [self.sens_main.CL90(Experiment={"number_density": rho})/eV for rho in self.rhos]
            opt_ref = np.argmin(limit)
            rho_opt = self.rhos[opt_ref]
            sens.Experiment.number_density = rho_opt
            limits.append(sens.CL90()/eV)
        print(limits)
        self.ax.plot(self.effs, limits, **kwargs)
        
    def add_text(self, x, y, text, color="k"):
        self.ax.text(x, y, text, color=color)

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
            
        if self.make_key_parameter_plots:
            self.kp_fig.savefig('key_parameters.pdf')


