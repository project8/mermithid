'''
Calculate sensitivity curve and plot vs. pressure
function.
Author: R. Reimann, C. Claessens, T. Weiss
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
from numericalunits import meV, eV, keV, MeV, cm, m, mm
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
        self.PhaseII_path = reader.read_param(params, 'PhaseII_config_path', '')


        # labels
        self.main_curve_upper_label = reader.read_param(params, 'main_curve_upper_label', r"molecular"+"\n"+r"$V_\mathrm{eff} = 2\, \mathrm{cm}^3$"+"\n"+r"$\sigma_B = 7\,\mathrm{ppm}$")
        self.main_curve_lower_label = reader.read_param(params, 'main_curve_lower_label', r"$\sigma_B = 1\,\mathrm{ppm}$")
        self.comparison_curve_label = reader.read_param(params, 'comparison_curve_label', r"atomic"+"\n"+r"$V_\mathrm{eff} = 5\, \mathrm{m}^3$"+"\n"+r"$\sigma_B = 0.13\,\mathrm{ppm}$")
        self.comparison_curve_colors = reader.read_param(params,'comparison_curve_colors', ["blue", "darkred", "red"])

        # options
        self.comparison_curve = reader.read_param(params, 'comparison_curve', False)
        self.B_error = reader.read_param(params, 'B_inhomogeneity', 7e-6)
        self.B_error_uncertainty = reader.read_param(params, 'B_inhom_uncertainty', 0.05)
        self.sigmae_theta_r = reader.read_param(params, 'sigmae_theta_r', 0.159) #eV


        # main plot configurations
        self.figsize = reader.read_param(params, 'figsize', (6,6))
        self.legend_location = reader.read_param(params, 'legend_location', 'upper left')
        self.fontsize = reader.read_param(params, 'fontsize', 12)
        self.density_range = reader.read_param(params, 'density_range', [1e14,1e21])
        self.year_range = reader.read_param(params, "years_range", [0.1, 10])
        self.exposure_range = reader.read_param(params, "exposure_range", [1e-10, 1e3])
        self.density_axis = reader.read_param(params, "density_axis", True)
        self.ylim = reader.read_param(params, 'y_limits', [1e-2, 1e2])
        self.track_length_axis = reader.read_param(params, 'track_length_axis', True)
        self.atomic_axis = reader.read_param(params, 'atomic_axis', False)
        self.molecular_axis = reader.read_param(params, 'molecular_axis', False)
        self.label_x_position = reader.read_param(params, 'label_x_position', 5e19)
        self.comparison_label_y_position = reader.read_param(params, 'comparison_label_y_position', 0.044)
        self.comparison_label_x_position = reader.read_param(params, 'comparison_label_x_position', 5e16)
        if self.comparison_label_x_position == None:
            self.comparison_label_x_position = reader.read_param(params, 'label_x_position', 5e16)
        self.upper_label_y_position = reader.read_param(params, 'upper_label_y_position', 5)
        self.lower_label_y_position = reader.read_param(params, 'lower_label_y_position', 2.3)
        self.goal_x_pos = reader.read_param(params, "goals_x_position", 1e14)
        self.add_PhaseII = reader.read_param(params, "add_PhaseII", False)
        self.goals_y_rel_position = reader.read_param(params, "goals_y_rel_position", 0.75)
        
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
        
        if self.add_PhaseII:
            self.sens_PhaseII = CavitySensitivity(self.PhaseII_path)


        # setup sensitivities
        self.cavity = reader.read_param(params, 'cavity', True)
        
        if self.cavity:
            self.sens_main = CavitySensitivity(self.config_file_path)
        else:
            self.sens_main = Sensitivity(self.config_file_path)
        self.sens_main_is_atomic = self.sens_main.Experiment.atomic

        if self.comparison_curve:
            ref = []
            if self.cavity:
                for file in self.comparison_config_file_path:
                    ref.append(CavitySensitivity(file))
            else:
                for file in self.comparison_config_file_path:
                    ref.append(CavitySensitivity(file))
            self.sens_ref = ref
            is_atomic = []
            for i in range(len(self.sens_ref)):
                is_atomic.append(self.sens_ref[i].Experiment.atomic)
            self.sens_ref_is_atomic = is_atomic

        # check atomic and molecular
        if self.molecular_axis:
            if not self.sens_main_is_atomic:
                #self.molecular_sens = self.sens_main
                logger.info("Main curve is molecular")
            elif False not in self.sens_ref_is_atomic:
                raise ValueError("No experiment is configured to be molecular")
                #self.molecular_sens = self.sens_ref
                #logger.info("Comparison curve is molecular")
            #else:

        if self.atomic_axis:
            if self.sens_main_is_atomic:
                #self.atomic_sens = self.sens_main
                logger.info("Main curve is atomic")
            elif True in self.sens_ref_is_atomic:
                #self.atomic_sens = self.sens_ref
                logger.info("A comparison curve is atomic")
            else:
                raise ValueError("No experiment is configured to be atomic")

        # densities
        self.rhos = np.logspace(np.log10(self.density_range[0]), np.log10(self.density_range[1]), 100)/m**3
        self.exposures = np.logspace(np.log10(self.exposure_range[0]), np.log10(self.exposure_range[1]), 100)*m**3*year
        self.years = np.logspace(np.log10(self.year_range[0]), np.log10(self.year_range[1]), 100)*year
        
        return True



    def InternalRun(self):

        self.create_plot()
        
        # optionally add Phase II curve and point to exposure plot
        if self.density_axis == False and self.add_PhaseII:
            self.add_Phase_II_exposure_sens_line(self.sens_PhaseII)

        # add second and third x axis for track lengths
        if self.track_length_axis:
            self.add_track_length_axis()

        for key, value in self.goals.items():
            logger.info('Adding goal: {} = {}'.format(key, value))
            self.add_goal(value, key)
            
        # first optimize density
        limit = [self.sens_main.CL90(Experiment={"number_density": rho})/eV for rho in self.rhos]
        opt = np.argmin(limit)
        rho_opt = self.rhos[opt]
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
            self.add_sens_line(self.sens_main, color='darkblue', label=self.main_curve_upper_label)
            #self.add_text(self.label_x_position, self.upper_label_y_position, self.main_curve_upper_label, color='blue')

        # add line for comparison using second config
        if self.comparison_curve:
            self.add_comparison_curve(label=self.comparison_curve_label)
            #self.add_arrow(self.sens_main)

        
        
        
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

        logger.info('Main curve (molecular):')
        # set optimum density back
        self.sens_main.CL90(Experiment={"number_density": rho_opt})
        logger.info('veff = {} m**3, rho = {} /m**3:'.format(self.sens_main.effective_volume/(m**3), rho_opt*(m**3)))
        logger.info('Larmor power = {} W, Hanneke power = {} W'.format(self.sens_main.larmor_power/W, self.sens_main.signal_power/W))
        logger.info('Hanneke / Larmor power = {}'.format(self.sens_main.signal_power/self.sens_main.larmor_power))
        
        if self.sens_main.FrequencyExtraction.crlb_on_sidebands:
            logger.info("Uncertainty of frequency resolution and energy reconstruction (for pitch angle): {} eV, {} eV".format(self.sens_main.sigma_K_f_CRLB/eV, self.sens_main.sigma_K_reconstruction/eV))
       
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

        # Optimize atomic density
        for i in range(len(self.sens_ref)):
            limit_ref = [self.sens_ref[i].CL90(Experiment={"number_density": rho})/eV for rho in self.rhos]
            opt_ref = np.argmin(limit_ref)
            rho_opt_ref = self.rhos[opt_ref]
            #self.sens_ref[i].Experiment.number_density = rho_opt_ref
            self.sens_ref[i].CL90(Experiment={"number_density": rho_opt_ref})

            logger.info('Comparison curve:')
            logger.info('veff = {} m**3, rho = {} /m**3:'.format(self.sens_ref[i].effective_volume/(m**3), rho_opt_ref*(m**3)))
            logger.info('Larmor power = {} W, Hanneke power = {} W'.format(self.sens_ref[i].larmor_power/W, self.sens_ref[i].signal_power/W))
            logger.info('Hanneke / Larmor power = {}'.format(self.sens_ref[i].signal_power/self.sens_ref[i].larmor_power))
            
            if self.sens_ref[i].FrequencyExtraction.crlb_on_sidebands:
                logger.info("Uncertainty of frequency resolution and energy reconstruction (for pitch angle): {} eV, {} eV".format(self.sens_ref[i].sigma_K_f_CRLB/eV, self.sens_ref[i].sigma_K_reconstruction/eV))
    
            logger.info('CL90 limit: {}'.format(self.sens_ref[i].CL90(Experiment={"number_density": rho_opt_ref})/eV))
            logger.info('T2 in Veff: {}'.format(rho_opt_ref*self.sens_ref[i].effective_volume))
            logger.info('Total signal: {}'.format(rho_opt_ref*self.sens_ref[i].effective_volume*
                                                   self.sens_ref[i].Experiment.LiveTime/
                                                   self.sens_ref[i].tau_tritium*2))
            logger.info('Signal in last eV: {}'.format(self.sens_ref[i].last_1ev_fraction*eV**3*
                                                   rho_opt_ref*self.sens_ref[i].effective_volume*
                                                   self.sens_ref[i].Experiment.LiveTime/
                                                   self.sens_ref[i].tau_tritium*2))

            self.sens_ref[i].print_statistics()
            self.sens_ref[i].print_systematics()
            
        # save plot
        self.save(self.plot_path)

        return True


    def create_plot(self):
        # setup axis
        plt.rcParams.update({'font.size': self.fontsize})
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
                
            ax.set_xlabel(axis_label)
            ax.set_ylim(self.ylim)
            ax.set_ylabel(r"90% CL $m_\beta$ (eV)")

            
            
        else:
            logger.info("Adding exposure axis")
            ax.set_xlim(self.exposure_range[0], self.exposure_range[-1])
            ax.tick_params(axis='x', which='minor', bottom=True)
            axis_label = r"Efficiency $\times$ Volume $\times$ Time (m$^3$y)"
            
            ax.set_xlabel(axis_label)
            ax.set_ylim((np.array(self.ylim)**2/np.sqrt(1.64)))
            ax.set_ylabel(r"Standard deviation in $m_\beta^2$ (eV$^2$)")
            
            self.ax2 = ax.twinx()
            self.ax2.set_ylabel(r"90% CL $m_\beta$ (eV)")
            self.ax2.set_yscale("log")
            self.ax2.set_ylim(self.ylim)
        
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
        N_ref = len(self.sens_ref)
        if self.atomic_axis:
            ax2 = self.ax.twiny()
            ax2.set_xscale("log")
            ax2.set_xlabel("(Atomic) track length (s)")
            ax2.set_xlim(self.sens_ref[N_ref - 1].track_length(self.rhos[0])/s,
                         self.sens_ref[N_ref - 1].track_length(self.rhos[-1])/s)

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
            ax3.set_xlim(self.sens_main.track_length(self.rhos[0])/s,
                         self.sens_main.track_length(self.rhos[-1])/s)

        else:
            logger.warning("No track length axis added since neither atomic nor molecular was requested")
        self.fig.tight_layout()

        
    def add_comparison_curve(self, label, color='k'):
        
        """    
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
        """
    
        for a in range(len(self.sens_ref)):
            limits = self.add_sens_line(self.sens_ref[a], plot_key_params=True, color=self.comparison_curve_colors[a], label=label[a])
            #self.ax.text(self.comparison_label_x_position[a], self.comparison_label_y_position[a], label[a], color=colors[a], fontsize=9.5)


        #self.ax.axhline(0.04, color="gray", ls="--")
        #self.ax.text(3e14, 0.044, "Phase IV (40 meV)")

    def add_arrow(self, sens):
        if not hasattr(self, "opt_ref"):
            self.add_comparison_curve()

        def get_relative(val, axis):
            xmin, xmax = self.ax.get_xlim() if axis == "x" else self.ax.get_ylim()
            return (np.log10(val)-np.log10(xmin))/(np.log10(xmax)-np.log10(xmin))
        
        """
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
        """

    def add_goal(self, value, label):
        self.ax.axhline(value, color="gray", ls="--")
        self.ax.text(self.goal_x_pos, self.goals_y_rel_position*value, label)

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
        
    def add_exposure_sens_line(self, sens, plot_key_params=False, **kwargs):
        
        limit = [sens.sensitivity(Experiment={"number_density": rho})/eV**2 for rho in self.rhos]
        opt = np.argmin(limit)
        rho_opt = self.rhos[opt]
        sens.Experiment.number_density = rho_opt
        
        logger.info("Optimum density: {} /m^3".format(rho_opt*m**3))
        logger.info("Years: {}".format(sens.Experiment.livetime/year))
        
        standard_exposure = sens.EffectiveVolume()*sens.Experiment.livetime/m**3/year
        
        print(kwargs)
        
        self.ax.scatter([standard_exposure], [np.min(limit)], marker="s", **kwargs)
        
        limits = []
        years = []
        for ex in self.exposures:
            lt = ex/sens.EffectiveVolume()
            years.append(lt/year)
            sens.Experiment.livetime = lt
            limits.append(sens.sensitivity()/eV**2)
            #exposures.append(sens.EffectiveVolume()/m**3*sens.Experiment.livetime/year)
            
        if sens.Experiment.atomic:
            gas = "T"
        else:
            gas = r"T$_2$"
        unit = r"m$^{-3}$"
        self.ax.plot(self.exposures/m**3/year, limits, color=kwargs["color"]) #label="{} density = {:.1e} {}".format(gas, rho_opt*m**3, unit))
        
    def add_Phase_II_exposure_sens_line(self, sens):
        sens.Experiment.number_density = 2.09e17/m**3
        sens.effective_volume = 1.2*mm**3
        sens.Experiment.sri_factor = 1 #0.389*0.918*0.32
        sens.Experiment.livetime = 7185228*s
        sens.CRLB_constant = 180
        
        standard_exposure = sens.effective_volume*sens.Experiment.livetime/m**3/year
        sens.print_systematics()
        sens.print_statistics()
        sens.sensitivity()
        logger.info("Phase II sensitivity for exposure {} calculated: {}".format(standard_exposure, sens.sensitivity()/eV**2))
        
        
        phaseIIsens = 9822
        phaseIIsense_error = 1520
        exposure_error = np.sqrt((standard_exposure*0.008)**2 + (standard_exposure*0.09)**2)
        
        self.ax.errorbar([standard_exposure], [phaseIIsens], xerr=[exposure_error], yerr=[phaseIIsense_error], fmt='.', color='k', label="Phase II (measured)")
        
        
        limits = []
        years = []
        for ex in self.exposures:
            lt = ex/sens.effective_volume
            years.append(lt/year)
            sens.Experiment.livetime = lt
            limits.append(sens.sensitivity()/eV**2)
            #exposures.append(sens.EffectiveVolume()/m**3*sens.Experiment.livetime/year)
            
        unit = r"m$^{-3}$"
        gas = r"T$_2$"
        self.ax.plot(self.exposures/m**3/year, limits, color='k', linestyle=':')#, label="{} density = {:.1e} {}".format(gas, 7.5e16, unit))
        
        def get_relative(val, axis):
            xmin, xmax = self.ax.get_xlim() if axis == "x" else self.ax.get_ylim()
            return (np.log10(val)-np.log10(xmin))/(np.log10(xmax)-np.log10(xmin))
        
        
        """x_start = get_relative(1e-3, "x")
        y_start = get_relative(6e1, "y")
        x_stop = get_relative(1e-4, "x")
        y_stop = get_relative(3e1, "y")"""
        
        x_start = get_relative(4e-7, "x")
        y_start = get_relative(6e3, "y")
        x_stop = get_relative(5e-8, "x")
        y_stop = get_relative(8e2, "y")
        self.ax.arrow(x_start, y_start, x_stop-x_start, y_stop-y_start,
                      transform = self.ax.transAxes,
                      facecolor = 'black',
                      edgecolor='k',
                      length_includes_head=True,
                      head_width=0.01,
                      head_length=0.01,
                      )
        print(x_start, y_start)
        self.ax.annotate("Phase II T$_2$ density \nand resolution", xy=[x_start*1.01, y_start*1.01],textcoords="axes fraction", fontsize=13)
        
        

        
    def add_text(self, x, y, text, color="k"): #, fontsize=9.5
        self.ax.text(x, y, text, color=color)

    def range(self, start, stop):
        cmap = matplotlib.cm.get_cmap('Spectral')
        norm = matplotlib.colors.Normalize(vmin=start, vmax=stop)
        return [(idx, cmap(norm(idx))) for idx in range(start, stop)]

    def save(self, savepath, **kwargs):
        if self.density_axis:
            legend=self.fig.legend(loc=self.legend_location, framealpha=0.95, bbox_to_anchor=(0.15,0,1,0.765))
        else:
            legend=self.fig.legend(loc=self.legend_location, framealpha=0.95, bbox_to_anchor=(-0.,0,0.89,0.97))
            
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


