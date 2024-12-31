'''
Calculate sensitivity curve and plot vs. number density, exposure, livetime, or frequency.

Author: C. Claessens, T. E. Weiss
Date: 06/07/2023
Updated: 12/16/2024
'''

from __future__ import absolute_import


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy


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
from mermithid.sensitivity.SensitivityCavityFormulas import CavitySensitivity


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
        self.main_curve_color = reader.read_param(params, 'main_curve_color', "darkblue")

        # options
        #self.optimize_main_density = reader.read_param(params, 'optimize_main_density', True)
        self.optimize_comparison_density = reader.read_param(params, 'optimize_comparison_density', True)
        self.verbose = reader.read_param(params, 'verbose', True)
        self.comparison_curve = reader.read_param(params, 'comparison_curve', False)
        self.B_error = reader.read_param(params, 'B_inhomogeneity', 7e-6)
        self.B_error_uncertainty = reader.read_param(params, 'B_inhom_uncertainty', 0.05)
        self.sigmae_theta_r = reader.read_param(params, 'sigmae_theta_r', 0.159) #eV
        self.configure_sigma_theta_r = reader.read_param(params, "configure_sigma_theta_r", False)

        # main plot configurations
        self.figsize = reader.read_param(params, 'figsize', (6,6))
        self.legend_location = reader.read_param(params, 'legend_location', 'upper left')
        self.fontsize = reader.read_param(params, 'fontsize', 12)
        
        self.density_axis = reader.read_param(params, "density_axis", True)
        self.frequency_axis = reader.read_param(params, "frequency_axis", False)
        self.exposure_axis = reader.read_param(params, "exposure_axis", False)
        self.livetime_axis = reader.read_param(params, "livetime_axis", False)
        self.track_length_axis = reader.read_param(params, 'track_length_axis', True)
        self.atomic_axis = reader.read_param(params, 'atomic_axis', False)
        self.molecular_axis = reader.read_param(params, 'molecular_axis', False)
        self.magnetic_field_axis = reader.read_param(params, 'magnetic_field_axis', False)
        
        self.density_range = reader.read_param(params, 'density_range', [1e14,1e21])
        self.year_range = reader.read_param(params, "year_range", [0.1, 20])
        self.exposure_range = reader.read_param(params, "exposure_range", [1e-10, 1e3])
        self.frequency_range = reader.read_param(params, "frequency_range", [1e6, 1e9])
        
        self.ylim = reader.read_param(params, 'y_limits', [1e-2, 1e2])
        
        self.label_x_position = reader.read_param(params, 'label_x_position', 5e19)
        self.upper_label_y_position = reader.read_param(params, 'upper_label_y_position', 5)
        self.lower_label_y_position = reader.read_param(params, 'lower_label_y_position', 2.3)
        self.goal_x_pos = reader.read_param(params, "goals_x_position", 1e14)
        
        self.comparison_label_y_position = reader.read_param(params, 'comparison_label_y_position', 0.044)
        self.comparison_label_x_position = reader.read_param(params, 'comparison_label_x_position', 5e16)
        if self.comparison_label_x_position == None:
            self.comparison_label_x_position = reader.read_param(params, 'label_x_position', 5e16)
        
        self.add_PhaseII = reader.read_param(params, "add_PhaseII", False)
        self.goals_y_rel_position = reader.read_param(params, "goals_y_rel_position", 0.75)
        self.add_1year_1cav_point_to_last_ref = reader.read_param(params, "add_1year_1cav_point_to_last_ref", False)
        
        # key parameter plots
        self.make_key_parameter_plots = reader.read_param(params, 'plot_key_parameters', False)
        
        if self.density_axis:
            self.add_sens_line = self.add_density_sens_line
            logger.info("Plotting sensitivity vs. density")
        elif self.frequency_axis:
            self.add_sens_line = self.add_frequency_sens_line
            logger.info("Plotting sensitivity vs. frequency")
        elif self.exposure_axis or self.livetime_axis:
            self.add_sens_line = self.add_exposure_sens_line
            logger.info("Plotting sensitivity vs. exposure or livetime")
        #elif self.livetime_axis:
        #    self.add_sens_line = self.add_exposure_sens_line(livetime_plot=True)
        #    logger.info("Plotting sensitivity vs. livetime")
        else: raise ValueError("No axis specified")


        # goals
        self.goals = reader.read_param(params, "goals", {})


        # setup sensitivities
        if self.add_PhaseII:
            self.sens_PhaseII = CavitySensitivity(self.PhaseII_path)
            
        
        self.sens_main = CavitySensitivity(self.config_file_path)
        self.sens_main_is_atomic = self.sens_main.Experiment.atomic

        if self.comparison_curve:
            ref = []
            for file in self.comparison_config_file_path:
                ref.append(CavitySensitivity(file))
            
            self.sens_ref = ref
            is_atomic = []
            for i in range(len(self.sens_ref)):
                is_atomic.append(self.sens_ref[i].Experiment.atomic)
            self.sens_ref_is_atomic = is_atomic
        else:
            self.sens_ref_is_atomic = [False]

        # check atomic and molecular
        if self.molecular_axis:
            if not self.sens_main_is_atomic:
                #self.molecular_sens = self.sens_main
                logger.info("Main curve is molecular")
            elif False not in self.sens_ref_is_atomic:
                logger.warn("No experiment is configured to be molecular")
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
                logger.warn("No experiment is configured to be atomic")

        # densities, exposures, runtimes
        self.rhos = np.logspace(np.log10(self.density_range[0]), np.log10(self.density_range[1]), 150)/m**3
        self.exposures = np.logspace(np.log10(self.exposure_range[0]), np.log10(self.exposure_range[1]), 100)*m**3*year
        self.years = np.logspace(np.log10(self.year_range[0]), np.log10(self.year_range[1]), 100)*year
        self.frequencies = np.logspace(np.log10(self.frequency_range[0]), np.log10(self.frequency_range[1]), 20)*Hz
        
        return True



    def InternalRun(self):
        logger.info("Systematics before density optimization:")
        self.sens_main.print_systematics()
        logger.info("Number density: {} \m^3".format(self.sens_main.Experiment.number_density*m**3))
        logger.info("Corresponding track length: {} s".format(self.sens_main.track_length(self.sens_main.Experiment.number_density)/s))
                
        if self.make_key_parameter_plots:
            
            # First key parameter plot: Stat and Syst vs. density
            
            sigma_startf, stat_on_mbeta2, syst_on_mbeta2 = [], [], []

            for n in self.rhos:
                temp_rho = deepcopy(self.sens_main.Experiment.number_density)
                self.sens_main.Experiment.number_density = n
                labels, sigmas, deltas = self.sens_main.get_systematics()
                sigma_startf.append(sigmas[1])
                stat_on_mbeta2.append(self.sens_main.StatSens())
                syst_on_mbeta2.append(self.sens_main.SystSens())
                self.sens_main.Experiment.number_density = temp_rho
                    
            sigma_startf, stat_on_mbeta2, syst_on_mbeta2 = np.array(sigma_startf), np.array(stat_on_mbeta2), np.array(syst_on_mbeta2)
            fig = plt.figure()
            plt.loglog(self.rhos*m**3, stat_on_mbeta2/eV**2, label='Statistical uncertainty')
            plt.loglog(self.rhos*m**3, syst_on_mbeta2/eV**2, label='Systematic uncertainty')
            plt.xlabel(r"Number density $n\, \, (\mathrm{m}^{-3})$")
            plt.ylabel(r"Standard deviation in $m_\beta^2$ (eV$^2$)")
            plt.legend()
            plt.tight_layout()
            plt.savefig("stat_and_syst_vs_density.pdf")

            fig = plt.figure()
            plt.loglog(self.rhos*m**3, sigma_startf/eV)
            plt.xlabel(r"Number density $n\, \, (\mathrm{m}^{-3})$")
            plt.ylabel(r"Resolution from $f$ reconstruction, axial field (eV)")
            plt.tight_layout()
            plt.savefig("resolution_from_CRLB_vs_density.pdf")


        # create main plot
        self.create_plot()
        
        # optionally add Phase II curve and point to exposure plot
        if self.exposure_axis and self.add_PhaseII:
            self.add_Phase_II_exposure_sens_line(self.sens_PhaseII)

        # add second and third x axis for track lengths
        if self.density_axis and self.track_length_axis:
            self.add_track_length_axis()
            
        # add magnetic field axis if frequency axis
        if self.frequency_axis and self.magnetic_field_axis:
            self.add_magnetic_field_axis()

        # add goals
        for key, value in self.goals.items():
            logger.info('Adding goal: {} = {}'.format(key, value))
            self.add_goal(value, key)
            
        # optimize density
        limit = [self.sens_main.CL90(Experiment={"number_density": rho})/eV for rho in self.rhos]
        opt = np.argmin(limit)
        rho_opt = self.rhos[opt]
        self.sens_main.Experiment.number_density = rho_opt

        # if B is list plot line for each B
        if self.configure_sigma_theta_r and (isinstance(self.sigmae_theta_r, list) or isinstance(self.sigmae_theta_r, np.ndarray)):
            N = len(self.sigmae_theta_r)
            for a, color in self.range(0, N):
                #sig = self.sens_main.BToKeErr(self.sens_main.MagneticField.nominal_field*self.B_error[a], self.sens_main.MagneticField.nominal_field)
                #self.sens_main.MagneticField.usefixedvalue = True
                #self.sens_main.MagneticField.default_systematic_smearing = sig
                #self.sens_main.MagneticField.default_systematic_uncertainty = 0.05*sig
                self.sens_main.MagneticField.sigmae_r = self.sigmae_theta_r[a] * eV
                self.sens_main.MagneticField.sigmae_theta = 0 * eV
                if self.livetime_axis:
                    self.add_sens_line(self.sens_main, livetime_plot=True, color=color)
                else:
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
            if self.configure_sigma_theta_r:
                self.sens_main.MagneticField.sigmaer = self.sigmae_theta_r * eV
                self.sens_main.MagneticField.sigmae_theta = 0 * eV
            if self.livetime_axis:
                    self.add_sens_line(self.sens_main, livetime_plot=True, color=self.main_curve_color, label=self.main_curve_upper_label)
            else:
                self.add_sens_line(self.sens_main, color=self.main_curve_color, label=self.main_curve_upper_label)
            #self.add_text(self.label_x_position, self.upper_label_y_position, self.main_curve_upper_label, color='blue')

        
        # PRINT OPTIMUM RESULTS
            
        # if the magnetic field uncertainties were configured above, set them back to the first value in the list    
        if self.configure_sigma_theta_r and (isinstance(self.sigmae_theta_r, list) or isinstance(self.sigmae_theta_r, np.ndarray)):
            self.sens_main.MagneticField.sigmae_r = self.sigmae_theta_r[0] * eV
            self.sens_main.MagneticField.sigmae_theta = 0 * eV
        elif self.configure_sigma_theta_r:
            self.sens_main.MagneticField.sigmaer = self.sigmae_theta_r * eV
            self.sens_main.MagneticField.sigmae_theta = 0 * eV

        logger.info('Main curve:')
        # set optimum density back
        #self.sens_main.CL90(Experiment={"number_density": rho_opt})
        rho = self.sens_main.Experiment.number_density
        logger.info('veff = {} m**3, rho = {} /m**3:'.format(self.sens_main.effective_volume/(m**3), rho*(m**3)))
        logger.info("Loaded Q: {}".format(self.sens_main.loaded_q))
        logger.info("Axial frequency for minimum detectable angle: {} MHz".format(self.sens_main.required_bw_axialfrequency/MHz))
        logger.info("Total bandwidth: {} MHz".format(self.sens_main.required_bw/MHz))
        logger.info('Larmor power = {} W, Hanneke power = {} W'.format(self.sens_main.larmor_power/W, self.sens_main.signal_power/W))
        logger.info('Hanneke / Larmor power = {}'.format(self.sens_main.signal_power/self.sens_main.larmor_power))
        
        if self.sens_main.FrequencyExtraction.crlb_on_sidebands:
            logger.info("Trap p: {}".format(self.sens_main.p))
            logger.info("Trap q: {}".format(self.sens_main.q))
            #print(self.sens_main.q_array)
            logger.info("Uncertainty from determination of f_carrier and f_lsb, due to noise: {} eV".format(self.sens_main.sigma_K_noise/eV))
       
        self.sens_main.print_SNRs(rho_opt)
        self.sens_main.print_Efficiencies()
        if self.exposure_axis or self.livetime_axis:
            logger.info("NUMBERS BELOW ARE FOR THE HIGHEST-EXPOSURE POINT ON THE CURVE:")
        logger.info('CL90 limit: {}'.format(self.sens_main.CL90(Experiment={"number_density": rho_opt})/eV))
        logger.info('T2 in Veff: {}'.format(rho_opt*self.sens_main.effective_volume))
        logger.info('RF background: {}/eV/s'.format(self.sens_main.RF_background_rate_per_eV*eV*s))
        logger.info('Total background: {}/eV/s'.format(self.sens_main.background_rate*eV*s))
        logger.info('Total signal: {}'.format(rho_opt*self.sens_main.effective_volume*
                                                   self.sens_main.Experiment.LiveTime/
                                                   self.sens_main.tau_tritium*2))
        logger.info('Signal in last eV: {}'.format(self.sens_main.last_1ev_fraction*eV**3*
                                                   rho*self.sens_main.effective_volume*
                                                   self.sens_main.Experiment.LiveTime/
                                                   self.sens_main.tau_tritium*2))

        self.sens_main.print_statistics()
        self.sens_main.print_systematics()

        # Optimize comparison curves over density
        if self.comparison_curve:
                    
            for i in range(len(self.sens_ref)):
                
                if self.optimize_comparison_density:
                    limit_ref = [self.sens_ref[i].CL90(Experiment={"number_density": rho})/eV for rho in self.rhos]
                    opt_ref = np.argmin(limit_ref)
                    rho_opt_ref = self.rhos[opt_ref]
                    #self.sens_ref[i].Experiment.number_density = rho_opt_ref
                    self.sens_ref[i].CL90(Experiment={"number_density": rho_opt_ref})
                
                
 

                logger.info('Comparison curve:')
                logger.info('veff = {} m**3, rho = {} /m**3:'.format(self.sens_ref[i].effective_volume/(m**3), rho_opt_ref*(m**3)))
                logger.info("Loaded Q: {}".format(self.sens_ref[i].loaded_q))
                logger.info('Larmor power = {} W, Hanneke power = {} W'.format(self.sens_ref[i].larmor_power/W, self.sens_ref[i].signal_power/W))
                logger.info('Hanneke / Larmor power = {}'.format(self.sens_ref[i].signal_power/self.sens_ref[i].larmor_power))
            
                if self.sens_ref[i].FrequencyExtraction.crlb_on_sidebands:
                    logger.info("Uncertainty from determination of f_carrier and f_lsb, due to noise: {} eV".format(self.sens_ref[i].sigma_K_noise/eV))
                        
                self.sens_ref[i].print_SNRs(rho_opt_ref)
                self.sens_ref[i].print_Efficiencies()
                if self.exposure_axis or self.livetime_axis:
                    logger.info("NUMBERS BELOW ARE FOR THE HIGHEST-EXPOSURE POINT ON THE CURVE:")
                logger.info('CL90 limit: {}'.format(self.sens_ref[i].CL90(Experiment={"number_density": rho_opt_ref})/eV))
                logger.info('T2 in Veff: {}'.format(rho_opt_ref*self.sens_ref[i].effective_volume))
                logger.info('RF background: {}/eV/s'.format(self.sens_ref[i].RF_background_rate_per_eV*eV*s))
                logger.info('Total background: {}/eV/s'.format(self.sens_ref[i].background_rate*eV*s))
                logger.info('Total signal: {}'.format(rho_opt_ref*self.sens_ref[i].effective_volume*
                                                   self.sens_ref[i].Experiment.LiveTime/
                                                   self.sens_ref[i].tau_tritium*2))
                logger.info('Signal in last eV: {}'.format(self.sens_ref[i].last_1ev_fraction*eV**3*
                                                   rho_opt_ref*self.sens_ref[i].effective_volume*
                                                   self.sens_ref[i].Experiment.LiveTime/
                                                   self.sens_ref[i].tau_tritium*2))

                self.sens_ref[i].print_statistics()
                self.sens_ref[i].print_systematics()
                
            self.add_comparison_curve(label=self.comparison_curve_label)
            #self.add_arrow(self.sens_main)
                
                
            
        # save plot
        self.save(self.plot_path)

        if self.verbose:
            self.print_disclaimers()

        return True


    def print_disclaimers(self):
        logger.info("Disclaimers / assumptions:")
        logger.info("1. Often, in practice, 'sigmae_r' is used to stand-in for combined radial, \
                    azimuthal, and temporal magnetic field spectral broadening effects. Check \
                    your experiment config file *and* your processor config dictionary to see \
                    if that is the case.")
        logger.info("2. Trap design is not yet linked to cavity L/D in the sensitivity model. So, \
                    the model does *not* capture how reducing L/D worsens the resolution.")
        logger.info("3. In reality, the frequency resolution could be worse or somewhat better \
                    than predicted by the general CRLB calculation used here. See work by Florian.")
        logger.info("4. The analytic sensitivity formula oaccounts for energy resolution contributions \
                    that are *normally distributed*. (Energy resolution = std of the response fn \
                    that broadens the spectrum.) To account for asymmetric contributions, generate \
                    spectra with MC sampling and then analyze them. This can be done in mermithid.")
        logger.info("5. The best-fit mbeta is assumed to be zero when converting to a 90\% limit.")
        if self.density_axis:
            logger.info("6. This sensitivity formula does not work for very small numbers of counts, \
                        because the analytic formula assumes Gaussian statistics. In typical Phase IV \
                        scenarios, if the minimum allowed density is 1e-20 atoms/m^3, the optimization \
                        over density still works.")
        logger.info("Once you have read these disclaimers and are familiar with them, you can set \
                    verbose==False in your config dictionary to stop seeing them.")

    
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
                axis_label = r"(Atomic / molecular) number density $n\, \, (\mathrm{m}^{-3})$"
            elif self.atomic_axis:
                axis_label = r"Atom number density $n\, \, (\mathrm{m}^{-3})$"
            elif self.molecular_axis:
                axis_label = r"Molecular number density $n\, \, (\mathrm{m}^{-3})$"
            else:
                axis_label = r"Number density $n\, \, (\mathrm{m}^{-3})$"
                
            ax.set_xlabel(axis_label)
            ax.set_ylim(self.ylim)
            ax.set_ylabel(r"90% CL $m_\beta$ (eV)")

            
        elif self.frequency_axis:
            logger.info("Adding frequency axis")
            ax.set_xlim(self.frequencies[0]/Hz, self.frequencies[-1]/Hz)
            ax.set_xlabel("TE011 frequency (Hz)")
            #ax.set_ylim((np.array(self.ylim)**2/np.sqrt(1.64)))
            ax.set_ylim((np.array(self.ylim)**2/1.64))
            ax.set_ylabel(r"Standard deviation in $m_\beta^2$ (eV$^2$)")
            
            self.ax2 = ax.twinx()
            self.ax2.set_ylabel(r"90% CL $m_\beta$ (eV)")
            self.ax2.set_yscale("log")
            self.ax2.set_ylim(self.ylim)
            
        elif self.exposure_axis:
            logger.info("Adding exposure axis")
            ax.set_xlim(self.exposure_range[0], self.exposure_range[-1])
            #ax.tick_params(axis='x', which='minor', bottom=True)
            #ax.tick_params(axis='y', which='minor', left=True)
            axis_label = r"Effective Volume $\times$ Time (m$^3$y)" #r"Efficiency $\times$ Volume $\times$ Time (m$^3$y)"
            
            ax.set_xlabel(axis_label)
            #ax.set_ylim((np.array(self.ylim)**2/np.sqrt(1.64)))
            ax.set_ylim((np.array(self.ylim)**2/1.64))
            ax.set_ylabel(r"Standard deviation in $m_\beta^2$ (eV$^2$)")
            
            self.ax2 = ax.twinx()
            self.ax2.set_ylabel(r"90% CL $m_\beta$ (eV)")
            self.ax2.set_yscale("log")
            self.ax2.set_ylim(self.ylim)

        else:
            logger.info("Adding livetime axis")
            ax.set_xlim(self.year_range[0], self.year_range[-1])
            axis_label = r"Livetime (years)"
            
            ax.set_xlabel(axis_label)
            ax.set_ylim((np.array(self.ylim)**2/1.64))
            ax.set_ylabel(r"Standard deviation in $m_\beta^2$ (eV$^2$)")
            
            self.ax2 = ax.twinx()
            self.ax2.set_ylabel(r"90% CL $m_\beta$ (eV)")
            self.ax2.set_yscale("log")
            self.ax2.set_ylim(self.ylim)
        
        if self.make_key_parameter_plots:
            
            
            if self.density_axis:
                
                self.kp_fig, self.kp_ax = plt.subplots(1,2, figsize=(10,6))
                self.kp_fig.tight_layout()
            
                for ax in self.kp_ax:
                    ax.set_xlim(self.rhos[0]*m**3, self.rhos[-1]*m**3)
                    ax.set_xscale("log")
                    ax.set_yscale("log")
                    axis_label = r"Number density $n\, \, (\mathrm{m}^{-3})$"
                    ax.set_xlabel(axis_label)
                    
                self.kp_ax[0].set_ylabel('Resolution (meV)')
                self.kp_ax[1].set_ylabel('Track analysis duration (ms)')
                    
            elif self.frequency_axis:
                
                self.kp_fig, self.kp_ax = plt.subplots(2,2, figsize=(10,10))
                
            
                self.kp_ax = self.kp_ax.flatten()
                for ax in self.kp_ax:
                    ax.set_xlim(self.frequencies[0]/Hz, self.frequencies[-1]/Hz)
                    ax.set_xscale("log")
                    ax.set_yscale("log")
                    axis_label = "TE011 frequency (Hz)"
                    ax.set_xlabel(axis_label)
                    ax.axvline(325e6, linestyle="--", color="grey")
                self.kp_ax[0].set_ylabel('Resolution (meV)')
                self.kp_ax[1].set_ylabel(r'Optimum desnity (1/m$^3$)')
                self.kp_ax[2].set_ylabel(r'Total and effective (dashed) Volume (m$^3$)')
                self.kp_ax[3].set_ylabel('Noise temperature (K)')

            self.kp_fig.tight_layout()
                    
    def add_track_length_axis(self):
       
        if self.atomic_axis:
            ax2 = self.ax.twiny()
            ax2.set_xscale("log")
            #ax2.set_xlabel("(Atomic) track duration (s)")
            ax2.set_xlabel("Track duration (s)")
            
            if self.sens_main_is_atomic:
                ax2.set_xlim(self.sens_main.track_length(self.rhos[0])/s,
                            self.sens_main.track_length(self.rhos[-1])/s)
            else:
                for sens in self.sens_ref:
                    if sens.Experiment.atomic:
                        ax2.set_xlim(sens.track_length(self.rhos[0])/s,
                                    sens.track_length(self.rhos[-1])/s)

        if self.molecular_axis:
            ax3 = self.ax.twiny()
            ax3.set_xscale("log")
            ax3.set_xlabel("(Molecular) track duration (s)")

            if self.atomic_axis:
                ax3.spines["top"].set_position(("axes", 1.2))
                ax3.set_frame_on(True)
                ax3.patch.set_visible(False)
                for sp in ax3.spines.values():
                    sp.set_visible(False)
                ax3.spines["top"].set_visible(True)
                
            if not self.sens_main_is_atomic:
                ax3.set_xlim(self.sens_main.track_length(self.rhos[0])/s,
                            self.sens_main.track_length(self.rhos[-1])/s)
            else:
                for sens in self.sens_ref:
                    if not sens.Experiment.atomic:
                        ax3.set_xlim(sens.track_length(self.rhos[0])/s,
                            sens.track_length(self.rhos[-1])/s)

        if not self.molecular_axis and not self.atomic_axis:
            logger.warning("No track length axis added since neither atomic nor molecular was requested")
        self.fig.tight_layout()
        
    def add_magnetic_field_axis(self):
        ax3 = self.ax.twiny()
        ax3.set_xscale("log")
        ax3.set_xlabel("Magnetic field strength (T)")
        
        gamma = self.sens_main.T_endpoint/(me*c0**2) + 1
        ax3.set_xlim(self.frequencies[0]/(e/(2*np.pi*me)/gamma)/T, self.frequencies[-1]/(e/(2*np.pi*me)/gamma)/T)
             
    def add_comparison_curve(self, label, color='k'):
    
        for a in range(len(self.sens_ref)):
            if self.livetime_axis == True:
                limits = self.add_sens_line(self.sens_ref[a], livetime_plot=True, plot_key_params=True, color=self.comparison_curve_colors[a], label=label[a])
            else:
                limits = self.add_sens_line(self.sens_ref[a], plot_key_params=True, color=self.comparison_curve_colors[a], label=label[a])
            #self.ax.text(self.comparison_label_x_position[a], self.comparison_label_y_position[a], label[a], color=colors[a], fontsize=9.5)

        if not self.density_axis and self.add_1year_1cav_point_to_last_ref:
            logger.info("Going to add single exposure point")
            self.add_single_exposure_point(self.sens_ref[-1], color=self.comparison_curve_colors[-1])
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
        det_effs = []
        
        temp_rho = deepcopy(sens.Experiment.number_density)
        for rho in self.rhos:
            limits.append(sens.CL90(Experiment={"number_density": rho})/eV)
            resolutions.append(sens.sigma_K_noise/meV)
            crlb_window.append(sens.best_time_window/ms)
            crlb_max_window.append(sens.time_window/ms)
            crlb_slope_zero_window.append(sens.time_window_slope_zero/ms)
            det_effs.append(self.sens_main.detection_efficiency)

        print(det_effs)   
        sens.Experiment.number_density = temp_rho
        self.ax.plot(self.rhos*m**3, limits, **kwargs)
        logger.info('Minimum limit at {}: {}'.format(self.rhos[np.argmin(limits)]*m**3, np.min(limits)))
        
        if self.make_key_parameter_plots and plot_key_params:
            self.kp_ax[0].plot(self.rhos*m**3, resolutions, **kwargs)
            
            self.kp_ax[1].plot(self.rhos*m**3, crlb_max_window, color='red', marker='.')
            self.kp_ax[1].plot(self.rhos*m**3, crlb_slope_zero_window, color='green', marker='.')
            self.kp_ax[1].plot(self.rhos*m**3, crlb_window, linestyle="--", marker='.', **kwargs)
        return limits
    
    def add_single_exposure_point(self, sens, livetime=1*year, n_cavities=1, color="red"):
        logger.info("Adding exposure point")
        
        if livetime/year % 1 == 0:
            livetime_for_label = int(round(livetime/year))
        else:
            livetime_for_label = round(livetime/year, 1)
            
        if sens.Experiment.atomic:
            label="Atomic, reaching pilot-T target".format(n_cavities, livetime_for_label)
        else:
            label="Molecular, {} cavity, {} year".format(n_cavities, livetime_for_label)
        
        #exposure_fraction = n_cavities*livetime/(sens.Experiment.n_cavities*sens.Experiment.livetime)
        
        sens.Experiment.n_cavities = n_cavities
        sens.Experiment.livetime = livetime
        sens.CavityVolume()
        sens.EffectiveVolume()
        sens.CavityPower()
        
        #limit = [sens.sensitivity(Experiment={"number_density": rho})/eV**2 for rho in self.rhos]
        #opt = np.argmin(limit)
        #rho_opt = self.rhos[opt]
        #sens.Experiment.number_density = rho_opt
        
        
        limit = sens.sensitivity()/eV**2
        
        standard_exposure = sens.EffectiveVolume()*sens.Experiment.livetime/m**3/year #*exposure_fraction
        #lt = standard_exposure/sens.EffectiveVolume()
        #sens.Experiment.livetime = lt
        limit = sens.sensitivity()/eV**2
            
        if self.exposure_axis:
            self.ax.scatter([standard_exposure], [limit], marker="s", s=25, color=color, label=label, zorder=10)
            logger.info("Exposure and mass limit for single point: {}, {}".format(standard_exposure, np.sqrt(1.64*limit)))
        if self.livetime_axis:
            self.ax.scatter([sens.Experiment.livetime/year], [limit], marker="s", s=25, color=color, label=label, zorder=10)
            logger.info("Livetime and mass limit for single point: {}, {}".format(sens.Experiment.livetime/year, np.sqrt(1.64*limit)))

        sens.print_statistics()
        sens.print_systematics()
             
    def add_exposure_sens_line(self, sens, livetime_plot=False, plot_key_params=False, **kwargs):
        
        sigma_mbeta = [sens.sensitivity(Experiment={"number_density": rho})/eV**2 for rho in self.rhos]
        opt = np.argmin(sigma_mbeta)
        rho_opt = self.rhos[opt]
        sens.Experiment.number_density = rho_opt
        
        #logger.info("Optimum density: {} /m^3".format(rho_opt*m**3))
        logger.info("Years: {}".format(sens.Experiment.livetime/year))
        
        sigma_mbetas = []
        years = []
        if livetime_plot:
            self.ax.scatter([sens.Experiment.livetime/year], [np.min(sigma_mbeta)], s=40, marker="d", zorder=20, **kwargs)
            for lt in self.years:
                sens.Experiment.livetime = lt
                sigma_mbetas.append(sens.sensitivity()/eV**2)
            self.ax.plot(self.years/year, sigma_mbetas, color=kwargs["color"])
        else:
            standard_exposure = sens.EffectiveVolume()*sens.Experiment.livetime/m**3/year
            self.ax.scatter([standard_exposure], [np.min(sigma_mbeta)], s=40, marker="d", zorder=20, **kwargs)
            for ex in self.exposures:
                lt = ex/sens.EffectiveVolume()
                years.append(lt/year)
                sens.Experiment.livetime = lt
                sigma_mbetas.append(sens.sensitivity()/eV**2)
            self.ax.plot(self.exposures/m**3/year, sigma_mbetas, color=kwargs["color"]) #label="{} density = {:.1e} {}".format(gas, rho_opt*m**3, unit))
 


    def add_Phase_II_exposure_sens_line(self, sens):
        logger.warning("Adding Phase II sensitivity")
        sens.Experiment.number_density = 2.09e17/m**3
        sens.effective_volume = 1.2*mm**3
        sens.Experiment.sri_factor = 1 #0.389*0.918*0.32
        sens.Experiment.livetime = 7185228*s
        sens.CRLB_constant = 180
        
        standard_exposure = sens.effective_volume*sens.Experiment.livetime/m**3/year
        #sens.sensitivity()
        #sens.CL90(Experiment={"number_density": 2.09e17/m**3})/eV
        
        sens.print_systematics()
        sens.print_statistics()
        sens.print_SNRs()
        sens.print_Efficiencies()
        
        logger.info("Phase II sensitivity for exposure {} calculated: {}".format(standard_exposure, sens.sensitivity()/eV**2))
        
        # Phase II experimental results from frequentist analysis
        phaseIIsens = 9822
        phaseIIsense_error = 1520
        exposure_error = np.sqrt((standard_exposure*0.008)**2 + (standard_exposure*0.09)**2)
        
        self.ax.errorbar([standard_exposure], [phaseIIsens], yerr=[phaseIIsense_error], 
                         marker="None", color='k', linestyle="None", elinewidth=3,
                         label="Phase II (measured)")
        
        
        sigma_mbetas = []
        years = []
        for ex in self.exposures:
            lt = ex/sens.effective_volume
            years.append(lt/year)
            sens.Experiment.livetime = lt
            sigma_mbetas.append(sens.sensitivity()/eV**2)
            #exposures.append(sens.EffectiveVolume()/m**3*sens.Experiment.livetime/year)
            
            
        # unit = r"m$^{-3}$"
        # gas = r"T$_2$"
        self.ax.plot(self.exposures/m**3/year, sigma_mbetas, color='k', linestyle=':')#, label="{} density = {:.1e} {}".format(gas, 7.5e16, unit))
        
        def get_relative(val, axis):
            xmin, xmax = self.ax.get_xlim() if axis == "x" else self.ax.get_ylim()
            return (np.log10(val)-np.log10(xmin))/(np.log10(xmax)-np.log10(xmin))
        
        
        """x_start = get_relative(1e-3, "x")
        y_start = get_relative(6e1, "y")
        x_stop = get_relative(1e-4, "x")
        y_stop = get_relative(3e1, "y")"""
        
        x_start = get_relative(1e-7, "x")
        y_start = get_relative(7e3, "y")
        x_stop = get_relative(2e-8, "x")
        y_stop = get_relative(1.3e3, "y")
        self.ax.arrow(x_start, y_start, x_stop-x_start, y_stop-y_start,
                      transform = self.ax.transAxes,
                      facecolor = 'black',
                      edgecolor='k',
                      length_includes_head=True,
                      head_width=0.01,
                      head_length=0.01,
                      )

        self.ax.annotate("Phase II T$_2$ density \nand resolution", xy=[x_start*0.9, y_start*1.01],textcoords="axes fraction", fontsize=13)
             
    def add_frequency_sens_line(self, sens, plot_key_params=True, **kwargs):
        limits = []
        resolutions = []
        crlb_window = []
        crlb_max_window = []
        crlb_slope_zero_window = []
        total_volumes = []
        effective_volumes = []
        opt_rho = []
        noise_power = []
        
        configured_magnetic_field = sens.MagneticField.nominal_field
        
        gamma = sens.T_endpoint/(me*c0**2) + 1
        temp_rho = deepcopy(sens.Experiment.number_density)
        temp_field = deepcopy(sens.MagneticField.nominal_field)
        for freq in self.frequencies:
            magnetic_field = freq/(e/(2*np.pi*me)/gamma)
            sens.MagneticField.nominal_field = magnetic_field
            
            logger.info("Frequency: {:2e} Hz, Magnetic field: {:2e} T".format(freq/Hz, magnetic_field/T))
            
            # calcualte new cavity properties
            sens.CavityRadius()
            total_volumes.append(sens.CavityVolume()/m**3)
            effective_volumes.append(sens.EffectiveVolume()/m**3)
            sens.CavityPower()
            
            rho_limits = [sens.CL90(Experiment={"number_density": rho})/eV for rho in self.rhos]
            
            limits.append(np.min(rho_limits))
            this_optimum_rho = self.rhos[np.argmin(rho_limits)]
            if this_optimum_rho == self.rhos[0] or this_optimum_rho == self.rhos[-1]:
                raise ValueError("Cannot optimize density. Ideal value {:2e} at edge of range".format(this_optimum_rho*m**3))
            opt_rho.append(this_optimum_rho*m**3)
            sens.Experiment.number_density = this_optimum_rho
        
            # other quantities
            resolutions.append(sens.syst_frequency_extraction()[0]/meV)
            crlb_window.append(sens.best_time_window/ms)
            crlb_max_window.append(sens.time_window/ms)
            crlb_slope_zero_window.append(sens.time_window_slope_zero/ms)
            noise_power.append(sens.noise_temp/K)
            
        # set rho and f back
        sens.Experiment.number_density = temp_rho
        sens.MagneticField.nominal_field = temp_field
        sens.CavityRadius()
        sens.CavityPower()
            
        
        self.ax2.plot(self.frequencies/Hz, limits, **kwargs)
        logger.info('Minimum limit at {}: {}'.format(self.rhos[np.argmin(limits)]*m**3, np.min(limits)))
        
        # change cavity back to config file
        """sens.MagneticField.nominal_field = configured_magnetic_field
        sens.CavityRadius()
        sens.CavityVolume()
        sens.EffectiveVolume()
        sens.CavityPower()"""
        
        if self.make_key_parameter_plots and plot_key_params:
            self.kp_ax[0].plot(self.frequencies/Hz, resolutions, **kwargs)
            
            self.kp_ax[1].plot(self.frequencies/Hz, opt_rho, **kwargs)
            self.kp_ax[2].plot(self.frequencies/Hz, total_volumes, **kwargs)
            self.kp_ax[2].plot(self.frequencies/Hz, effective_volumes, linestyle="--", **kwargs)
            self.kp_ax[3].plot(self.frequencies/Hz, noise_power, linestyle="-", **kwargs)
        return limits  
      
    def add_text(self, x, y, text, color="k"): #, fontsize=9.5
        self.ax.text(x, y, text, color=color)

    def range(self, start, stop):
        cmap = matplotlib.cm.get_cmap('Spectral')
        norm = matplotlib.colors.Normalize(vmin=start, vmax=stop)
        return [(idx, cmap(norm(idx))) for idx in range(start, stop)]

    def save(self, savepath, **kwargs):
        logger.info("Saving")
        if self.density_axis:
            if self.track_length_axis:
                legend=self.fig.legend(loc=self.legend_location, framealpha=0.95, bbox_to_anchor=(0.15,0,1,0.85))
            else:
                legend=self.fig.legend(loc=self.legend_location, framealpha=0.95, bbox_to_anchor=(0.15,0,1,0.765))
        elif self.frequency_axis:
            if self.magnetic_field_axis:
                legend=self.fig.legend(loc=self.legend_location, framealpha=0.95, bbox_to_anchor=(0.14,0,1,0.85  ))
            else:
                legend=self.fig.legend(loc=self.legend_location, framealpha=0.95, bbox_to_anchor=(0.14,0,1,0.95  ))
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


