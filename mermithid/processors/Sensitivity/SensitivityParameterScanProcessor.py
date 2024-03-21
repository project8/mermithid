'''
Scan a parameter and calculate the sensitivity curve for each value of the parameter.
Author: C. Claessens
Date: 03/14/2024

More description
'''

from __future__ import absolute_import


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os



# Numericalunits is a package to handle units and some natural constants
# natural constants
from numericalunits import e, me, c0, eps0, kB, hbar
from numericalunits import meV, eV, keV, MeV, cm, m, mm
from numericalunits import nT, uT, mT, T, mK, K,  C, F, g, W
from numericalunits import hour, year, day, ms, ns, s, Hz, kHz, MHz, GHz
ppm = 1e-6
ppb = 1e-9
deg = np.pi/180

# morpho imports
from morpho.utilities import morphologging, reader
from morpho.processors import BaseProcessor
from mermithid.misc.SensitivityFormulas import Sensitivity
from mermithid.misc.SensitivityCavityFormulas import CavitySensitivity


logger = morphologging.getLogger(__name__)



__all__ = []
__all__.append(__name__)


def return_var_name(variable):
 for name in globals():
     if eval(name) == variable:
        return name



class SensitivityParameterScanProcessor(BaseProcessor):
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
        self.plot_path = reader.read_param(params, 'plot_path', "required")


        # options
        self.scan_parameter_name = reader.read_param(params, 'scan_parameter_name', 'MagneticField.sigmae_r') 
        self.scan_parameter_range = reader.read_param(params, "scan_parameter_range", [0.1, 2.1])
        self.scan_parameter_steps = reader.read_param(params, "scan_parameter_steps", 3)
        scan_parameter_unit = reader.read_param(params, "scan_parameter_unit", eV)
        
        
        self.scan_parameter_unit_string = return_var_name(scan_parameter_unit)
        print("Unit string: ", self.scan_parameter_unit_string)
        self.scan_parameter_unit = scan_parameter_unit

        # main plot configurations
        self.figsize = reader.read_param(params, 'figsize', (6,6))
        self.legend_location = reader.read_param(params, 'legend_location', 'upper left')
        self.fontsize = reader.read_param(params, 'fontsize', 12)
        self.plot_sensitivity_scan_on_log_scale = reader.read_param(params, 'plot_sensitivity_scan_on_log_scale', True)
        
        self.density_axis = reader.read_param(params, "density_axis", True)
        self.track_length_axis = reader.read_param(params, 'track_length_axis', True)
        self.atomic_axis = reader.read_param(params, 'atomic_axis', False)
        self.molecular_axis = reader.read_param(params, 'molecular_axis', False)
        
        self.density_range = reader.read_param(params, 'density_range', [1e14,1e21])
        self.ylim = reader.read_param(params, 'y_limits', [1e-2, 1e2])
        
        

        # key parameter plots
        self.make_key_parameter_plots = reader.read_param(params, 'plot_key_parameters', False)
        
        if self.density_axis:
            self.add_sens_line = self.add_density_sens_line
            logger.info("Doing density lines")



        # goals
        self.goals = reader.read_param(params, "goals", {})
        self.goal_x_pos = reader.read_param(params, "goals_x_position", 1e14)
        self.goals_y_rel_position = reader.read_param(params, "goals_y_rel_position", 0.75)


        # setup sensitivities
             
        self.cavity = reader.read_param(params, 'cavity', True)
        
        if self.cavity:
            self.sens_main = CavitySensitivity(self.config_file_path)
        else:
            self.sens_main = Sensitivity(self.config_file_path)
        self.sens_main_is_atomic = self.sens_main.Experiment.atomic


        # check atomic and molecular
        if self.molecular_axis:
            if not self.sens_main_is_atomic:
                #self.molecular_sens = self.sens_main
                logger.info("Main curve is molecular")


        if self.atomic_axis:
            if self.sens_main_is_atomic:
                #self.atomic_sens = self.sens_main
                logger.info("Main curve is atomic")
            else:
                logger.warn("No experiment is configured to be atomic")

        # densities, exposures, runtimes
        self.rhos = np.logspace(np.log10(self.density_range[0]), np.log10(self.density_range[1]), 100)/m**3
        self.scan_parameter_values = np.linspace(self.scan_parameter_range[0], self.scan_parameter_range[1], self.scan_parameter_steps)*self.scan_parameter_unit
         
        return True



    def InternalRun(self):
        
        self.create_plot(self.scan_parameter_values/self.scan_parameter_unit)
        # add second and third x axis for track lengths
        if self.density_axis and self.track_length_axis:
            self.add_track_length_axis()
            
        # add goals to density plot
        for key, value in self.goals.items():
            logger.info('Adding goal: {} = {}'.format(key, value))
            self.add_goal(value, key)
                
        
        self.optimum_limits = []
        self.optimum_rhos = []


        for i, color in self.range(self.scan_parameter_values/self.scan_parameter_unit):
            parameter_value = self.scan_parameter_values[i]
            
            category, param = self.scan_parameter_name.split(".")
            
            self.sens_main.__dict__[category].__dict__[param] = parameter_value 
            read_back = self.sens_main.__dict__[category].__dict__[param]
            #setattr(self.sens_main, self.scan_parameter_name, parameter_value)
            #read_back = getattr(self.sens_main, self.scan_parameter_name)
            logger.info(f"Setting {self.scan_parameter_name} to {parameter_value/self.scan_parameter_unit} and reading back: {read_back/ self.scan_parameter_unit}")
            
    
            logger.info("Calculating cavity experiment")   
            self.sens_main.CavityVolume()
            self.sens_main.EffectiveVolume()
            self.sens_main.CavityPower()
            
            # optimize density
            logger.info("Optimizing density")
            limit = [self.sens_main.CL90(Experiment={"number_density": rho}) for rho in self.rhos]
            self.optimum_limits.append(np.min(limit))
            opt = np.argmin(limit)
            rho_opt = self.rhos[opt]
            self.optimum_rhos.append(rho_opt)
            self.sens_main.Experiment.number_density = rho_opt
              
            # add main curve
            logger.info("Drawing main curve")  
            label = f"{param} = {parameter_value/self.scan_parameter_unit:.2f} {self.scan_parameter_unit_string}"
            self.add_sens_line(self.sens_main, label=label, color=color)
                 
            if self.make_key_parameter_plots:
                logger.info("Making key parameter plots")
                # First key parameter plot: Stat and Syst vs. density
                
                sigma_startf, stat_on_mbeta2, syst_on_mbeta2 = [], [], []

                for n in self.rhos:
                    self.sens_main.Experiment.number_density = n
                    labels, sigmas, deltas = self.sens_main.get_systematics()
                    sigma_startf.append(sigmas[1])
                    stat_on_mbeta2.append(self.sens_main.StatSens())
                    syst_on_mbeta2.append(self.sens_main.SystSens())
                        
                sigma_startf, stat_on_mbeta2, syst_on_mbeta2 = np.array(sigma_startf), np.array(stat_on_mbeta2), np.array(syst_on_mbeta2)
                fig = plt.figure()
                plt.loglog(self.rhos*m**3, stat_on_mbeta2/eV**2, label='Statistical uncertainty')
                plt.loglog(self.rhos*m**3, syst_on_mbeta2/eV**2, label='Systematic uncertainty')
                plt.xlabel(r"Number density $n\, \, (\mathrm{m}^{-3})$")
                plt.ylabel(r"Standard deviation in $m_\beta^2$ (eV$^2$)")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(self.plot_path, f"{param}_{parameter_value/self.scan_parameter_unit}_stat_and_syst_vs_density.pdf"))

                




            logger.info('Experiment info:')
            # set optimum density back
            self.sens_main.CL90(Experiment={"number_density": rho_opt})
            logger.info('veff = {} m**3, rho = {} /m**3:'.format(self.sens_main.effective_volume/(m**3), rho_opt*(m**3)))
            logger.info("Loaded Q: {}".format(self.sens_main.loaded_q))
            logger.info("Axial frequency for minimum detectable angle: {} MHz".format(self.sens_main.required_bw_axialfrequency/MHz))
            logger.info("Total bandwidth: {} MHz".format(self.sens_main.required_bw/MHz))
            logger.info('Larmor power = {} W, Hanneke power = {} W'.format(self.sens_main.larmor_power/W, self.sens_main.signal_power/W))
            logger.info('Hanneke / Larmor power = {}'.format(self.sens_main.signal_power/self.sens_main.larmor_power))
            
            if self.sens_main.FrequencyExtraction.crlb_on_sidebands:
                logger.info("Uncertainty of frequency resolution and energy reconstruction (for pitch angle): {} eV, {} eV".format(self.sens_main.sigma_K_f_CRLB/eV, self.sens_main.sigma_K_reconstruction/eV))
        
            self.sens_main.print_SNRs(rho_opt)
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
            
        self.save("sensitivity_vs_density_for_{}_scan.pdf".format(param))
            

        # plot and print best limits
        self.results = {"scan_parameter": self.scan_parameter_name, "scan parameter_unit": self.scan_parameter_unit_string,
                        "scan_parameter_values": self.scan_parameter_values, "optimum_limits_eV": np.array(self.optimum_limits)/eV,
                        "optimum_densities_/m3": np.array(self.optimum_rhos)*(m**3)}
        
        logger.info("Scan parameter: {}".format(self.scan_parameter_name))
        logger.info("Tested parameter values: {}".format(self.scan_parameter_values/self.scan_parameter_unit))
        logger.info("Best limits: {}".format(np.array(self.optimum_limits)/eV))
        
        plt.figure(figsize=self.figsize)
        #plt.title("Sensitivity vs. {}".format(self.scan_parameter_name))
        plt.plot(self.scan_parameter_values/self.scan_parameter_unit, np.array(self.optimum_limits)/eV, marker=".", label="Density optimized scenarios")
        plt.xlabel(f"{param} ({self.scan_parameter_unit_string})", fontsize=self.fontsize)
        plt.ylabel(r"90% CL $m_\beta$ (eV)", fontsize=self.fontsize)
        if self.plot_sensitivity_scan_on_log_scale:
            plt.yscale("log")
            
        for key, value in self.goals.items():
            logger.info('Adding goal: {} = {}'.format(key, value))
            plt.axhline(value, label=key, color="grey", linestyle="--")
        plt.legend(fontsize=self.fontsize)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_path, f"{param}_scan_optimum_limits.pdf"))
        plt.show()
        
        
        
       
  

        return True


    def create_plot(self, param_range=[]):
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
                axis_label = r"(Atomic) number density $n\, \, (\mathrm{m}^{-3})$"
            elif self.molecular_axis:
                axis_label = r"(Molecular) number density $n\, \, (\mathrm{m}^{-3})$"
            else:
                axis_label = r"Number density $n\, \, (\mathrm{m}^{-3})$"
                
            ax.set_xlabel(axis_label)
            ax.set_ylim(self.ylim)
            ax.set_ylabel(r"90% CL $m_\beta$ (eV)")
            
        if len(param_range)>4:
            # add colorbar with colors from self.range
            cmap = matplotlib.cm.get_cmap('Spectral')
            norm = matplotlib.colors.Normalize(vmin=np.min(param_range), vmax=np.max(param_range))
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            self.fig.colorbar(sm, ticks=np.round(param_range, 2), label=f"{self.scan_parameter_name} ({self.scan_parameter_unit_string})")
            
        

                    
    def add_track_length_axis(self):
       
        if self.atomic_axis:

            ax2 = self.ax.twiny()
            ax2.set_xscale("log")
            ax2.set_xlabel("(Atomic) track length (s)")
            ax2.set_xlim(self.sens_main.track_length(self.rhos[0])/s,
                         self.sens_main.track_length(self.rhos[-1])/s)

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

        if not self.atomic_axis and not self.molecular_axis:
            logger.warning("No track length axis added since neither atomic nor molecular was requested")
        self.fig.tight_layout()
        
    

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
        rho_opt = self.rhos[np.argmin(limits)]
        self.sens_main.Experiment.number_density = rho_opt
        logger.info('Minimum limit at {}: {}'.format(rho_opt*m**3, np.min(limits)))
        
        if self.make_key_parameter_plots and plot_key_params:
            self.kp_ax[0].plot(self.rhos*m**3, resolutions, **kwargs)
            
            self.kp_ax[1].plot(self.rhos*m**3, crlb_max_window, color='red', marker='.')
            self.kp_ax[1].plot(self.rhos*m**3, crlb_slope_zero_window, color='green', marker='.')
            self.kp_ax[1].plot(self.rhos*m**3, crlb_window, linestyle="--", marker='.', **kwargs)
        return limits
    
 
      
    def add_text(self, x, y, text, color="k"): #, fontsize=9.5
        self.ax.text(x, y, text, color=color)

    def range(self, param_range):
        cmap = matplotlib.cm.get_cmap('Spectral')
        norm = matplotlib.colors.Normalize(vmin=0, vmax=len(param_range)-1)
        return [(idx, cmap(norm(idx))) for idx, _ in enumerate(param_range)]

    def save(self, filename, **kwargs):
        
        if self.density_axis:
            if self.scan_parameter_steps < 5:
                legend=self.fig.legend(loc=self.legend_location, framealpha=0.95, bbox_to_anchor=(0.15,0,1,0.765))
            
        else:
            legend=self.fig.legend(loc=self.legend_location, framealpha=0.95, bbox_to_anchor=(-0.,0,0.89,0.97))

            
            
        #keywords = ", ".join(["%s=%s"%(key, value) for key, value in kwargs.items()])
        metadata = {"Author": "p8/mermithid",
                    "Title": "Neutrino mass sensitivity",
                    "Subject":"90% CL upper limit on neutrino mass assuming true mass is zero."
                    }
                    #"Keywords": keywords}
                    
        self.fig.tight_layout()
        self.fig.savefig(os.path.join(self.plot_path, filename), bbox_inches="tight", metadata=metadata)
        self.fig.savefig(os.path.join(self.plot_path, filename.replace(".pdf", ".png")), bbox_inches="tight", metadata=metadata)
            


