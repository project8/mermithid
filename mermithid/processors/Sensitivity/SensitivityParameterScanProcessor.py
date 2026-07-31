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
import re
import concurrent.futures



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
from mermithid.sensitivity.SensitivityFormulas import Sensitivity, NameSpace
from mermithid.sensitivity.SensitivityCavityFormulas import CavitySensitivity


logger = morphologging.getLogger(__name__)



__all__ = []
__all__.append(__name__)


def return_var_name(variable):
    for name in globals():
        if eval(name) == variable:
            return name


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------

import numericalunits as _nu

# Namespace used to interpret unit strings such as "meV", "m^3", "1/s".
# Mirrors what the .cfg files are allowed to use, so unit strings and config
# values follow the same conventions.
_UNIT_NAMESPACE = {k: v for k, v in vars(_nu).items() if not k.startswith("_")}
_UNIT_NAMESPACE.update({"np": np, "pi": np.pi, "deg": deg, "ppm": ppm, "ppb": ppb})

_DIMENSIONLESS_STRINGS = ("", "1", "-", "none", "dimensionless", "unitless")


def resolve_unit(spec, context=""):
    """Turn a unit specification into a (divisor, label) pair.

    Accepts either
      * a string, evaluated against numericalunits, e.g. "meV", "m^3", "1/(eV*s)".
        Both "^" and "**" work for powers.
      * a numericalunits value, e.g. eV (kept for backwards compatibility).
      * None or 1, meaning dimensionless.

    Anything that cannot be interpreted is treated as dimensionless with a
    warning, so a typo in a unit never aborts a scan.

    Note on already-scaled quantities: several mermithid methods return numbers
    that have *already* been divided by their unit (print_systematics returns
    meV, accumulated_activity_* is in Ci). Those must be declared dimensionless
    here, otherwise the unit gets divided out twice.
    """
    if spec is None:
        return 1.0, ""

    if isinstance(spec, str):
        text = spec.strip()
        if text.lower() in _DIMENSIONLESS_STRINGS:
            return 1.0, ""
        try:
            value = eval(text.replace("^", "**"), {"__builtins__": {}}, dict(_UNIT_NAMESPACE))
        except Exception as err:
            logger.warning("Could not interpret unit '{}'{}: {}. Treating as dimensionless.".format(
                spec, context, err))
            return 1.0, ""
        if not np.isscalar(value) or value == 0:
            logger.warning("Unit '{}'{} did not evaluate to a non-zero scalar. "
                           "Treating as dimensionless.".format(spec, context))
            return 1.0, ""
        return float(value), text

    # Numeric unit (legacy style: "scan_parameter_unit": eV)
    value = float(spec)
    if value == 1.0:
        # "unit = 1" means dimensionless. Looking the value up by name returns
        # None (or something arbitrary), which is where the old "(None)" axis
        # labels came from.
        return 1.0, ""
    return value, (return_var_name(spec) or "")


# ---------------------------------------------------------------------------
# Attribute paths
# ---------------------------------------------------------------------------

def _split_token(token):
    """Split a single path element into (name, subscripts, is_call).

    'loaded_q'      -> ('loaded_q', [], False)
    'q_array[2]'    -> ('q_array', ['2'], False)
    'DeltaEWidth()' -> ('DeltaEWidth', [], True)

    Raises ValueError on anything malformed, which is what lets
    configure_diagnostics reject a bad name at startup rather than part way
    through a scan.
    """
    token = token.strip()

    is_call = token.endswith("()")
    if is_call:
        token = token[:-2]

    indices = []
    while token.endswith("]"):
        start = token.rfind("[")
        if start == -1:
            raise ValueError("Unbalanced brackets in path element '{}'".format(token))
        indices.insert(0, token[start + 1:-1])
        token = token[:start]

    if is_call and indices:
        # The resolver calls before it indexes, so this spelling would silently
        # evaluate as name()[i]. Reject it rather than guess.
        raise ValueError("Cannot combine subscripts and a call in '{}'".format(token))

    if not token.isidentifier():
        raise ValueError("Cannot parse path element '{}'".format(token))

    return token, indices, is_call


def get_by_path(root, path):
    """Resolve a dotted attribute path on an object.

    Supports plain attributes ("loaded_q"), attributes of the config
    namespaces ("Experiment.number_density"), no-argument method calls
    ("DeltaEWidth()", "SignalRatio()") and subscripts ("q_array[0]").

    Uses getattr rather than __dict__, which matters because the config
    sections are NameSpace objects whose __getattribute__ lowercases every
    lookup: __dict__ access fails on any name that is not already lowercase.
    """
    obj = root
    for token in str(path).split("."):
        name, indices, is_call = _split_token(token)
        obj = getattr(obj, name)
        if is_call:
            obj = obj()
        for index in indices:
            obj = obj[int(index)]
    return obj


def set_by_path(root, path, value):
    """Set a dotted attribute path on an object, and return the value read back.

    NameSpace overrides __getattribute__ (lowercasing the name) but not
    __setattr__. Setting a mixed-case name directly would therefore write a key
    that can never be read back, silently leaving the original value in place.
    Names on NameSpace targets are lowercased here to avoid that.
    """
    tokens = str(path).split(".")
    parent = root if len(tokens) == 1 else get_by_path(root, ".".join(tokens[:-1]))
    name, indices, is_call = _split_token(tokens[-1])
    if is_call or indices:
        raise ValueError("Cannot assign to path '{}'".format(path))
    if isinstance(parent, NameSpace):
        name = name.lower()
    setattr(parent, name, value)
    return getattr(parent, name)



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
        self.scan_parameter_scale = reader.read_param(params, "scan_parameter_scale", "lin")
        scan_parameter_unit = reader.read_param(params, "scan_parameter_unit", eV)

        # Accepts either a numericalunits value (eV) or a string ("eV", "m^3", "1")
        self.scan_parameter_unit, self.scan_parameter_unit_string = resolve_unit(
            scan_parameter_unit, context=" for scan_parameter_unit")
        logger.info("Scan parameter unit: {}".format(self.scan_parameter_unit_string or "dimensionless"))

        # Short name of the scanned parameter, used for labels and file names
        self.scan_parameter_short_name = self.scan_parameter_name.split(".")[-1]

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

        # diagnostic parameters: extra quantities of the sensitivity object that
        # are recorded at the density-optimized working point of every scan step
        self.diagnostic_parameters = reader.read_param(params, 'diagnostic_parameters', [])
        self.plot_diagnostics = reader.read_param(params, 'plot_diagnostics', True)
        self.save_diagnostics = reader.read_param(params, 'save_diagnostics', True)
        self.combine_diagnostic_plots = reader.read_param(params, 'combine_diagnostic_plots', False)
        self.diagnostics_file_prefix = reader.read_param(params, 'diagnostics_file_prefix', 'diagnostics')
        self.diagnostics = self.configure_diagnostics(self.diagnostic_parameters)
        self.diagnostics_warned = set()
        self.diagnostics_needing_providers = set()
        self.diagnostic_values = {spec["key"]: [] for spec in self.diagnostics}
        
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


        # Setting natoms_per_particle for atomic and molecular cases, to use later
        # when calculating and printing event rates
        if self.sens_main_is_atomic:
            self.sens_main_natoms_per_particle = 1
        else:
            self.sens_main_natoms_per_particle = 2


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
        if(self.scan_parameter_scale == "lin"):
            self.scan_parameter_values = np.linspace(self.scan_parameter_range[0], self.scan_parameter_range[1], self.scan_parameter_steps)*self.scan_parameter_unit
        elif(self.scan_parameter_scale == "log"):
            self.scan_parameter_values = np.logspace(np.log10(self.scan_parameter_range[0]), np.log10(self.scan_parameter_range[1]), self.scan_parameter_steps)*self.scan_parameter_unit
        else:
            logger.warn("Unexpected parameter scale, assuming linear")
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
                
        # arrays for results values for output
        self.optimum_limits = []
        self.optimum_rhos = []
        
        self.noise_temp = []
        self.SNR = []
        self.track_duration = []
        self.total_sigma = []
        self.sys_lim = []

        # arrays for the diagnostic parameters (kept separate from the results above)
        self.diagnostic_values = {spec["key"]: [] for spec in self.diagnostics}

        for i, color in self.range(self.scan_parameter_values/self.scan_parameter_unit):
            parameter_value = self.scan_parameter_values[i]
            
            param = self.scan_parameter_short_name

            # read current value of param
            try:
                current_value = get_by_path(self.sens_main, self.scan_parameter_name)
                logger.info(f"Current value of {param}: {current_value/self.scan_parameter_unit}")
            except AttributeError as e:
                logger.error(f"Parameter {self.scan_parameter_name} not found on the sensitivity object")
                raise e

            # Set to scan param value and verify that it actually took effect
            read_back = set_by_path(self.sens_main, self.scan_parameter_name, parameter_value)
            logger.info(f"Setting {self.scan_parameter_name} to {parameter_value/self.scan_parameter_unit} and reading back: {read_back/self.scan_parameter_unit}")
            if read_back != parameter_value:
                logger.warning(f"{self.scan_parameter_name} did not take the requested value")

            # pitch angle set equal
            if(param == "min_pitch_used_in_analysis"):
                set_by_path(self.sens_main, "FrequencyExtraction.minimum_angle_in_bandwidth", parameter_value)
            
            # DEPRECATED: If the scanned param isn't trap length, calc trap length for cavity L/D
#            if (param != "trap_length"):
#                self.sens_main.TrapLength() # method no longer exist in CavitySensitivity
            
            logger.info("Calculating cavity experiment radius, volume, effective volume, power") 
            self.sens_main.CavityRadius()  
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
              
            # add main curve
            logger.info("Drawing main curve")  
            label = f"{param} = {parameter_value/self.scan_parameter_unit:.2f} {self.scan_parameter_unit_string}".strip()
            self.add_sens_line(self.sens_main, label=label, color=color)
                 
            if self.make_key_parameter_plots:
                logger.info("Making key parameter plots")
                # First key parameter plot: Stat and Syst vs. density
                
                sigma_startf, stat_on_mbeta2, syst_on_mbeta2 = [], [], []

                for n in self.rhos:
                    self.sens_main.CL90(Experiment={"number_density": n})
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
                logger.info("Uncertainty from determination of f_carrier and f_lsb, due to noise: {} eV".format(self.sens_main.sigma_K_noise/eV))
       
            noise_temp, SNR, track_duration = self.sens_main.print_SNRs(rho_opt)
            # Store relevant values
            self.noise_temp.append(noise_temp/K)
            self.SNR.append(SNR)
            self.track_duration.append(track_duration/ms)
        
            logger.info('CL90 limit: {}'.format(self.sens_main.CL90(Experiment={"number_density": rho_opt})/eV))
            logger.info('T2 in Veff: {}'.format(rho_opt*self.sens_main.effective_volume))
            logger.info('Total signal: {}'.format(rho_opt*self.sens_main.effective_volume*
                                                    self.sens_main.Experiment.LiveTime/
                                                    self.sens_main.tau_tritium*self.sens_main_natoms_per_particle))
            logger.info('Signal in last eV: {}'.format(self.sens_main.last_1ev_fraction*eV**3*
                                                    rho_opt*self.sens_main.effective_volume*
                                                    self.sens_main.Experiment.LiveTime/
                                                    self.sens_main.tau_tritium*self.sens_main_natoms_per_particle))

            self.sens_main.print_Efficiencies()
            self.sens_main.print_statistics()
            systematic_limit, total_sigma = self.sens_main.print_systematics()
            self.sys_lim.append(systematic_limit)
            self.total_sigma.append(total_sigma) 

            # diagnostics: the object is at the optimum density here, since CL90
            # was last evaluated with number_density = rho_opt
            self.store_diagnostics()

        self.save("sensitivity_vs_density_for_{}_scan.pdf".format(param))
            

        # plot and print best limits
        self.results = {"scan_parameter": self.scan_parameter_name, "scan parameter_unit": self.scan_parameter_unit_string,
                        "scan_parameter_values": self.scan_parameter_values, "optimum_limits_eV": np.array(self.optimum_limits)/eV,
                        "optimum_densities/m3": np.array(self.optimum_rhos)*(m**3),
                        "Noise Temperatures/K": np.array(self.noise_temp),
                        "SNRs 1eV from temperature": np.array(self.SNR), "track durations": np.array(self.track_duration),
                        "Systematic limits": np.array(self.sys_lim), "Total Sigmas": np.array(self.total_sigma)}

        results_array = [np.array(self.scan_parameter_values/self.scan_parameter_unit),np.array(self.noise_temp),np.array(self.SNR),
                         np.array(self.optimum_rhos)*(m**3),np.array(self.track_duration),np.array(self.total_sigma),
                         1000*np.array(self.optimum_limits)/eV,np.array(self.sys_lim)]
        fmt_array = '%.2g','%.3f','%.1f','%.2E','%.2f','%.1f','%.1f','%.1f'
        header_string = 'Param Value, Noise temperature /K, SNR, Gas Density /m3, Track Length /ms, Resolution, Sensitivity /meV, Systematic Limit /meV'
        np.savetxt("results_array_{}.csv".format(param),np.array(results_array).T,delimiter=',',fmt=fmt_array,header=header_string)        
        logger.info("Scan parameter: {}".format(self.scan_parameter_name))
        logger.info("Tested parameter values: {}".format(self.scan_parameter_values/self.scan_parameter_unit))
        logger.info("Best limits: {}".format(np.array(self.optimum_limits)/eV))

        plt.figure(figsize=self.figsize)
        #plt.title("Sensitivity vs. {}".format(self.scan_parameter_name))
        plt.plot(self.scan_parameter_values/self.scan_parameter_unit, np.array(self.optimum_limits)/eV, marker=".", label="Density optimized scenarios")
        plt.xlabel(self.parameter_axis_label(), fontsize=self.fontsize)
        plt.ylabel(r"90% CL on $m_\beta$ (eV)", fontsize=self.fontsize)
        if self.plot_sensitivity_scan_on_log_scale:
            plt.yscale("log")
        # TODO log x here    
        if(self.scan_parameter_scale == "log"):
            plt.xscale("log")
        for key, value in self.goals.items():
            logger.info('Adding goal: {} = {}'.format(key, value))
            plt.axhline(value, label=key, color="grey", linestyle="--")
        plt.legend(fontsize=self.fontsize)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_path, f"{param}_scan_optimum_limits.pdf"))
        plt.show()

        # diagnostic parameters: own dict, own csv, own figures
        self.diagnostic_results = {"scan_parameter": self.scan_parameter_name,
                                   "scan_parameter_unit": self.scan_parameter_unit_string,
                                   "scan_parameter_values": np.array(self.scan_parameter_values)/self.scan_parameter_unit}
        for spec in self.diagnostics:
            key = spec["name"] + (" [{}]".format(spec["unit_string"]) if spec["unit_string"] else "")
            self.diagnostic_results[key] = np.array(self.diagnostic_values[spec["key"]], dtype=float)

        if self.diagnostics and self.save_diagnostics:
            self.save_diagnostics_csv()
        if self.diagnostics and self.plot_diagnostics:
            self.plot_diagnostics_vs_parameter()
        
        
        
       
  

        return True


    # -----------------------------------------------------------------------
    # DIAGNOSTIC PARAMETERS
    # -----------------------------------------------------------------------

    # Methods that populate attributes which are not touched by the CL90 call
    # chain (the atomic-calculator quantities: trap lifetime, atom currents,
    # pumping speeds, accumulated activities, ...). They are only invoked if a
    # requested diagnostic cannot be read without them. Order matters:
    # print_pumping_requirements consumes T2_total_density from the first one.
    diagnostic_provider_methods = ["print_T2_background_atomic_trap",
                                   "print_pumping_requirements"]

    def configure_diagnostics(self, specs):
        """Normalize the diagnostic_parameters configuration.

        Each entry is either a plain name string or a dict:
            {"name": "DeltaEWidth()", "unit": "eV", "label": "Energy window",
             "log": False, "fmt": "%.4g", "index": 0, "aggregate": "mean"}
        """
        diagnostics = []
        if specs is None:
            specs = []
        if isinstance(specs, (str, dict)):
            specs = [specs]

        for spec in specs:
            if isinstance(spec, str):
                spec = {"name": spec}
            if not isinstance(spec, dict):
                raise ValueError("Diagnostic parameter entries must be strings or dicts, "
                                 "got {}".format(type(spec)))
            name = spec.get("name")
            if not name:
                raise ValueError("Diagnostic parameter entry is missing 'name': {}".format(spec))

            # Fail early on unparseable paths rather than mid-scan
            for token in str(name).split("."):
                _split_token(token)

            unit_value, unit_label = resolve_unit(spec.get("unit"),
                                                  context=" for diagnostic '{}'".format(name))
            key = re.sub(r"[^A-Za-z0-9]+", "_", str(name)).strip("_")
            diagnostics.append({"name": name,
                                "key": key,
                                "unit": unit_value,
                                "unit_string": unit_label,
                                "label": spec.get("label", str(name)),
                                "log": spec.get("log", False),
                                "fmt": spec.get("fmt", "%.4g"),
                                "index": spec.get("index", None),
                                "aggregate": spec.get("aggregate", None)})

        if diagnostics:
            logger.info("Configured {} diagnostic parameter(s): {}".format(
                len(diagnostics), ", ".join(d["name"] for d in diagnostics)))
        return diagnostics

    def run_diagnostic_providers(self):
        """Call the provider methods so late-computed attributes exist."""
        for method_name in self.diagnostic_provider_methods:
            method = getattr(self.sens_main, method_name, None)
            if method is None:
                logger.warning("Sensitivity object has no method {}".format(method_name))
                continue
            try:
                method()
            except Exception as err:
                logger.warning("Call to {} failed: {}".format(method_name, err))

    def read_diagnostic(self, spec):
        """Read one diagnostic. Returns (raw_value, error)."""
        try:
            return get_by_path(self.sens_main, spec["name"]), None
        except Exception as err:
            return None, err

    def scale_diagnostic(self, spec, raw):
        """Reduce a raw value to a single float in the requested unit."""
        if raw is None:
            # e.g. SignalRatio() returns None when T2_background_atomic_trap is off
            return np.nan

        value = raw
        if spec["index"] is not None:
            try:
                value = value[spec["index"]]
            except Exception as err:
                self.warn_once(spec, "could not apply index {}: {}".format(spec["index"], err))
                return np.nan
        elif spec["aggregate"] is not None:
            reducers = {"mean": np.mean, "max": np.max, "min": np.min, "sum": np.sum}
            reducer = reducers.get(spec["aggregate"])
            if reducer is None:
                self.warn_once(spec, "unknown aggregate '{}'".format(spec["aggregate"]))
                return np.nan
            value = reducer(value)

        try:
            value = np.asarray(value, dtype=float)
        except (TypeError, ValueError) as err:
            self.warn_once(spec, "is not numeric ({}): {}".format(type(raw).__name__, err))
            return np.nan
        if value.size != 1:
            self.warn_once(spec, "is not a scalar (shape {}). Use 'index' or "
                                 "'aggregate' to reduce it.".format(value.shape))
            return np.nan

        return float(value) / spec["unit"]

    def warn_once(self, spec, message):
        if spec["key"] not in self.diagnostics_warned:
            self.diagnostics_warned.add(spec["key"])
            logger.warning("Diagnostic '{}' {}".format(spec["name"], message))

    def evaluate_diagnostics(self):
        """Record all diagnostics for the current state of the sensitivity object.

        Call this with the object at the working point of interest (in the scan
        loop: after CL90 has been re-evaluated at the optimum density), since
        most of these quantities are side effects of the last calculation.
        """
        if not self.diagnostics:
            return {}

        # Attributes known to need a provider call are refreshed every step;
        # without this, later steps would silently re-read the first step's value.
        providers_ran = False
        if self.diagnostics_needing_providers:
            self.run_diagnostic_providers()
            providers_ran = True

        values = {}
        for spec in self.diagnostics:
            raw, err = self.read_diagnostic(spec)

            if err is not None and not providers_ran:
                # Might just not have been computed yet: run the providers once
                # and retry before giving up on it.
                logger.info("Diagnostic '{}' not available yet, running provider "
                            "methods".format(spec["name"]))
                self.run_diagnostic_providers()
                providers_ran = True
                raw, retry_err = self.read_diagnostic(spec)
                if retry_err is None:
                    self.diagnostics_needing_providers.add(spec["key"])
                    err = None
                else:
                    err = retry_err

            if err is not None:
                self.warn_once(spec, "could not be read: {}. Storing NaN.".format(err))
                raw = None

            values[spec["key"]] = self.scale_diagnostic(spec, raw)

        return values

    def store_diagnostics(self):
        """Evaluate the diagnostics and append them to the per-step arrays."""
        values = self.evaluate_diagnostics()
        for spec in self.diagnostics:
            value = values.get(spec["key"], np.nan)
            self.diagnostic_values[spec["key"]].append(value)
            logger.info("Diagnostic {} = {:.6g} {}".format(
                spec["name"], value, spec["unit_string"] or ""))

    def diagnostic_label(self, spec):
        if spec["unit_string"]:
            return "{} ({})".format(spec["label"], spec["unit_string"])
        return spec["label"]

    def parameter_axis_label(self, name=None):
        name = name or self.scan_parameter_short_name
        if self.scan_parameter_unit_string:
            return "{} ({})".format(name, self.scan_parameter_unit_string)
        return name

    def save_diagnostics_csv(self):
        """Write the diagnostics to their own csv file, separate from the results."""
        if not self.diagnostics:
            return None

        param = self.scan_parameter_short_name
        x = np.array(self.scan_parameter_values) / self.scan_parameter_unit
        columns = [x]
        formats = ['%.6g']
        headers = ["{}{}".format(param, " [{}]".format(self.scan_parameter_unit_string)
                                   if self.scan_parameter_unit_string else "")]

        for spec in self.diagnostics:
            columns.append(np.array(self.diagnostic_values[spec["key"]], dtype=float))
            formats.append(spec["fmt"])
            headers.append("{}{}".format(spec["name"], " [{}]".format(spec["unit_string"])
                                         if spec["unit_string"] else ""))

        filename = os.path.join(self.plot_path,
                                "{}_{}_scan.csv".format(self.diagnostics_file_prefix, param))
        np.savetxt(filename, np.array(columns).T, delimiter=',',
                   fmt=tuple(formats), header=', '.join(headers))
        logger.info("Wrote diagnostics to {}".format(filename))
        return filename

    def plot_diagnostics_vs_parameter(self):
        """One figure per diagnostic: diagnostic vs. scanned parameter."""
        if not self.diagnostics:
            return

        param = self.scan_parameter_short_name
        x = np.array(self.scan_parameter_values) / self.scan_parameter_unit

        for spec in self.diagnostics:
            y = np.array(self.diagnostic_values[spec["key"]], dtype=float)
            if np.all(np.isnan(y)):
                logger.warning("Diagnostic '{}' is NaN at every scan point, "
                               "skipping its plot".format(spec["name"]))
                continue

            plt.figure(figsize=self.figsize)
            plt.plot(x, y, marker=".")
            plt.xlabel(self.parameter_axis_label(), fontsize=self.fontsize)
            plt.ylabel(self.diagnostic_label(spec), fontsize=self.fontsize)
            if self.scan_parameter_scale == "log":
                plt.xscale("log")
            if spec["log"]:
                plt.yscale("log")
            plt.tight_layout()
            filename = os.path.join(self.plot_path, "{}_scan_diagnostic_{}.pdf".format(
                param, spec["key"]))
            plt.savefig(filename)
            plt.close()
            logger.info("Saved {}".format(filename))

        if self.combine_diagnostic_plots:
            self.plot_diagnostics_combined(x)

    def plot_diagnostics_combined(self, x):
        """All diagnostics in one multi-panel figure sharing the parameter axis."""
        usable = [s for s in self.diagnostics
                  if not np.all(np.isnan(np.array(self.diagnostic_values[s["key"]], dtype=float)))]
        if not usable:
            return

        n_columns = int(np.ceil(np.sqrt(len(usable))))
        n_rows = int(np.ceil(len(usable) / n_columns))
        fig, axes = plt.subplots(n_rows, n_columns, sharex=True,
                                 figsize=(4.0 * n_columns, 3.0 * n_rows), squeeze=False)
        flat_axes = axes.flatten()

        for ax, spec in zip(flat_axes, usable):
            ax.plot(x, np.array(self.diagnostic_values[spec["key"]], dtype=float), marker=".")
            ax.set_ylabel(self.diagnostic_label(spec), fontsize=self.fontsize - 2)
            if self.scan_parameter_scale == "log":
                ax.set_xscale("log")
            if spec["log"]:
                ax.set_yscale("log")
        for ax in flat_axes[len(usable):]:
            ax.set_visible(False)
        for ax in axes[-1]:
            if ax.get_visible():
                ax.set_xlabel(self.parameter_axis_label(), fontsize=self.fontsize - 2)

        fig.tight_layout()
        filename = os.path.join(self.plot_path, "{}_scan_diagnostics_overview.pdf".format(
            self.scan_parameter_short_name))
        fig.savefig(filename)
        plt.close(fig)
        logger.info("Saved {}".format(filename))

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
            ax.set_ylabel(r"90% CL on $m_\beta$ (eV)")
            
        if len(param_range)>4:
            # add colorbar with colors from self.range
            cmap = matplotlib.cm.get_cmap('Spectral')
            norm = matplotlib.colors.Normalize(vmin=np.min(param_range), vmax=np.max(param_range))
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            self.fig.colorbar(sm, ticks=np.round(param_range, 2), label=self.parameter_axis_label(self.scan_parameter_name))
            
        

                    
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
        track_durations = []

        # The key parameter quantities used to be collected for every density even
        # when the plots were switched off. They are only gathered on demand now:
        # best_time_window and time_window_slope_zero no longer exist (the CRLB
        # model samples a distribution of track durations instead), and
        # time_window is an array rather than a scalar.
        collect_key_params = self.make_key_parameter_plots and plot_key_params

        for rho in self.rhos:
            limits.append(sens.CL90(Experiment={"number_density": rho})/eV)
            if collect_key_params:
                resolutions.append(sens.sigma_K_noise/meV)
                track_durations.append(np.mean(np.atleast_1d(sens.time_window))/ms)

        self.ax.plot(self.rhos*m**3, limits, **kwargs)
        rho_opt = self.rhos[np.argmin(limits)]
        # set experiment to optimum density
        sens.CL90(Experiment={"number_density": rho_opt})
        logger.info('Minimum limit at {}: {}'.format(rho_opt*m**3, np.min(limits)))

        if collect_key_params:
            if not hasattr(self, "kp_ax"):
                logger.warning("Key parameter axes (self.kp_ax) were never created, "
                               "skipping the key parameter lines")
            else:
                self.kp_ax[0].plot(self.rhos*m**3, resolutions, **kwargs)
                self.kp_ax[1].plot(self.rhos*m**3, track_durations,
                                   linestyle="--", marker='.', **kwargs)
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
