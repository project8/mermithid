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
import configparser
import argparse
import sys
import os
from numpy import pi

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
from mermithid.misc import Constants
#from mermithid.misc import ConversionFunctions

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
        self.config_file_path = reader.read_param(params, 'config_file_path', "required")
        self.comparison_config_file_path = reader.read_param(params, 'comparison_config_file_path', '')
        self.add_comparison_curve = reader.read_param(params, 'add_comparison_curve', False)
        self.add_track_length_axis = reader.read_param(params, 'add_track_length_axis', False)



        # setup sensitivities
        self.sens_main = Sensitivity(self.config_file_path)

        if self.add_comparison_curve:
            self.sens_ref = Sensitivity(self.comparison_config_file_path)
        density_range = [1e14,1e21]
        self.rhos = np.logspace(np.log10(density_range[0]), np.log10(density_range[1]), 100)/m**3


        return True

    def InternalRun(self):

        self.create_plot()

        if self.add_track_length_axis:
            self.add_track_length_axis()
        #self.add_Phase_IV(color="k")
        self.add_goal(2*eV, "Phase III (%.1f eV)"%(2*eV/eV))
        sens = self.sens_main
        #self.add_arrow(sens)
        for a, color in self.range(1, 8):
            sig = sens.BToKeErr(a*ppm*T, sens.MagneticField.nominal_field)
            sens.MagneticField.usefixedvalue = True
            sens.MagneticField.default_systematic_smearing = sig
            sens.MagneticField.default_systematic_uncertainty = 0.05*sig
            self.add_sens_line(sens, color=color)
        self.add_text(5e19, 5, r"molecular"+"\n"+r"$V_\mathrm{eff} = 2\, \mathrm{cm}^3$"+"\n"+r"$\sigma_B = 7\,\mathrm{ppm}$")
        self.add_text(5e19, 2.3, r"$\sigma_B = 1\,\mathrm{ppm}$")
        self.save("./sensitivity_phase_III.png", Veff="2cm^3")

        return True


    def create_plot(self):
        # setup axis
        self.fig, self.ax = plt.subplots(figsize=(6,6))
        ax = self.ax
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(self.rhos[0]*m**3, self.rhos[-1]*m**3)
        ax.set_ylim(1e-2, 1e2)
        ax.set_xlabel(r"(atomic / molecular) number density $\rho\, /\, \mathrm{m}^{-3}$")
        ax.set_ylabel(r"90% CL $m_\beta$ / eV")

    def add_track_length_axis(self):
        ax2 = self.ax.twiny()
        ax2.set_xscale("log")
        ax2.set_xlabel("(atomic) track length / s")
        ax2.set_xlim(self.sens_IV.track_length(self.rhos[0])/s,
                     self.sens_IV.track_length(self.rhos[-1])/s)

        ax3 = self.ax.twiny()
        ax3.spines["top"].set_position(("axes", 1.2))
        ax3.set_frame_on(True)
        ax3.patch.set_visible(False)
        for sp in ax3.spines.values():
            sp.set_visible(False)
        ax3.spines["top"].set_visible(True)
        ax3.set_xscale("log")
        ax3.set_xlabel("(molecular) track length / s")
        ax3.set_xlim(self.sens_III.track_length(self.rhos[0])/s,
                     self.sens_III.track_length(self.rhos[-1])/s)
        self.fig.tight_layout()

    def add_Phase_IV(self, **kwargs):
        limit_IV = [self.sens_IV.CL90(Experiment={"number_density": rho})/eV for rho in self.rhos]
        self.opt_IV = np.argmin(limit_IV)
        self.ax.plot(self.rhos*m**3, limit_IV, **kwargs)
        self.ax.axvline(self.rhos[self.opt_IV]*m**3, ls=":", color="gray", alpha=0.4)

        self.ax.axhline(0.04, color="gray", ls="--")
        self.ax.text(3e14, 0.044, "Phase IV (40 meV)")
        self.ax.text(1.5e14, 0.11, r"atomic"+"\n"+r"$V_\mathrm{eff} = 5\, \mathrm{m}^3$"+"\n"+r"$\sigma_B = 0.13\,\mathrm{ppm}$")

    def add_arrow(self, sens):
        if not hasattr(self, "opt_IV"):
            self.add_Phase_IV()

        def get_relative(val, axis):
            xmin, xmax = self.ax.get_xlim() if axis == "x" else self.ax.get_ylim()
            return (np.log10(val)-np.log10(xmin))/(np.log10(xmax)-np.log10(xmin))

        rho_IV = self.rhos[self.opt_IV]
        track_length_IV = self.sens_IV.track_length(rho_IV)
        track_length_III = self.sens_III.track_length(rho_IV)
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
        self.ax.text(3e14, 1.1*value/eV, label)

    def add_sens_line(self, sens, **kwargs):
        self.ax.plot(self.rhos*m**3, [sens.CL90(Experiment={"number_density": rho})/eV for rho in self.rhos], **kwargs)

    def add_text(self, x, y, text):
        self.ax.text(x, y, text)

    def range(self, start, stop):
        cmap = matplotlib.cm.get_cmap('Spectral')
        norm = matplotlib.colors.Normalize(vmin=start, vmax=stop-1)
        return [(idx, cmap(norm(idx))) for idx in range(start, stop)]

    def save(self, savepath, **kwargs):
        self.fig.tight_layout()
        keywords = ", ".join(["%s=%s"%(key, value) for key, value in kwargs.items()])
        metadata = {"Author": "René Reimann",
                    "Title": "Phase III neutrino mass sensitivity",
                    "Subject":"90% CL upper limit on neutrino mass assuming true mass is zero. We use different Phase III design parameters and plot as a function of number density. As a comparison the Phase IV sensitivity is shown as well. The upper axis gives the corresponding track length for the assumed density. The arrow indicatess the Phase III point with the same track length as in the optimal point of Phase IV.",
                    "Keywords": keywords}
        if savepath is not None:
            self.fig.savefig(savepath.replace(".pdf", ".png"), dpi=200, metadata=metadata)
            self.fig.savefig(savepath.replace(".png", ".pdf"), bbox_inches="tight", metadata=metadata)


###############################################################################

class NameSpace(object):
    def __init__(self, iteritems):
        if type(iteritems) == dict:
            iteritems = iteritems.items()
        for k, v in iteritems:
            setattr(self, k.lower(), v)
    def __getattribute__(self, item):
        return object.__getattribute__(self, item.lower())



###############################################################################
class Sensitivity(object):
    """
    Documentation:
        * Phase IV sensitivity document: https://www.overleaf.com/project/5de3e02edd267500011b8cc4
        * Talias sensitivity script: https://3.basecamp.com/3700981/buckets/3107037/documents/2388170839
        * Nicks CRLB for frequency resolution: https://3.basecamp.com/3700981/buckets/3107037/uploads/2009854398
        * Molecular contamination in atomic tritium: https://3.basecamp.com/3700981/buckets/3107037/documents/3151077016
    """
    def __init__(self, config_path):
        self.cfg = configparser.ConfigParser()
        with open(config_path, 'r') as configfile:
            self.cfg.read_file(configfile)

        # display configuration
        logger.info("Config file content:")
        for sect in self.cfg.sections():
           logger.info('    Section: {}'.format(sect))
           for k,v in self.cfg.items(sect):
              logger.info('        {} = {}'.format(k,v))


        self.Experiment = NameSpace({opt: eval(self.cfg.get('Experiment', opt)) for opt in self.cfg.options('Experiment')})

        self.Tritium_atomic = NameSpace({opt: eval(self.cfg.get('Tritium_atomic', opt)) for opt in self.cfg.options('Tritium_atomic')})
        self.Tritium_molecular = NameSpace({opt: eval(self.cfg.get('Tritium_molecular', opt)) for opt in self.cfg.options('Tritium_molecular')})
        if self.Experiment.atomic:
            self.Tritium = self.Tritium_atomic
        else:
            self.Tritium = self.Tritium_molecular
        self.DopplerBroadening = NameSpace({opt: eval(self.cfg.get('DopplerBroadening', opt)) for opt in self.cfg.options('DopplerBroadening')})
        self.FrequencyExtraction = NameSpace({opt: eval(self.cfg.get('FrequencyExtraction', opt)) for opt in self.cfg.options('FrequencyExtraction')})
        self.MagneticField = NameSpace({opt: eval(self.cfg.get('MagneticField', opt)) for opt in self.cfg.options('MagneticField')})
        self.MissingTracks = NameSpace({opt: eval(self.cfg.get('MissingTracks', opt)) for opt in self.cfg.options('MissingTracks')})
        self.PlasmaEffects = NameSpace({opt: eval(self.cfg.get('PlasmaEffects', opt)) for opt in self.cfg.options('PlasmaEffects')})

    # SENSITIVITY
    def SignalRate(self):
        """signal events in the energy interval before the endpoint, scale with DeltaE**3"""
        signal_rate = self.Experiment.number_density*self.Experiment.v_eff*self.Tritium.last_1ev_fraction/self.Tritium.Livetime
        if not self.Experiment.atomic:
            signal_rate *= 2
        return signal_rate

    def DeltaEWidth(self):
        """optimal energy bin width"""
        labels, sigmas, deltas = self.get_systematics()
        return np.sqrt(self.Experiment.background_rate_per_eV/self.SignalRate()
                              + 8*np.log(2)*(np.sum(sigmas**2)))

    def StatSens(self):
        """Pure statistic sensitivity assuming Poisson count experiment in a single bin"""
        sig_rate = self.SignalRate()
        DeltaE = self.DeltaEWidth()
        sens = 4/(6*sig_rate*self.Experiment.LiveTime)*np.sqrt(sig_rate*self.Experiment.LiveTime*DeltaE
                                                                  +self.Experiment.background_rate_per_eV*self.Experiment.LiveTime/DeltaE)
        return sens

    def SystSens(self):
        """Pure systematic componenet to sensitivity"""
        labels, sigmas, deltas = self.get_systematics()
        sens = 4*np.sqrt(np.sum((sigmas*deltas)**2))
        return sens

    def sensitivity(self, **kwargs):
        for sect, options in kwargs.items():
            for opt, val in options.items():
                self.__dict__[sect].__dict__[opt] = val

        StatSens = self.StatSens()
        SystSens = self.SystSens()

        # Standard deviation on a measurement of m_beta**2 assuming a mass of zero
        sigma_m_beta_2 =  np.sqrt(StatSens**2 + SystSens**2)
        return sigma_m_beta_2

    def CL90(self, **kwargs):
        return np.sqrt(np.sqrt(1.64)*self.sensitivity(**kwargs))

    # PHYSICS Functions
    def frequency(self, energy, magnetic_field):
        # cyclotron frequency
        gamma = lambda energy: energy/(me*c0**2) + 1  # E_kin / E_0 + 1
        frequency = e*magnetic_field/(2*np.pi*me)/gamma(energy)
        return frequency

    def BToKeErr(self, BErr, B):
         return e*BErr/(2*np.pi*self.frequency(self.Tritium.endpoint, B)/c0**2)

    def track_length(self, rho):
        Ke = self.Tritium.endpoint
        betae = np.sqrt(Ke**2+2*Ke*me*c0**2)/(Ke+me*c0**2) # electron speed at energy Ke
        return 1 / (rho * self.Tritium.crosssection_te*betae*c0)

    # SYSTEMATICS

    def get_systematics(self):
        # Different types of uncertainty contributions
        sigma_trans, delta_sigma_trans = self.syst_doppler_broadening()
        sigma_f, delta_sigma_f = self.syst_frequency_extraction()
        sigma_B, delta_sigma_B = self.syst_magnetic_field()
        sigma_Miss, delta_sigma_Miss = self.syst_missing_tracks()
        sigma_Plasma, delta_sigma_Plasma = self.syst_plasma_effects()

        labels = ["Termal Doppler Broadening", "Start Frequency Resolution", "Magnetic Field", "Missing Tracks", "Plasma Effects"]
        sigmas = [sigma_trans, sigma_f, sigma_B, sigma_Miss, sigma_Plasma]
        deltas = [delta_sigma_trans, delta_sigma_f, delta_sigma_B, delta_sigma_Miss, delta_sigma_Plasma]

        if not self.Experiment.atomic:
            labels.append("Molecular final state")
            sigmas.append(self.Tritium.ground_state_width)
            deltas.append(self.Tritium.ground_state_width_uncertainty)

        return np.array(labels), np.array(sigmas), np.array(deltas)

    def print_statistics(self):
        print("Statistic", " "*18, "%.2f"%(np.sqrt(self.StatSens())/meV), "meV")

    def print_systematics(self):
        labels, sigmas, deltas = self.get_systematics()

        print()
        for label, sigma, delta in zip(labels, sigmas, deltas):
            print(label, " "*(np.max([len(l) for l in labels])-len(label)),  "%8.2f"%(sigma/meV), "+/-", "%8.2f"%(delta/meV), "meV")

    def syst_doppler_broadening(self):
        # estimated standard deviation of Doppler broadening distribution from
        # translational motion of tritium atoms / molecules
        # Predicted uncertainty on standard deviation, based on expected precision
        # of temperature knowledge
        if self.DopplerBroadening.UseFixedValue:
            sigma = self.DopplerBroadening.Default_Systematic_Smearing
            delta = self.DopplerBroadening.Default_Systematic_Uncertainty
            return sigma, delta

        # termal doppler broardening
        gasTemp = self.DopplerBroadening.gas_temperature
        mass_T = self.Tritium.mass
        endpoint = self.Tritium.endpoint

        # these factors are mainly neglidible in the recoil equation below
        E_rec = 3.409 * eV # maximal value # same for molecular tritium?
        mbeta = 0*eV # term neglidible
        betanu = 1 # neutrinos are fast
        # electron-neutrino correlation term: 1 + 0.105(6)*betae*cosThetanu
        # => expectation value of cosThetanu = 0.014
        cosThetaenu = 0.014

        Ke = endpoint
        betae = np.sqrt(Ke**2+2*Ke*me*c0**2)/(Ke+me*c0**2) ## electron speed at energy Ke
        Emax = endpoint + me*c0**2
        Ee = endpoint + me*c0**2
        p_rec = np.sqrt( Emax**2-me**2*c0**4 + (Emax - Ee - E_rec)**2 - mbeta**2 + 2*Ee*(Emax - Ee - E_rec)*betae*betanu*cosThetaenu )
        sigma_trans = np.sqrt(p_rec**2/(2*mass_T)*2*kB*gasTemp)

        delta_trans = np.sqrt(p_rec**2/(2*mass_T)*kB/gasTemp*self.DopplerBroadening.gas_temperature_uncertainty**2)
        return sigma_trans, delta_trans

    def syst_frequency_extraction(self):
        # cite{https://3.basecamp.com/3700981/buckets/3107037/uploads/2009854398} (Section 1.2, p 7-9)
        # Are we double counting the antenna collection efficiency? We use it here. Does it also impact the effective volume, v_eff ?
        # Are we double counting the effect of magnetic field uncertainty here? Is 'sigma_f_Bfield' the same as 'rRecoErr', 'delta_rRecoErr', 'rRecoPhiErr', 'delta_rRecoPhiErr'?

        if self.FrequencyExtraction.UseFixedValue:
            sigma = self.FrequencyExtraction.Default_Systematic_Smearing
            delta = self.FrequencyExtraction.Default_Systematic_Uncertainty
            return sigma, delta

        ScalingFactorCRLB = self.FrequencyExtraction.CRLB_scaling_factor # Cramer-Rao lower bound / how much worse are we than the lower bound
        ts = self.FrequencyExtraction.track_timestep
        Gdot = self.FrequencyExtraction.track_onset_rate
        Ke = self.Tritium.endpoint

        fEndpoint = self.frequency(self.Tritium.endpoint, self.MagneticField.nominal_field) # cyclotron frequency at the endpoint
        betae = np.sqrt(Ke**2+2*Ke*me*c0**2)/(Ke+me*c0**2) # electron speed at energy Ke
        Pe = 2*np.pi*(e*fEndpoint*betae*np.sin(self.FrequencyExtraction.pitch_angle))**2/(3*eps0*c0*(1-(betae)**2)) # electron radiation power
        alpha_approx = fEndpoint * 2 * np.pi * Pe/me/c0**2 # track slope
        sigNoise = np.sqrt(kB*self.FrequencyExtraction.noise_temperature/ts) # noise level
        Amplitude = np.sqrt(self.FrequencyExtraction.epsilon_collection*Pe)
        Nsteps = 1 / (self.Experiment.number_density * self.Tritium.crosssection_te*betae*c0*ts) # Number of timesteps of length ts

        # sigma_f from Cramer-Rao lower bound in Hz
        sigma_f_CRLB = (ScalingFactorCRLB /(2*np.pi) * sigNoise/Amplitude * np.sqrt(alpha_approx**2/(2*Gdot)
                    + 96.*Nsteps/(ts**2*(Nsteps**4-5*Nsteps**2+4))))
        # uncertainty in alpha
        delta_alpha = 6*sigNoise/(Amplitude*ts**2) * np.sqrt(10/(Nsteps*(Nsteps**4-5*Nsteps**2+4)))
        # uncetainty in sigma_f in Hz due to uncertainty in alpha
        delta_sigma_f_CRLB = delta_alpha * alpha_approx *sigNoise**2/(8*np.pi**2*Amplitude**2*Gdot*sigma_f_CRLB*ScalingFactorCRLB**2)

        # sigma_f from Cramer-Rao lower bound in eV
        sigma_K_f_CRLB =  e*self.MagneticField.nominal_field/(2*np.pi*fEndpoint**2)*sigma_f_CRLB*c0**2
        delta_sigma_K_f_CRLB = e*self.MagneticField.nominal_field/(2*np.pi*fEndpoint**2)*delta_sigma_f_CRLB*c0**2

        # combined sigma_f in eV
        sigma_f = np.sqrt(sigma_K_f_CRLB**2 + self.FrequencyExtraction.magnetic_field_smearing**2)
        delta_sigma_f = np.sqrt((delta_sigma_K_f_CRLB**2 + self.FrequencyExtraction.magnetic_field_uncertainty**2)/2)

        return sigma_f, delta_sigma_f

    def syst_magnetic_field(self):
        if self.MagneticField.UseFixedValue:
            sigma = self.MagneticField.Default_Systematic_Smearing
            delta = self.MagneticField.Default_Systematic_Uncertainty
            return sigma, delta

        B = self.MagneticField.nominal_field
        if self.MagneticField.useinhomogenarity:
            inhomogenarity = self.MagneticField.inhomogenarity
            sigma = self.BToKeErr(inhomogenarity*B, B)
            return sigma, 0.05*sigma

        BMapErr = self.MagneticField.probe_repeatability  # Probe Repeatability
        delta_BMapErr = self.MagneticField.probe_resolution # Probe resolution

        BFlatErr = self.MagneticField.BFlatErr # averaging over the flat part of the field
        delta_BFlatErr = self.MagneticField.relative_uncertainty_BFlatErr*BFlatErr # UPDATE ?

        Delta_t_since_calib = self.MagneticField.time_since_calibration
        shiftBdot = self.MagneticField.shift_Bdot
        smearBdot = self.MagneticField.smear_Bdot
        delta_shiftBdot = self.MagneticField.uncertainty_shift_Bdot
        delta_smearBdot = self.MagneticField.uncertainty_smearBdot
        BdotErr = Delta_t_since_calib * np.sqrt(shiftBdot**2 + smearBdot**2)
        delta_BdotErr = Delta_t_since_calib**2/BdotErr * np.sqrt(shiftBdot**2 * delta_shiftBdot**2 + smearBdot**2 * delta_smearBdot**2)

        rRecoErr = self.MagneticField.rRecoErr
        delta_rRecoErr = self.MagneticField.relative_Uncertainty_rRecoErr * rRecoErr

        rRecoPhiErr = self.MagneticField.rRecoPhiErr
        delta_rRecoPhiErr = self.MagneticField.relative_uncertainty_rRecoPhiErr * rRecoPhiErr

        rProbeErr = self.MagneticField.rProbeErr
        delta_rProbeErr = self.MagneticField.relative_uncertainty_rProbeErr * rProbeErr

        rProbePhiErr = self.MagneticField.rProbePhiErr
        delta_rProbePhiErr = self.MagneticField.relative_uncertainty_rProbePhiErr * rProbePhiErr

        Berr = np.sqrt(BMapErr**2 +
                       BFlatErr**2 +
                       BdotErr**2 +
                       rRecoErr**2 +
                       rRecoPhiErr**2 +
                       rProbeErr**2 +
                       rProbePhiErr**2)

        delta_Berr = 1/Berr * np.sqrt(BMapErr**2 * delta_BMapErr**2 +
                                      BFlatErr**2 * delta_BFlatErr**2 +
                                      BdotErr**2 * delta_BdotErr**2 +
                                      rRecoErr**2 * delta_rRecoErr**2 +
                                      rRecoPhiErr**2 * delta_rRecoPhiErr**2 +
                                      rProbeErr**2 * delta_rProbeErr**2 +
                                      rProbePhiErr**2 * delta_rProbePhiErr**2)

        return self.BToKeErr(Berr, B), self.BToKeErr(delta_Berr, B)

    def syst_missing_tracks(self):
        if self.MissingTracks.UseFixedValue:
            sigma = self.MissingTracks.Default_Systematic_Smearing
            delta = self.MissingTracks.Default_Systematic_Uncertainty
            return sigma, delta

    def syst_plasma_effects(self):
        if self.PlasmaEffects.UseFixedValue:
            sigma = self.PlasmaEffects.Default_Systematic_Smearing
            delta = self.PlasmaEffects.Default_Systematic_Uncertainty
            return sigma, delta
        else:
            raise NotImplementedError()

###############################################################################