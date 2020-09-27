#
# fake_data_stan_analysis.py
# Author: T. E. Weiss
# Date modified: June 2, 2020
#
# This script generates fake data, then analyzes the data in Stan to infer posteriors.
# Pathnames configured for running from: termite/phase2_main_scripts/
#

"""
To-do:
    - In FakeExperimentEnsemble, add:
        1. Tracking of convergence issues, so that a summary of problems can be saved/printed
        2. An option to parallelize with slurm instead of multiprocessing
    - Run morpho processor for ensemble-analysis plotting, once it has been tested sufficiently
"""


#import unittest
from morpho.utilities import morphologging
logger = morphologging.getLogger(__name__)
import logging
import numpy as np
import time
import argparse
parser = argparse.ArgumentParser()

from scipy.interpolate import interp1d
import json


#Importing constants from mermithid
from mermithid.misc.Constants import *
from mermithid.processors.TritiumSpectrum.FakeDataGenerator import FakeDataGenerator
from mermithid.processors.misc.TritiumAndEfficiencyBinner import TritiumAndEfficiencyBinner
#Importing processors from morpho
from morpho.processors.sampling import PyStanSamplingProcessor, PriorSamplingProcessor
from morpho.processors.plots import Histogram, APosterioriDistribution, Histo2dDivergence
from morpho.processors.IO import IOROOTProcessor, IORProcessor

#Defining processors
priorSampler = PriorSamplingProcessor("sample")
specGen = FakeDataGenerator("specGen")
writerProcessor = IOROOTProcessor("writer")
rReaderProcessor = IORProcessor("reader")
analysisProcessor = PyStanSamplingProcessor("analyzer")
histPlotter = Histogram("histo")
aposterioriPlotter = APosterioriDistribution("posterioriDistrib")
divPlotter = Histo2dDivergence("2dDivergence")



def DefineGeneratorInputs(root_file='./results/tritium_analysis.root'):
    """
    Samples inputs to a fake data generator from priors, then combines them in a dictionary with fixed inputs to the generator. Saves all the inputs to a root file.
    
    Returns:
        generator_inputs: dictionary of all inputs to a fake data generator
    """
    prior_sampler_config = {
    "fixed_inputs": {
        'Nscatters': const_dict['Nscatters_generation'],
        'minf': const_dict['minf'], #In Hz
        'err_from_B': const_dict['err_from_B_generation'],
        'H2_scatter_prop': const_dict['H2_scatter_prop_tritium'],
    },
    "priors": [
        {'name': 'Q', 'prior_dist': 'normal', 'prior_params': const_dict['Q']},
        {'name': 'mass', 'prior_dist': 'gamma', 'prior_params': const_dict['mass']},
        {'name': 'sigma', 'prior_dist': 'normal', 'prior_params': const_dict['sigma']}, #From Ali's complex lineshape fits. Final states also included
        {'name': 'S', 'prior_dist': 'poisson', 'prior_params': const_dict['S']},
        {'name': 'B_1kev', 'prior_dist': 'lognormal', 'prior_params': const_dict['B_1kev']},
        {'name': 'survival_prob', 'prior_dist': 'beta', 'prior_params': const_dict['survival_prob']}, #Centered around 0.736. To be replaced given complex lineshape result/systematic assessment
        {'name': 'Bfield', 'prior_dist': 'normal', 'prior_params': const_dict['Bfield']}, #From complex lineshape fit to calibration data. More sig figs on mean needed?
       ]
    }
    
    inputs_writer_config = {
    "action": "write",
    "tree_name": "input",
    "filename": root_file,
    "variables": [
        {"variable": "Nscatters", "type":"int"},
        {"variable": "minf", "type": "float"},
        {"variable": "err_from_B", "type": "float"},
        {"variable": "Q", "type": "float"},
        {"variable": "mass", "type": "float"},
        {"variable": "sigma", "type": "float"},
        {"variable": "S", "type": "float"},
        {"variable": "B_1kev", "type": "float"},
        {"variable": "survival_prob", "type": "float"},
        {"variable": "Bfield", "type": "float"}
    ]}
    
    #Configuration step
    priorSampler.Configure(prior_sampler_config)
    writerProcessor.Configure(inputs_writer_config)
    
    #Sampling inputs
    priorSampler.Run()
    generator_inputs = priorSampler.results
    
    #Saving results
    gen_inputs_root = {key:[value] for key, value in generator_inputs.items()}
    writerProcessor.data = gen_inputs_root
    writerProcessor.Run()
    
    return generator_inputs


def GenerateFakeData(inputs_dict):
    """
    Generates fake Phase II tritium beta spectrum data and plots it.
    
    Arguments:
    - inputs_dict: dictionary of parameter values inputted to the fake data generator.
    
   Returns:
   - results: dict with
        1) keys: K (energy), F (frequency)
        2) mapped values: energy, frequency
    """
    specGen_config = {
        "apply_efficiency": True,
        "efficiency_path": "/host-termite/analysis_input/combined_energy_corrected_eff_at_quad_trap_frequencies.json",
        "detailed_or_simplified_lineshape": "detailed",
        "return_frequency": True,
        "Q": inputs_dict["Q"],
        "mass": inputs_dict["mass"],
        "minf": inputs_dict["minf"],
        "scattering_sigma": inputs_dict["sigma"],
        "S": inputs_dict["S"],
        "B_1kev": inputs_dict["B_1kev"],
        "survival_prob": inputs_dict["survival_prob"],
        "err_from_B": inputs_dict["err_from_B"],
        "Nscatters": inputs_dict["Nscatters"],
        "B_field": inputs_dict["Bfield"],
        "scatter_proportion": inputs_dict["H2_scatter_prop"],
        "n_steps": 100000,
    }
    
    histo_config = {
        "variables": "F",
        "n_bins_x": 65,
        "output_path": "./results/",
        "title": "psuedo-data"+str(inputs_dict['Q']),
        "format": "pdf"
    }
    
    #Configuration step
    specGen.Configure(specGen_config)
    histPlotter.Configure(histo_config)
    #Generate data
    specGen.Run()
    results = specGen.results
    #Plot histograms of generated data
    histPlotter.data = {'F':results['F'].tolist()}
    histPlotter.Run()
    
    return results


def BinAndSaveData(tritium_data, nbins, root_file="./results/tritium_analysis.root"):
    eff_path = "/host-termite/analysis_input/combined_energy_corrected_eff_at_quad_trap_frequencies.json"
    
    binner_config = {
        "energy_or_frequency": 'frequency',
        "variables": "F",
        "title": "corrected_spectrum",
        "efficiency_filepath": eff_path,
        'bins': np.linspace(tritium_data['minf'], tritium_data['maxf'], nbins), #(tritium_data['maxf']-tritium_data['minf'])/float(nbins)
        'fss_bins': False # If fss_bins is True, bins is ignored and overridden
        }
        
    binner = TritiumAndEfficiencyBinner("binner")
    binner.Configure(binner_config)
    binner.data = tritium_data
    binner.Run()
    results = binner.results
    
    eff_means = results['bin_efficiencies']
    eff_errs = (results['bin_efficiency_errors'][0]+results['bin_efficiency_errors'][1])/2.
    for i in range(len(eff_means)):
        if (eff_means[i]<eff_errs[i]): #These bins are far past the endpoint and do not affect the fit. Efficiencies are modified to facilitate model convergence.
            eff_means[i] = eff_errs[i]/2.
            eff_errs[i] = eff_errs[i]/40.
    
    stan_inputs = {'freq':results['F'], 'N':results['N'], 'eff_means':eff_means, 'eff_errs':eff_errs, 'minf':tritium_data['minf'], 'maxf':tritium_data['maxf']}
    results['minf'], results['maxf'] = tritium_data['minf'], tritium_data['maxf']

    data_writer_config = {
        "action": "write",
        "tree_name": "data",
        "file_option": "update",
        "filename": root_file,
        "variables": [
        {"variable": "freq", "type": "float"},
        {"variable": "N", "type":"float"},
        {"variable": "eff_means", "type":"float"},
        ]}
    #It would be nice to save energies, too, but okay if not.
    # {"variable": "K", "root_alias":"KE", "type": "float"}

    #Save data points
    writerProcessor.Configure(data_writer_config)
    writerProcessor.data = stan_inputs
    writerProcessor.Run()

    return stan_inputs
    
    
    
def SaveUnbinnedData(tritium_data, root_file="./results/tritium_analysis.root"):

    data_writer_config = {
    "action": "write",
    "tree_name": "data",
    "file_option": "update",
    "filename": root_file,
    "variables": [
    {"variable": "K", "root_alias":"KE", "type": "float"},
    {"variable": "F", "type": "float", "root_alias":"freq"}
    ]}

    #Save data points
    writerProcessor.Configure(data_writer_config)
    writerProcessor.data = tritium_data
    writerProcessor.Run()



def StanTritiumAnalysis(tritium_data, fit_parameters=None, root_file='./results/tritium_analysis.root', stan_files_location='/host-termite/stan_models/', model_code='phase2/models/tritium_phase_II_analyzer_binned.stan', scattering_params_R='/host-termite/analysis_input/simplified_scattering_params.R'):
    """
    Analyzes frequency or kinetic energy data using a Stan model. Saves and plots posteriors.
    
    Required argument:
        1) data: dictionary with key(s) KE (energy) and/or F (frequency)
    Optional arguments:
        2) fit_parameters: dictionary parameters describing any priors, or other inputs to the Stan model, that are produced by other analyses in mermithid
        3) root_file: string; name of root file where posteriors are saved
        4) F_or_K: string; set to "F" or "K"
        5) stan_files_location: string; path to directory that contains folders with Stan function files, models, and model caches
        6) model_code: string; path to Stan model (absolute, or path from within stan_files_location)
        7) scattering_params_R: string; path to R file containing parameters employed for Stan scattering model
    """
    #Read in scattering parameters from file
    scattering_reader_config = {
        "action": "read",
        "filename": scattering_params_R,
        "variables": ['p0', 'p1', 'p2', 'p3'],
        "format": "R"
    }
    
    Nbins = len(tritium_data['N'])
    interest_params = ['Q', 'mass', 'survival_prob', 'Bfield', 'sigma', 'S', 'B_1kev', 'B', 'KEmin', 'rate_param', 'Ndata_gen',  'KE_sample', 'Nfit_signal', 'Nfit_bkgd']
    for i in range(Nbins):
        interest_params.append('Nfit_bins[{}]'.format(str(i)))
    
    #Stan analysis parameters
    analyzer_config = {
        "model_code": stan_files_location+model_code,
        "function_files_location": stan_files_location+"functions",
        #***Add "temp_" when switching to cmdstan.***
        "model_name": stan_files_location+"phase2/models/tritium_phase_II_analyzer_binned", #Binned version of Stan model
        "warmup": const_dict['warmup'], #Increase for real run (to 3000-5000)
        "iter": const_dict['iter'], #Increase for real run (to 6000-9000)
        "chain": const_dict['chain'], #Increase for real run (to 3-4)
        "control": {'adapt_delta':const_dict['adapt_delta']},
        "no_diagnostics": False,
        "init": {
            "sigma": const_dict['sigma'][0],
            "survival_prob": const_dict['survival_prob_mean'],
            "Q": const_dict['Q'][0],
            "mass": const_dict['mass_init'],
            "Bfield": const_dict['Bfield'][0],
            "S": const_dict['S'][0],
            "B_1kev": const_dict['B_1kev_init'],
            "B": const_dict['B_init'],
            "eff": tritium_data['eff_means']+0.5*tritium_data['eff_errs']  #Add a displacement to prevent initialization to parameter boundary at 0
        },
        "input_data": {
            "sigma_ctr": const_dict['sigma'][0], #sigma params from Ali's complex lineshape fits.
            "sigma_std": const_dict['sigma'][1], #Final states now included
            "err_from_B": const_dict['err_from_B_analysis'], #Tiny smearing from f_c->K conversion
            "Bfield_ctr": const_dict['Bfield'][0], #From complex lineshape fit to calibration
            "Bfield_std": const_dict['Bfield'][1], #data. More sig figs on mean needed?
            "survival_prob_alpha": const_dict['survival_prob'][0], #Centered around ~0.736. To be replaced
            "survival_prob_beta": const_dict['survival_prob'][1], #given complex lineshape result+systematics
            "Q_ctr": QT2(),
            "Q_std": const_dict['Q_std_analysis'],
            "m_alpha": const_dict['mass'][0],
            "m_beta": 1./const_dict['mass'][1],
            "B_1kev_logctr": const_dict['B_1kev'][0],
            "B_1kev_logstd": const_dict['B_1kev'][1],
            "KEscale": const_dict['KEscale'], #This enables the option of cmdstan running
#            "slope": 0.000390369173, #For efficiency modeling with unbinned data
#            "intercept": -6.00337656,
            "Nscatters": const_dict['Nscatters_analysis'] #Because peaks>16 in simplified linesahpe have means->inf as FWHM->0
        },
        "interestParams": interest_params,
    }
    
    
    vars_to_save = [
        {"variable": "Q", "type": "float"},
        {"variable": "mass", "type": "float"},
        {"variable": "survival_prob", "type": "float"},
        {"variable": "Bfield", "type": "float"},
        {"variable": "sigma", "type": "float"},
        {"variable": "S", "type": "float"},
        {"variable": "B_1kev", "type": "float"},
        {"variable": "B", "type": "float"},
        {"variable": "KEmin", "type": "float"},
        {"variable": "Ndata_gen", "type": "float"},
        {"variable": "rate_param", "type": "float"},
        {"variable": "KE_sample", "type": "float"},
        {"variable": "Nfit_signal", "type": "float"},
        {"variable": "Nfit_bkgd", "type": "float"},
        {"variable": "divergent__", "root_alias": "divergence", "type": "float"},
        {"variable": "energy__", "root_alias": "energy", "type": "float"},
        {"variable": "lp_prob", "root_alias": "lp_prob", "type": "float"}
    ]
    
    for i in range(Nbins):
        vars_to_save.append({"variable": 'Nfit_bins[{}]'.format(str(i)), "root_alias": 'Nfit_bins{}'.format(str(i)), "type": "float"})
    
    
    posteriors_writer_config = {
        "action": "write",
        "tree_name": "analysis",
        "file_option": "update",
        "filename": root_file,
        "variables": vars_to_save}
    
    #Configuration step
    rReaderProcessor.Configure(scattering_reader_config)
    analysisProcessor.Configure(analyzer_config)
    writerProcessor.Configure(posteriors_writer_config)

    #Make data accessible to analyzer
    analysisProcessor.data = tritium_data
    analysisProcessor.data = {'Nbins': Nbins}
    
    #Make scattering parameters accessible to analyzer
    rReaderProcessor.Run()
    pi = rReaderProcessor.data
    analysisProcessor.data = {key:val[:analysisProcessor.data['Nscatters']] for key, val in pi.items()}

    #Run analysis
    analysisProcessor.Run()
    results = analysisProcessor.results
    
    #Save results
    writerProcessor.data = results
    writerProcessor.Run()
    
    return results


def PlotStanResults(posteriors, correlation_vars=['Q', 'mass'], divergence_vars=['Q', 'Bfield', 'survival_prob']):
    """
    Creates and saves two plots:
        a) posteriors and correlations between them, and
        b) plot showing where in parameter space Hamiltonian Monte Carlo divergences occured.
        
    Required argument:
        1) posteriors: dict; output of the PyStanSamplingProcessor
    Optional arguments:
        2) correlation_vars: list of strings; names of variables to be included in correlation plot
        3) divergence_vars: list of strings; names of variables to be included in divergences plot
    """
    aposteriori_config = {
        "n_bins_x": 50, #Potentially increase
        "n_bins_y": 50,
        "variables": correlation_vars,
        "title": "Q_vs_Kmin",
        "output_path": "./plots"
    }
    
    div_plot_config = {
        "n_bins_x": 50,
        "n_bins_y": 50,
        "variables": divergence_vars,
        "title": "div_plot",
        "output_path": "./plots"
    }
    
    #Configuration step
    aposterioriPlotter.Configure(aposteriori_config)
    divPlotter.Configure(div_plot_config)
    
    #Plots correlations between posteriors
    aposterioriPlotter.data = posteriors
    aposterioriPlotter.Run()
    
    #Plot 2D grid of divergent points
    divPlotter.data = posteriors
    divPlotter.Run()


def PerformFakeExperiment(root_filename, plot_results=False, parallelized=True, bin_data=True, wait=45):
    """
    Generate fake tritium data and analyze it in Stan. If plot_results==True, create correlation and divergence plots.
    
    Saves generation inputs, generation results, and analysis results to different trees of one ROOT file.
    """
    if parallelized==True:
        flist = root_filename.split('/')
        run_num = float(flist[len(flist)-1][0])
        time.sleep(run_num*wait) #Optionally stagger the runs slightly, either for debugging/clarity of output, or to avoid any possible memory overflows
        logger.info("----------------------MORPHO RUN #{}----------------------".format(int(run_num)))
        
    
    #Sample inputs to the data generator and save to one branch of a root file:
    inputs_dict = DefineGeneratorInputs(root_filename)
    
    #Generate data using the inputs
    tritium_data_unbinned = GenerateFakeData(inputs_dict)
    
    #Optionally bin data, then save it
    if bin_data:
        nbins = const_dict['f_nbins']
        tritium_data = BinAndSaveData(tritium_data_unbinned, nbins, root_filename)
    else:
        tritium_data = SaveUnbinnedData(tritium_data, root_filename)
    
    #Analyze data and save posteriors
    posteriors = StanTritiumAnalysis(tritium_data, root_file=root_filename)
    
    #Optionally plot posteriors
    if plot_results == True:
        PlotStanResults(posteriors)
    

def CalibrateResults(root_filenames, vars_to_calibrate, cred_interval=[0.05, 0.95]):
    from morpho.processors.diagnostics.CalibrationProcessor import CalibrationProcessor
    calibrator = CalibrationProcessor("calib")

    calib_config = {
    "files": root_filenames,
    "in_param_names": vars_to_calibrate,
    "cred_interval": cred_interval,
    "quantiles": True
    }
    #Configuration step
    check_success = calibrator.Configure(calib_config)
    if check_success == False:
        return
    
    calibrator.Run()


def FakeExperimentEnsemble(n_runs, root_basename, parallelize=True, n_processes=4, vars_to_calibrate=['Q']):
    """
    To-do: add parallelization option for a Slurm environment.
    
    n_runs: int; number of pseudo-experiments to be performed
    root_basename: str; results are saved in rootfiles labeled by root_basename and the pseudo-experiment number
    """
    if n_runs==1:
        logger.info("PERFORMING 1 MORPHO PSEUDO-EXPERIMENT")
    else:
        logger.info("PERFORMING {} MORPHO PSEUDO-EXPERIMENTS".format(n_runs))

    
    if parallelize == False:
        root_filenames = []
        for i in range(n_runs):
            logger.info("----------MORPHO RUN #{}----------".format(i))
            temp_root_name = "./results/"+str(i)+root_basename
            root_filenames.append(temp_root_name)
            PerformFakeExperiment(temp_root_name)
    else:
        from multiprocessing import Pool
        root_filenames = ["./results/"+str(i)+root_basename for i in range(n_runs)]
        with Pool(n_processes) as p:
            p.map(PerformFakeExperiment, root_filenames)
            
    #coverages = CalibrateResults(root_filenames, vars_to_calibrate)
    coverages = None
    return coverages


if __name__ == '__main__':
    const_dict = {
       # physical
           'Q':[QT2(),0.07], 'Q_std_analysis':75, 'mass':[1.1532, 0.4291], 'mass_init':0.2,
       # instrumental
           'sigma': [15.524869238312053, 2.1583740056146], #[14.542692695653235, 1.3080022178297848], #1.301788807656317
           'S':[3594],
           'err_from_B_generation':0, 'err_from_B_analysis':0.001,
           'B_1kev':[-3.057997933394048, 1.9966097834821164], 'B_1kev_init':0.0469817, 'B_init':0.06,
           'minf':1353.125e+06 - 40e+06 + 24.5e+09, 'f_nbins':65,
       # complex lineshape
           'survival_prob_mean': 0.672621806411212, #0.688, #0.736,
           'survival_prob_err': 0.10760576977058844, #0.0368782, #0.00850261,
           'survival_prob': [12.118849968055722, 5.898481394892654], #[107.90268235294104, 48.93261176470582], #[2042.1165325779257, 926.0761019830128]
           'Bfield': [0.957809551203932, 8.498072412705302e-6], #[0.957805552, 1.3926736876423273e-6], #[0.957805552, 7.620531477410062e-7]
           'Nscatters_generation':20, 'Nscatters_analysis':16,
           'H2_scatter_prop_tritium':1.,
        # Stan fit
           'chain':1, 'warmup':2000, 'iter':3000,
           'KEscale':16323, 'adapt_delta':0.8
    }

    interest_vars = ['Q', 'mass','survival_prob', 'Bfield', 'sigma', 'S', 'B_1kev']

    parser.add_argument("root_filename", type=str)
    args = parser.parse_args()
    
    #CalibrateResults([args.root_filename], interest_vars, [0.16, 0.84])
    FakeExperimentEnsemble(3, args.root_filename, vars_to_calibrate=interest_vars)

