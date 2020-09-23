#
# fake_data_stan_analysis.py
# Author: T. E. Weiss
# Date modified: June 2, 2020
#
# This script generates fake data, then analyzes the data in Stan to infer posteriors.
# Pathnames configured for running from: scripts/Phase-II-official-analysis/fake-data-study/
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
import numpy as np

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
        'Nscatters': 20,
        'minf': 1353.125e+06 - 40e+06 + 24.5e+09, #In Hz
        'err_from_B': 0.
        },
    "priors": [
        {'name': 'Q', 'prior_dist': 'normal', 'prior_params': [QT2(), 0.07]},
        {'name': 'mass', 'prior_dist': 'gamma', 'prior_params': [1.1532, 0.4291]},
        {'name': 'sigma', 'prior_dist': 'normal', 'prior_params': [17.7084, 1.14658]}, #From Ali's complex lineshape fits. Final states not yet included.
        {'name': 'S', 'prior_dist': 'poisson', 'prior_params': [3300.]},
        {'name': 'B_1kev', 'prior_dist': 'lognormal', 'prior_params': [-3.826151398234498, 2.1673316326073935]},
        {'name': 'survival_prob', 'prior_dist': 'beta', 'prior_params': [55, 17]}, #Centered approximately around 0.77. To be replaced given complex lineshape result/systematic assessment
        {'name': 'Bfield', 'prior_dist': 'normal', 'prior_params': [0.9574762, 1.51e-6]}, #From complex lineshape fit to calibration data. More sig figs on mean needed?
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
        "efficiency_path": "../tests/combined_energy_corrected_eff_at_quad_trap_frequencies.json",
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
        "n_steps": 100000,
    }
    
    histo_config = {
        "variables": "F",
        "n_bins_x": 65,
        "output_path": "./results/",
        "title": "0Psuedo-data",
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
    binner_config = {
        "energy_or_frequency": 'frequency',
        "variables": "F",
        "title": "corrected_spectrum",
        "efficiency_filepath": "../tests/combined_energy_corrected_eff_at_quad_trap_frequencies.json",
        'bins': np.linspace(tritium_data['minf'], tritium_data['maxf'], nbins),
        'fss_bins': False # If fss_bins is True, bins is ignored and overridden
        }
        
    binner = TritiumAndEfficiencyBinner("binner")
    binner.Configure(binner_config)
    binner.data = tritium_data
    binner.Run()
    results = binner.results
    stan_inputs = {'freq':results['F'], 'N':results['N'], 'eff_means':results['bin_efficiencies'], 'minf':tritium_data['minf'], 'maxf':tritium_data['maxf']}
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

    return results



def StanTritiumAnalysis(tritium_data, fit_parameters=None, root_file='./results/tritium_analysis.root', stan_files_location='../../morpho_models/', model_code='tritium_model/models/tritium_phase_II_analyzer_binned.stan', scattering_params_R='simplified_scattering_params.R'):
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
    
    #Stan analysis parameters
    analyzer_config = {
        "model_code": stan_files_location+model_code,
        "function_files_location": stan_files_location+"functions",
        #***Add "temp_" when switching to cmdstan.***
        "model_name": stan_files_location+"tritium_model/models/tritium_phase_II_analyzer_binned", #Binned version of Stan model
        "cache_dir": stan_files_location+"tritium_model/cache",
        "warmup": 4000, #Increase for real run (to 3000-5000)
        "iter": 8000, #Increase for real run (to 6000-9000)
        "chain": 3, #Increase for real run (to 3-4)
        "control": {'adapt_delta':0.97},
        "init": {
            "sigma": 17.7084,
            "survival_prob": 0.77,
            "Q": 18573.24,
            "mass": 0.2,
            "Bfield": 0.9574762,
            "S": 3300.,
            "B_1kev": 0.0217933282798889,
            "B": 0.3,
        },
        "input_data": {
            "sigma_ctr": 17.7084, #sigma params from Ali's complex lineshape fits.
            "sigma_std": 1.14658, #Final states not yet included.
            "err_from_B": 0.001, #Tiny smearing from f_c->K conversion
            "Bfield_ctr": 0.9574762, #From complex lineshape fit to calibration
            "Bfield_std": 1.51e-06, #data. More sig figs on mean needed?
            "survival_prob_alpha": 55, #Centered around ~0.77. To be replaced
            "survival_prob_beta": 17, #given complex lineshape result+systematics
            "Q_ctr": QT2(),
            "Q_std": 75.,
            "m_alpha": 1.1532,
            "m_beta": 2.33046,
            "B_1kev_logctr": -3.826151398234498,
            "B_1kev_logstd": 2.1673316326073935,
            "KEscale": 16323, #This enables the option of cmdstan running
#            "slope": 0.000390369173, #For efficiency modeling with unbinned data
#            "intercept": -6.00337656,
            "Nscatters": 16 #Because peaks>16 in simplified linesahpe have means->inf as FWHM->0
        },
        "interestParams": ['Q', 'mass', 'survival_prob', 'Bfield', 'sigma', 'S', 'B_1kev', 'KEmin', 'KE_sample', 'Nfit_signal', 'Nfit_bkgd'],
    }
    
    posteriors_writer_config = {
        "action": "write",
        "tree_name": "analysis",
        "file_option": "update",
        "filename": root_file,
        "variables": [
            {"variable": "Q", "type": "float"},
            {"variable": "mass", "type": "float"},
            {"variable": "survival_prob", "type": "float"},
            {"variable": "Bfield", "type": "float"},
            {"variable": "sigma", "type": "float"},
            {"variable": "S", "type": "float"},
            {"variable": "B_1kev", "type": "float"},
            {"variable": "KEmin", "type": "float"},
            {"variable": "KE_sample", "type": "float"},
            {"variable": "Nfit_signal", "type": "float"},
            {"variable": "Nfit_bkgd", "type": "float"},
            {"variable": "divergent__", "root_alias": "divergence", "type": "float"},
            {"variable": "energy__", "root_alias": "energy", "type": "float"},
            {"variable": "lp_prob", "root_alias": "lp_prob", "type": "float"}
        ]}
    
    #Configuration step
    rReaderProcessor.Configure(scattering_reader_config)
    analysisProcessor.Configure(analyzer_config)
    writerProcessor.Configure(posteriors_writer_config)

    #Make data accessible to analyzer
    analysisProcessor.data = tritium_data
    analysisProcessor.data = {'Nbins': len(tritium_data['N'])}
    
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


def PerformFakeExperiment(root_filename, plot_results=True, parallelized=True, bin_data=True):
    """
    Generate fake tritium data and analyze it in Stan. If plot_results==True, create correlation and divergence plots.
    
    Saves generation inputs, generation results, and analysis results to different trees of one ROOT file.
    """
    if parallelized==True:
        flist = root_filename.split('/')
        logger.info("----------MORPHO RUN #{}----------".format(flist[len(flist)-1][0]))
    
    #Sample inputs to the data generator and save to one branch of a root file:
    inputs_dict = DefineGeneratorInputs(root_filename)
    
    #Generate data using the inputs
    tritium_data_unbinned = GenerateFakeData(inputs_dict)
    
    #Optionally bin data, then save it
    if bin_data:
        nbins = 65
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
    "verbose": False
    }
    #Configuration step
    calibrator.Configure(calib_config)
    
    calibrator.Run()


def FakeExperimentEnsemble(n_runs, root_basename, wait_before_runs=0, parallelize=True, n_processes=4, vars_to_calibrate=['Q']):
    """
    To-do: add parallelization option for a Slurm environment.
    
    n_runs: int; number of pseudo-experiments to be performed
    root_basename: str; results are saved in rootfiles labeled by root_basename and the pseudo-experiment number
    wait_before_runs: int or float; number of seconds to pause between the start of one pseudo-experiment and the start of the next
    """
    if n_runs==1:
        logger.info("PERFORMING 1 MORPHO PSEUDO-EXPERIMENT")
    else:
        logger.info("PERFORMING {} MORPHO PSEUDO-EXPERIMENTS".format(n_runs))

    
    if parallelize == False:
        root_filenames = []
        for i in range(n_runs):
            time.sleep(i*wait_before_runs) #Optionally stagger the runs slightly, either for debugging/clarity of output, or to avoid any possible memory overflows
            logger.info("----------MORPHO RUN #{}----------".format(i))
            temp_root_name = "./results/"+str(i)+root_basename
            root_filenames.append(temp_root_name)
            PerformFakeExperiment(temp_root_name)
    else:
        from multiprocessing import Pool
        root_filenames = ["./results/"+str(i)+root_basename for i in range(n_runs)]
        with Pool(n_processes) as p:
            p.map(PerformFakeExperiment, root_filenames)
            
    coverages = CalibrateResults(root_filenames, vars_to_calibrate)
    
    return coverages


if __name__ == '__main__':
    interest_vars = ['Q', 'mass','survival_prob', 'Bfield', 'sigma', 'S', 'B_1kev']
    FakeExperimentEnsemble(1, "fake_data_and_results.root", wait_before_runs=30, vars_to_calibrate=interest_vars)
