#
# fake_data_stan_analysis.py
# Author: T. E. Weiss
# Date modified: May 3, 2020
#
# This script generates fake data, then analyzes the data in Stan to infer posteriors.
# Pathnames configured for running from: scripts/Phase-II-official-analysis/fake-data-study/
#

"""
To-do:
    - In FakeExperimentEnsemble, add:
        1. Tracking of convergence issues, so that a summary of problems can be saved/printed
        2. An option to parallelize with slurm instead of multiprocessing
    - Run morpho processors a) for pre-generation sampling and b) for continuous-variable calibration (once the processors have been tested sufficiently)
    - Improve some comments (e.g. describing Stan priors)
"""


#import unittest
from morpho.utilities import morphologging
logger = morphologging.getLogger(__name__)
import matplotlib.pyplot as plt

#Importing constants from mermithid
from mermithid.misc.Constants import *
from mermithid.processors.TritiumSpectrum.FakeDataGenerator import FakeDataGenerator
#Importing processors from morpho
from morpho.processors.sampling import PyStanSamplingProcessor, PriorSamplingProcessor
from morpho.processors.plots import APosterioriDistribution, Histo2dDivergence
from morpho.processors.IO import IOROOTProcessor, IORProcessor

#Defining processors
priorSampler = PriorSamplingProcessor("sample")
specGen = FakeDataGenerator("specGen")
writerProcessor = IOROOTProcessor("writer")
readerProcessor = IOROOTProcessor("reader")
rReaderProcessor = IORProcessor("reader")
analysisProcessor = PyStanSamplingProcessor("analyzer")
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
        'KEmax': QT2()+1000.,
        'Nscatters': 20
        },
    "priors": [
        {'name': 'Q', 'prior_dist': 'normal', 'prior_params': [QT2(), 0.07]},
        {'name': 'mass', 'prior_dist': 'gamma', 'prior_params': [1.135, 0.4344]},
        {'name': 'KEmin', 'prior_dist': 'normal', 'prior_params': [16323., 5.]},
        {'name': 'sigma', 'prior_dist': 'normal', 'prior_params': [18.6, 3.]},
        {'name': 'S', 'prior_dist': 'normal', 'prior_params': [3300, 57.45]},
        {'name': 'B_1kev', 'prior_dist': 'gamma', 'prior_params': [0.25, 0.4]},
        {'name': 'scattering_prob', 'prior_dist': 'beta', 'prior_params': [55, 17]},
        {'name': 'err_from_B', 'prior_dist': 'gamma', 'prior_params': [10, 0.05]}
        ]
    }
    
    inputs_writer_config = {
    "action": "write",
    "tree_name": "input",
    "filename": root_file,
    "variables": [
        {"variable": "KEmax", "type": "float"},
        {"variable": "Nscatters", "type":"int"},
        {"variable": "Q", "type": "float"},
        {"variable": "mass", "type": "float"},
        {"variable": "KEmin", "type": "float"},
        {"variable": "sigma", "type": "float"},
        {"variable": "S", "type": "float"},
        {"variable": "B_1kev", "type": "float"},
        {"variable": "scattering_prob", "type": "float"},
        {"variable": "err_from_B", "type": "float"},
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


def GenerateFakeData(inputs_dict, root_file="./results/tritium_analysis.root"):
    """
    Generates fake Phase II tritium beta spectrum data, saves the data to a root file, and plots it.
    
    Arguments:
    - inputs_dict: dictionary of parameter values inputted to the fake data generator.
    
   Returns:
   - results: dict with
        1) keys: K (energy), F (frequency)
        2) mapped values: energy, frequency
    """
    specGen_config = {
        "apply_efficiency": True,
        "efficiency_path": "../phase2_detection_efficiency_curve/combined_energy_corrected_count_rates/combined_energy_corrected_eff_at_quad_trap_frequencies.json",
        "detailed_or_simplified_lineshape": "detailed",
        "return_frequency": True,
        "Q": inputs_dict["Q"],
        "mass": inputs_dict["mass"],
        "Kmin": inputs_dict["KEmin"],
        "Kmax": inputs_dict["KEmax"],
        "scattering_sigma": inputs_dict["sigma"],
        "S": inputs_dict["S"],
        "B_1kev": inputs_dict["B_1kev"],
        "scattering_prob": inputs_dict["scattering_prob"],
        "err_from_B": inputs_dict["err_from_B"],
        "Nscatters": inputs_dict["Nscatters"],
        "B_field": 0.9578186017836624,
        "n_steps": 60000,
    }
    
    data_writer_config = {
    "action": "write",
    "tree_name": "gen_results",
    "file_option": "update",
    "filename": root_file,
    "variables": [
        {"variable": "K", "root_alias":"KE", "type": "float"},
        {"variable": "F", "type": "float"}
    ]}
    
    #Configuration step
    specGen.Configure(specGen_config)
    writerProcessor.Configure(data_writer_config)
    #Generate data
    specGen.Run()
    results = specGen.results
    #Save data points
    writerProcessor.data = results
    writerProcessor.Run()
    
    #Plot histograms of generated data
    Kgen = results['K']
    Fgen = results['F']

    plt.figure(figsize=(7, 7))
    plt.subplot(211)
    n, b, _ = plt.hist(Kgen, bins=100, label='Fake data')
    plt.plot(specGen.Koptions, specGen.probs/(specGen.Koptions[1]-specGen.Koptions[0])*(b[1]-b[0])*len(Kgen), label='Model')
    plt.xlabel('K [eV]')
    plt.ylabel('N')
    plt.legend()

    plt.subplot(212)
    n, b, p = plt.hist(Fgen, bins=100, label='Fake data')
    plt.xlabel('F [Hz]')
    plt.ylabel('N')
    plt.legend()

    plt.tight_layout()
    plt.savefig('GeneratedData.png', dpi=100)
    
    return results


def StanTritiumAnalysis(data, fit_parameters=None, root_file='./results/tritium_analysis.root', F_or_K='F', stan_files_location='../../morpho_models/', model_code='tritium_model/models/tritium_phase_II_analyzer_unbinned.stan', scattering_params_R='simplified_scattering_params.R'):
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
        #***Add "temp_" when I switch to cmdstan.***
        "model_name": stan_files_location+"tritium_model/models/tritium_phase_II_analyzer_unbinned",
        "cache_dir": stan_files_location+"tritium_model/cache",
        "warmup": 50,
        "iter": 100,
        "chain": 1,
        "control": {'adapt_delta':0.90},
        "init": {
            "sigma": 18.6,
            "Q": 18573.24,
            "mass": 0.2,
            "KEmin": 16323.,
            "S": 3300.,
            "B_1kev": 0.1,
            "B": 0.35,
            "A2": 9762.5
        },
        "input_data": {
            "KEmin_ctr": 16323.,
            "KEmin_std": 5.,
            "KEmax": 19573.24,
            "sigma_ctr": 18.6,
            "sigma_std": 3.,
            "err_from_B": 0.5,
            "Q_ctr": QT2(),
            "Q_std": 75., #350
            "m_alpha": 1.135,
            "m_beta": 2.302,
            "S_ctr": 3300.,
            "S_std": 100, #50.
            "B_1kev_alpha": 0.25,
            "B_1kev_beta": 2.5,
            "slope": 0.000390369173, #1.
            "intercept": -6.00337656, #20380.5153
            "scattering_prob": 0.77,
            "Nscatters": 20
        },
        "interestParams": ['Q', 'mass', 'KEmin', 'sigma', 'S', 'B_1kev', 'KE_sample', 'spectrum_fit'],
    }
    
    posteriors_writer_config = {
    "action": "write",
    "tree_name": "analysis",
    "file_option": "update",
    "filename": root_file,
    "variables": [
        {"variable": "Q", "type": "float"},
        {"variable": "mass", "type": "float"},
        {"variable": "KEmin", "type": "float"},
        {"variable": "sigma", "type": "float"},
        {"variable": "B_1kev", "type": "float"},
        {"variable": "S", "type": "float"},
        {"variable": "KE_sample", "type": "float"},
        {"variable": "spectrum_fit", "type": "float"},
        {"variable": "divergent__", "root_alias": "divergence", "type": "float"},
        {"variable": "energy__", "root_alias": "energy", "type": "float"},
        {"variable": "lp_prob", "root_alias": "lp_prob", "type": "float"}
    ]}
    
    #Configuration step
    rReaderProcessor.Configure(scattering_reader_config)
    analysisProcessor.Configure(analyzer_config)
    writerProcessor.Configure(posteriors_writer_config)

    #Make data and scattering parameters accessible to analyzers
    analysisProcessor.data = {'N_data': [len(data[F_or_K])]} #List format here will be changed
    if F_or_K == 'K':
        analysisProcessor.data = {'KE':data['K']}
    elif F_or_K == 'F':
        analysisProcessor.data = {'F':data['F']}
    else:
        logger.warning("Input 'F_or_K'='F' for frequency or 'K' for energy.")
    rReaderProcessor.Run()
    analysisProcessor.data = rReaderProcessor.data

    #Run analysis
    analysisProcessor.Run()
    results = analysisProcessor.results
    
    #Save results
    writerProcessor.data = results
    writerProcessor.Run()
    
    return results


def PlotStanResults(posteriors, correlation_vars=['Q', 'KEmin', 'lp_prob'], divergence_vars=['Q', 'sigma', 'mass']):
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
        "n_bins_x": 50,
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


def PerformFakeExperiment(root_filename, plot_results=False, parallelized=True):
    """
    Generate fake tritium data and analyze it in Stan. If plot_results==True, create correlation and divergence plots.
    
    Saves generation inputs, generation results, and analysis results to different trees of one ROOT file.
    """
    if parallelized==True:
        flist = root_filename.split('/')
        logger.info("----------MORPHO RUN #{}----------".format(flist[len(flist)-1][0]))
    
    #Sample inputs to the data generator and save to one branch of a root file:
    inputs_dict = DefineGeneratorInputs(root_filename)
    
    #If fixing all inputs:
    #inputs_dict = {"Q":QT2(), "mass":0.2, "KEmin":16323, "KEmax":19573.24, "sigma":18.6, "S":3300, "B_1kev":0.1, "scattering_prob":0.77, "err_from_B":0.5, "Nscatters":20}
    
    #Generate data using the inputs and save data
    tritium_data = GenerateFakeData(inputs_dict, root_filename)
    
    #Analyze data and save posteriors
    posteriors = StanTritiumAnalysis(tritium_data, root_file=root_filename, F_or_K='K')
    
    #Optionally plot posteriors
    if plot_results == True:
        PlotStanResults(posteriors)
        
    #Dictionary containing Stan convergence diagnostics information
    #return analysis_diagnostics


def FakeExperimentEnsemble(n_runs, root_basename, wait_before_runs=0, parallelize=True, n_processes=4):
    """
    To-do: add parallelization option for a Slurm environment.
    
    n_runs: int; number of pseudo-experiments to be performed
    root_basename: str; results are saved in rootfiles labeled by root_basename and the pseudo-experiment number
    wait_before_runs: int or float; number of seconds to pause between the start of one pseudo-experiment and the start of the next
    """
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
    #_print_diag_summary(analysis_diagnostics)
    
    return root_filenames


if __name__ == '__main__':
    root_files = FakeExperimentEnsemble(1, "fake_data_and_results.root", wait_before_runs=30)
    
    
