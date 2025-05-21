import numpy as np
from src.utils.plot import *
from scripts.experiments import *
from tests.tests import *
from src.utils.visualise_data import *
import cProfile
import sys

if __name__ == '__main__':
    
    # visualise_coverage("CoverageProblem", "log_linear_potentials.pckl", "modified_log_linear_potentials.pckl", True, "Paper/Experiments", "coverage_fixed", "modifiedcoverage_fixed")
    # runExperiments()
    
    test_log_linear()
    
