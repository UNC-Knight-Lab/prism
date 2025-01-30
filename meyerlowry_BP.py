from fitting_functions.ODE_solving import PetRAFTKineticFitting
from fitting_functions.traditional_methods import MeyerLoweryFitting
import pandas as pd

# read in data
data = pd.read_excel('/Users/bridgettepoff/Library/CloudStorage/OneDrive-UniversityofNorthCarolinaatChapelHill/Knight Lab -  Bridgette/kinetics_BP/kinetics_all_meyerlowry.xlsx', header=0)

# extract data using Meyer Lowery fitting
f = MeyerLoweryFitting() #initialize object
f.extract_rates(data) #extract r1 and r2