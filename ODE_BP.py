from fitting_functions.traditional_methods import MeyerLoweryFitting
from fitting_functions.ODE_solving import PetRAFTKineticFitting
from fitting_functions.traditional_methods import MeyerLoweryFitting
import pandas as pd

# read in data
data = pd.read_excel('/Users/bridgettepoff/Library/CloudStorage/OneDrive-UniversityofNorthCarolinaatChapelHill/Knight Lab -  Bridgette/kinetics_BP/kinetics_all_meyerlowry.xlsx', header=0)

# initialize fitting object
f = PetRAFTKineticFitting(data, 108.08, 38.37) # add your feed ratios

##### functions to assist in fitting and visualizing data #

# visualize the conversion plot with just some test values of r_1 and r_2
# f.test_values(1, 1)

# display the overlay of your data with a test 
# f.display_overlay([1, 1])

# extract rates using guess data
f.extract_rates(1, 1)