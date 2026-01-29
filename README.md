# prism
Welcome to the PRISM (Polymer Rate Insights and Sequence Modeling) package, offered by the Knight Lab at UNC Chapel Hill! This script was developed by Supraja Chittari (supraja.chittari@gmail.com) and is compiled as a Python package.

## Table of Contents
- [Installation](#installation)
- [Documentation](#documentation)
- [Usage](#usage)
- [Instrumentation](#instrumentation)

## Installation Instructions
A version of Python >= 3.10 is required to use this package. We recommend using [Anaconda](https://www.anaconda.com) to install Python for those new to Python.
1. Open the terminal (MacOS) or Command Prompt (Windows).
2. Download the package by either:
   1. Download the zip from GitHub (Code -> Download ZIP). Unzip the package somewhere (note the extraction path). The extracted package can be deleted after installation. Navigate to this directory.
   2. Create a virtual directory by running `conda create --name myenv` and replacing `myenv` with the name of your desired environment. Then activate the environment using `conda activate myenv`.
   3. Clone this repository (requires git to be installed) with:
      
   `git clone https://github.com/UNC-Knight-Lab/prism.git`

3. Install the package using pip. This command will install this package and required dependencies to your Python environment.
    The package path should be the current working directory `.` if cloned using git. Otherwise, replace it with the path to the `prism` folder.
      
   `pip install .`
   or `pip install /path/to/package/prism`

That's it!

## Usage

### Sample codes and data
We recommend users going through our associated publication to better understand and contextualize the capabilities of this package, which is divided into four main modules. These include fitting (which take polymerization kinetics data to return reactivity ratios), simulation functions (which take reactivity ratios to return an ensemble of sequences), and analysis functions (which take an ensemble of sequences and return statistics and visualizations.) Examples for these modules are included in as Jupyer notebooks in the `examples` folder. Sample data that is called for these functions in the appropriate input formats are included in the `sample data` folder. 

### Publication and data archive
For more information, we point you to our publication: "Bottom-Up Simulation, Reconstruction, and Quantification of Macromolecule Sequences from Experimental Polymerizations." Macromolecules 2025, 58, 20, 11029–11039. DOI: 10.1021/acs.macromol.5c02363. All raw data in our paper is deposited in the following repository: https://dataverse.unc.edu/dataverse/polymerization-sequences
