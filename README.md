# prism
Welcome to the PRISM (Polymer Rate Insights and Sequence Modeling) package, offered by the Knight Lab at UNC Chapel Hill! This script was developed by Supraja Chittari (suprajac@email.unc.edu) and is compiled as a Python package.

## Installation Instructions
A version of Python >= 3.7 is required to use this package. We recommend using [Anaconda](https://www.anaconda.com) to install Python for those new to Python.
1. Open the terminal (MacOS) or Command Prompt (Windows).
2. Download the package by either:
   1. Download the zip from GitHub (Code -> Download ZIP). Unzip the package somewhere (note the extraction path). The extracted package can be deleted after installation.
   2. Clone this repository (requires git to be installed) with:
      
   `git clone https://github.com/UNC-Knight-Lab/prism.git`

3. Install the package using pip. This command will install this package to your Python environment.
    The package path should be the current working directory `.` if cloned using git. Otherwise, replace it with the path to the `prism` folder.
      
   `pip install .`
   or `pip install /path/to/package/prism`

That's it!

## How to use
We recommend users going through our associated publication to better understand and contextualize the capabilities of this package, which is divided into four main modules. These include fitting functions (which take polymerization kinetics data to return reactivity ratios), simulation functions (which take reactivity ratios to return an ensemble of sequences), and analysis functions (which take an ensemble of sequences and return statistics and visualizations.)

Further examples of functions and calls are delineated in Jupyer notebook files in the `examples` folder.
