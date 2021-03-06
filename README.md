# Vorticity Contour Analysis For Nemo (VCAN)

This software can be used on NEMO output data to study the barotropic vorticity budget of gyre systems. 

The code has been tested on data from the NEMO global model and the GYRE model but could be adapted to use on other configurations.

### Main features

* Calculate the depth-integrated stream function
* Calculate vorticity diagnostics from NEMO's momentum diagnostics
* Integrate the vorticity diagnostics over the areas enclosed by calculated streamlines.
* Plot the results of these calculations in an iPython notebook

### Structure
There are three folders in this repository

`exe/` - Containing the files:
* `executable.py` which can be edited to determine the parameters for the analysis. When you want to run the analysis, execute the following command from the exe folder

```
cd exe
python executable.py /path/to/data/directory/ n
```
Where n is the inverse resolution of the gridded data. For example, to analyse data on an ORCA025 (1/4 degree) grid you would run

```
cd exe
python executable.py /path/to/data/directory/ 4
```

* VCAN.py -The functions in `VCAN.py` are called by `executable.py` and is the main body of the program. This script shouldn't need to be edited unless you aim to significantly alter the analysis.

`lib/` - Contains the files:
* `CGridOperations.py` - Functions for differentiation on a C grid and rolling arrays appropriately
* `ContourInt.py` - Functions for integrating vorticity diagnostics over areas enclosed by streamlines
* `cube_prep.py` - Operations for loading NEMO data correctly and preserving metadata
* `VortDiagnostics.py` - Operations for calculating vorticity diagnostics and decomposing the contribution due to planetary vorticity

`int/` - Contains interactive notebooks:
* `plotting.ipynb` - Plots the depth-integrated stream function, the vorticity diagnostics and contour integration results.

# Final comments

This repository is still very much a work in progress. The code is still being tested to improve performance and versatility and any feedback would be greatly appreciated.

In the future I will add details on how to ideally configure NEMO for this analysis and add more iPython notebooks to demonstrate the methods used.
