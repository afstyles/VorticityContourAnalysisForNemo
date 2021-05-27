#  _   _ _____  ___  _   _ 
# | | | /  __ \/ _ \| \ | |
# | | | | /  \/ /_\ \  \| |
# | | | | |   |  _  | . ` |
# \ \_/ / \__/\ | | | |\  |
#  \___/ \____|_| |_|_| \_/
                         
"""
VCAN.py

Executable script

Run this script to calculate any of the following

- Depth integrated stream function
- Barotropic vorticity diagnostics
- Contour integrals

On NEMO data

"""

import VCAN
import sys
import os

data_dir = sys.argv[1]
nn_res = float(sys.argv[2])

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#Input settings 
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

out_label = "VCAN_test"  #Label for output folder

sf_zint_log = True   # = True --> calculate the barotropic stream function
vort_diag_log = True # = True --> calculate vorticity diagnostics
cont_int_log = True  # = True --> carry out contour area integrals of the vorticity diagnostics

#Details of the NEMO configuration
model = 'gyre'      # = 'gyre' --> Data comes from GYRE_PISCES reference configuration
                    # = 'global' --> Data comes from the global model

if model.lower() == "global": domcfg_path = os.path.abspath("path/to/domcfg/file/.nc") 
if model.lower() == "gyre":  domcfg_path = os.path.abspath(data_dir + '/mesh_mask.nc')

fscheme = 'een_1'   # = 'een_0' --> The configuration uses the EEN vorticity scheme with nn_een_e3f = 0
                    # = 'een_0' --> The configuration uses the EEN vorticity scheme with nn_een_e3f = 1
                    # Use een_1 if the model used the ENE or ENS vorticity scheme. Recreations of PVO are not exact but are very close.

# Minimum and maximum i and j indices
if model.lower() == "global":
    imin = int(850*(nn_res/4))
    imax = None
    jmin = int(150*(nn_res/4))
    jmax = int(500*(nn_res/4))

if model.lower() == "gyre":
    imin = None
    imax = None
    jmin = None
    jmax = None

#Contour integration settings

#Number of streamline values to integrate within
nlevels = 1001

#Region to find contours within
if model.lower() == 'global' : lonlatbounds = (-72.0,72.0,-84.0,-42.0)
if model.lower() == 'gyre' : lonlatbounds = (-86,-50,20,50)

#Interpolation settings
interp = 'linear' # = 'linear' for linear interpolation, = None for no interpolation
interp_res = 1/12.0 # Lon-Lat resolution to interpolate to if interp=='linear'
level_min = False #Minimum streamline to integrate. == False to use smallest value
level_max = False #Maxumums streamlines to integrate witrhin. == False to use largest value
R = 6400e+3 #Radius of the earth in metres

#Run the analysis code in VCAN.py
VCAN.save_namelist(data_dir, nn_res, out_label, sf_zint_log, vort_diag_log, cont_int_log, model, domcfg_path, fscheme, imin, imax, jmin, jmax, nlevels, lonlatbounds, interp, interp_res, level_min, level_max, R)
VCAN.run_VCAN(data_dir, out_label, sf_zint_log, vort_diag_log, cont_int_log, model, domcfg_path, fscheme, imin, imax, jmin, jmax, nlevels, lonlatbounds, interp, interp_res, level_min, level_max, R)
