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

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#Input settings 
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

out_label = "VCAN_test"  #Label for output folder

sf_zint_log = True   # = True --> calculate the barotropic stream function
vort_diag_log = True  # = True --> calculate vorticity diagnostics
cont_int_log = True  # = True --> carry out contour area integrals of the vorticity diagnostics
domcfg_path = os.path.abspath(data_dir + '/mesh_mask.nc')

intdirection = 'N'  # Direction of integration for calculating the stream function
                    # 'N' for northwards integration (S --> N)
                    # 'E' for eastwards integration (W --> E)
                    # 'S' for southwards integration (N --> S)
                    # 'W' for westwards integration (E --> W)
                    # In a system with closed boundaries on all sides, the streamfunction should be similar 
                    # in all four cases. Accunulating errors from divergences in the depth-integrated flow 
                    # will be different however

fscheme = 'een_0'   # = 'een_0' --> The configuration uses the EEN vorticity scheme with nn_een_e3f = 0
                    # = 'een_1' --> The configuration uses the EEN vorticity scheme with nn_een_e3f = 1
                    # = 'ens' --> The configuration uses the ENS vorticity scheme
                    # = 'ene' --> The configuration uses the ENE vorticity scheme

VarDictName = 'gyre'  #Name of the variable name dictionary to use. Included dictionaries are listed below
                      # gyre --> Designed for the gyre_pisces configuration
                      # global --> Designed for the global model configuration

ice_log = False        # If icelog == True --> VCAN will search for diagnostics associated with sea ice stress
                      #           == False --> VCAN will use total surface stress instead (tau = wind + ice)

no_pen_masked_log = True  # Select masking convention used. ==  True if incident (zero value) velocity points are masked in data files
                         #                                 == False if incident (zero value) velocity points are unmasked in data files

mask_method = 'detached'   # State if masks are attached to data files. == 'detached' --> Masks will be extracted from data files
                           #                                            == 'attached' --> Masks will be loaded from the domcfg_path file

# Minimum and maximum i and j indices
imin = None
imax = None
jmin = None
jmax = None

#Contour integration settings
#Number of streamline values to integrate within
nlevels = 201

#Region to find contours within
lonlatbounds = (-86,-50,20,50)

#Interpolation settings
interp = 'linear' # = 'linear' for linear interpolation, = None for no interpolation
interp_ref = 2 # Grid refinement factor for interpolation (interp_ref = 2 --> Halve cell thicknesses)
level_min = 0 #Minimum streamline to integrate. == None to use smallest value
level_max = None #Maxumums streamlines to integrate witrhin. == None to use largest value

#Run the analysis code in VCAN.py
VCAN.save_namelist(data_dir, out_label, VarDictName, ice_log, no_pen_masked_log, mask_method, intdirection, sf_zint_log, vort_diag_log, cont_int_log, domcfg_path, fscheme, imin, imax, jmin, jmax, nlevels, lonlatbounds, interp, interp_ref, level_min, level_max)
VCAN.run_VCAN(data_dir, out_label, VarDictName, ice_log, no_pen_masked_log, mask_method, intdirection, sf_zint_log, vort_diag_log, cont_int_log, domcfg_path, fscheme, imin, imax, jmin, jmax, nlevels, lonlatbounds, interp, interp_ref, level_min, level_max)
