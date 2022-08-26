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
from pickletools import int4


def run_VCAN(data_dir, out_label, VarDictName, ice_log, no_pen_masked_log, mask_method, intdirection, sf_zint_log, vort_diag_log, cont_int_log, domcfg_path, fscheme, imin, imax, jmin, jmax, nlevels, lonlatbounds, interp, interp_ref, level_min, level_max):
    """
    VCAN(data_dir, out_label, sf_zint_log, vort_diag_log, cont_int_log, model, fscheme, imin, imax, jmin, jmax, nlevels, lonlatbounds, inter, interp_res, level_min, level_max, R)

    Method that performs calculations with NEMO data based off a series of input variables set by the user in executable.py

    INPUT VARIABLES
    data_dir      - String, path to directory of the model output files
    out_label     - String, label for saved output files
    sf_zint_log   - Calculate the depth-integrated stream function if == True
    vort_diag_log - Calculate the barotropic vorticity diagnostics if == True
    cont_int_log  - Integrate barotropic vorticity diagnostics over area enclosed by streamlines if == True
    model         - String, describing the source model for the data
                        =="global" data comes from the NEMO global model using an ORCA grid
                        =="gyre" data comes from the GYRE_PISCES reference configuration in NEMO
    
    fscheme       - Vorticity scheme used by the model
                    = 'een_0' --> The configuration uses the EEN vorticity scheme with nn_een_e3f = 0
                    = 'een_0' --> The configuration uses the EEN vorticity scheme with nn_een_e3f = 1
                    Use een_1 if the model used the ENE or ENS vorticity scheme. Recreations of PVO are not exact but are very close.

    imin, imax    - Minimum/maximum i index to carry out analysis over. Set both to None to use the full domain
    jmin, jmax    - Minimum/maximum j index to carry out analysis over. Set both to None to use the full domain

    nlevels       - Integer, number of streamline values to integrate over if cont_int_log == True
    lonlatbounds  - Tuple of longitudes and latitudes to describe a rectangular subregion of interest for contour integration
                    (lon_min, lon_max, lat_min, lat_max)  [deg]
                    Set to None if you wish to use the full 
                    
    interp - String, == 'linear' to use linear interpolation before contour integrating
                     == None to not use any interpolation

    interp_res - Float, lon-lat resolution to interpolate to if inter !=  [deg]
    level_min - Float, minimum value of streamline to integrate within [Sv]
    level_max - Float, maximum value of streamline to integrate within [Sv]
    R - Radius of the earth [m]

    """
    import iris
    import numpy as np
    import pickle
    import sys
    import os

    sys.path.append(os.path.abspath("../lib"))

    import CGridOperations
    from cube_prep import CubeListExtract, apply_mask, coord_repair, GenVarDict, get_mask
    import ContourInt
    import VortDiagnostics

    data_list = iris.load(data_dir + "/*grid*.nc")
    
    grid_list = iris.load(domcfg_path)
    
    VarDict = GenVarDict(VarDictName=VarDictName) #Load the dictionary of variable names that is model specific

    print("Carrying our following operations >>>")
    print(f"sf_zint_log = {sf_zint_log}")
    print(f"vort_diag_log = {vort_diag_log}")
    print(f"cont_int_log = {cont_int_log}")

    print("Using the following variable name dictionary in cube_prep.py >>>")
    print(f"VarNameDict = {VarDictName}")
    
    print("NEMO config details have been set to >>>")
    print(f"ice_log = {ice_log}")
    print(f"fscheme = {fscheme}")
    
    if cont_int_log == True:
        print("Contour integration settings are >>>")
        print(f"interp = {interp}")
        print(f"1/interp_res = {1/interp_ref}")
        print(f"level_min = {level_min}")
        print(f"level_max = {level_max}")
        
    print(">>>")    

    #Create output folder in data directory
    out_dir = os.path.abspath(f"{data_dir}/OUTPUT_{out_label}/")

    if not os.path.exists(out_dir): os.mkdir(out_dir)

    #Extract the masks needed for the analysis
    MaskDict = {}
    for label in ['tmask', 'umask', 'vmask', 'tmaskutil', 'umaskutil', 'vmaskutil']:
        #Create a mask dictionary of the boolean arrays listed above
        # T = Unmasked, F = Masked
        MaskDict[label] = get_mask( data_list, grid_list, label, VarDict, method=mask_method )
    

    if sf_zint_log == True:
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #Calculate the depth-integrated stream function 
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        print("Calculating the depth integrated stream function")

        sf_zint_cube = ContourInt.DepIntSF_orca(data_list, grid_list, VarDict, MaskDict, intdirection=intdirection)

        iris.save(sf_zint_cube, out_dir + f'/sf_zint.{out_label}.nc' )

    if vort_diag_log == True:
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #Calculate the barotropic vorticity diagnostics 
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        print("Calculating the barotropic vorticity diagnostics")

        #Calculate components of PVO momentum diagnostic
        PVO_mdecomp_dict = {}
        mdecomp_opts = { 'data_list':data_list, 'grid_list':grid_list,
                             'VarDict':VarDict, 'MaskDict':MaskDict  , 
                             'no_pen_masked_log':no_pen_masked_log, 'fscheme':fscheme.lower() }

        PVO_mdecomp_dict['u_bet'], PVO_mdecomp_dict['v_bet']  = VortDiagnostics.PVO_betcalc(  **mdecomp_opts )    
        PVO_mdecomp_dict['u_prc'], PVO_mdecomp_dict['v_prc']  = VortDiagnostics.PVO_prccalc(  **mdecomp_opts )
        PVO_mdecomp_dict['u_nul'], PVO_mdecomp_dict['v_nul']  = VortDiagnostics.PVO_nulcalc(  **mdecomp_opts )  
        PVO_mdecomp_dict['u_pvo2'], PVO_mdecomp_dict['v_pvo2']= VortDiagnostics.PVO_fullcalc( **mdecomp_opts )

        PVO_mdecomp_list = []
        for label in PVO_mdecomp_dict:
            PVO_mdecomp_list = PVO_mdecomp_list + [PVO_mdecomp_dict[label]]

        iris.save(PVO_mdecomp_list, out_dir + f'/PVO_mdecomp.{out_label}.nc')

        if ice_log == True:
            vort_diag_dict = VortDiagnostics.VortDiagnostic2D(data_list, grid_list, VarDict, MaskDict, PVO_mdecomp_dict, icelog = True, uice_cube = u_ice_cube, vice_cube = v_ice_cube)
            
        else:
            vort_diag_dict = VortDiagnostics.VortDiagnostic2D(data_list, grid_list, VarDict, MaskDict, PVO_mdecomp_dict, icelog = False)
        
        vort_diag_list = []
        for vort_label in vort_diag_dict:
            vort_diag_list = vort_diag_list + [vort_diag_dict[vort_label]]
            
        iris.save(vort_diag_list, out_dir + f"/vort_2D.{out_label}.nc")
                                                            

    if cont_int_log == True:
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #Integrate vorticity diagnostics over areas
        #enclosed by steamlines 
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        print("Performing integrals over areas enclosed by streamlines")

        if sf_zint_log == False:
            print("Loading depth integrated streamfunction")

            sf_zint_cube = iris.load(out_dir + "/sf_zint*.nc")[0]

        if vort_diag_log == False:
            print("Loading barotropic vorticity diagnostics")

            vort_diag_list = iris.load(out_dir + "/vort_2D.*.nc")

            vort_diag_dict = { '_keg' :CubeListExtract(vort_diag_list, 'curl_keg_zint'),
                    '_rvo' :CubeListExtract(vort_diag_list, 'curl_rvo_zint'),
                    '_pvo' :CubeListExtract(vort_diag_list, 'curl_pvo_zint'),
                    '_hpg' :CubeListExtract(vort_diag_list, 'curl_hpg_zint'),
                    '_ldf' :CubeListExtract(vort_diag_list, 'curl_ldf_zint'),
                    '_zad' :CubeListExtract(vort_diag_list, 'curl_zad_zint'),
                    '_tot' :CubeListExtract(vort_diag_list, 'curl_tot_zint'),
                    '_wnd' :CubeListExtract(vort_diag_list, 'curl_wnd_zint'),
                    '_frc' :CubeListExtract(vort_diag_list, 'curl_frc_zint'),
                    '_fdu' :CubeListExtract(vort_diag_list, 'curl_fdu_zint'),
                    '_mfdu' :CubeListExtract(vort_diag_list, 'curl_mfdu_zint'),
                    '_div' :CubeListExtract(vort_diag_list, 'curl_div_zint'),
                    '_mlv' :CubeListExtract(vort_diag_list, 'curl_mlv_zint'),
                    '_bet' :CubeListExtract(vort_diag_list, 'curl_bet_zint'),
                    '_prc' :CubeListExtract(vort_diag_list, 'curl_prc_zint')}
            
            if ice_log == True: vort_diag_dict['_ice'] = CubeListExtract(vort_diag_list, 'curl_ice_zint')
            
        else:
            #Remove diagnostics we don't want to explicitly contour integrate
            del vort_diag_dict['_adv']
            del vort_diag_dict['_zdf']
            del vort_diag_dict['_res']
            del vort_diag_dict['_vorticity']


        keywords = {'lonlatbounds':lonlatbounds, 'interpolation':interp, 'ref':interp_ref, 'nlevels':nlevels, 'level_min':level_min, 'level_max':level_max }
        Ni_cubes_out_dict, contour_masks_cube, enclosed_area_cube = ContourInt.niiler_integral2D(vort_diag_dict, sf_zint_cube, grid_list, VarDict, MaskDict, **keywords)
        
        iris.save(contour_masks_cube, f"{out_dir}/NI_contour_masks.{out_label}.nc")     

        Ni_cube_list = [enclosed_area_cube]
        for Ni_label in Ni_cubes_out_dict:
            Ni_cube_list = Ni_cube_list + [Ni_cubes_out_dict[Ni_label]]


        #Also calculate integrals that are linear combinations of others
        print("Calculating _adv")
        NI_ADV_cube = ContourInt.NI_ADV_calc(Ni_cubes_out_dict['_keg'], Ni_cubes_out_dict['_rvo'], Ni_cubes_out_dict['_zad'])
        
        print("Calculating _zdf")
        if ice_log == True:
            NI_ZDF_cube = ContourInt.NI_ZDF_calc(Ni_cubes_out_dict['_wnd'], Ni_cubes_out_dict['_frc'], icelog=True, Ni_ice_cube=Ni_cubes_out_dict['_ice'] )
        else:
            NI_ZDF_cube = ContourInt.NI_ZDF_calc(Ni_cubes_out_dict['_wnd'], Ni_cubes_out_dict['_frc'], icelog=False)
        
        print("Calculating _res")
        NI_RES_cube = ContourInt.NI_RES_calc(NI_ADV_cube             , Ni_cubes_out_dict['_pvo'], Ni_cubes_out_dict['_hpg'],
                                             Ni_cubes_out_dict['_ldf'], NI_ZDF_cube             , Ni_cubes_out_dict['_tot'])

        iris.save(Ni_cube_list + [NI_ADV_cube, NI_ZDF_cube, NI_RES_cube], out_dir + f'/NI_2D_vort.{out_label}.nc')
        

    return


def save_namelist(data_dir, out_label, VarDictName, ice_log, no_pen_masked_log, mask_method, intdirection, sf_zint_log, vort_diag_log, cont_int_log, domcfg_path, fscheme, imin, imax, jmin, jmax, nlevels, lonlatbounds, interp, interp_ref, level_min, level_max):
    """
    Generate an output file that keeps a record of all input variables
    """
    f = open(f"{data_dir}/namelist.{out_label}.out", "w")

    f.write(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
    f.write("Input variables for VCAN\n")
    f.write(f"{out_label}\n")
    f.write(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
    f.write(" \n")
    f.write(f"data_dir = {data_dir}\n")
    f.write(f"VarDictName = {VarDictName}\n")
    f.write(f"ice_log = {ice_log}\n")
    f.write(f"no_pen_masked_log = {no_pen_masked_log}\n")
    f.write(f"mask_method = {mask_method}\n")
    f.write(f"sf_zint_log = {sf_zint_log}\n")
    f.write(f"vort_diag_log = {vort_diag_log}\n")
    f.write(f"cont_int_log = {cont_int_log}\n")
    f.write(f"domcfg_path = {domcfg_path}\n")
    f.write(f"fscheme = {fscheme}\n")
    f.write(f"imin = {imin}\n")
    f.write(f"imax = {imax}\n")
    f.write(f"jmin = {jmin}\n")
    f.write(f"jmax = {jmax}\n")
    f.write(f"nlevels = {nlevels}\n")
    f.write(f"lonlatbounds = {lonlatbounds}\n")
    f.write(f"interp = {interp}\n")
    f.write(f"interp_ref = {interp_ref}\n")
    f.write(f"level_min = {level_min}\n")
    f.write(f"level_max = {level_max}\n")
    f.write(f" \n")
    f.write(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    f.close()

    return

