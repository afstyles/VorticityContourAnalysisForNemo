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
def run_VCAN(data_dir, out_label, sf_zint_log, vort_diag_log, cont_int_log, model, domcfg_path, fscheme, imin, imax, jmin, jmax, nlevels, lonlatbounds, interp, interp_res, level_min, level_max, R):
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
    from cube_prep import CubeListExtract, apply_mask, coord_repair
    import ContourInt
    import VortDiagnostics

    data_list = iris.load(data_dir + "/*grid*.nc")
    
    grid_list = iris.load(domcfg_path)
    
    print("Carrying our following operations >>>")
    print(f"sf_zint_log = {sf_zint_log}")
    print(f"vort_diag_log = {vort_diag_log}")
    print(f"cont_int_log = {cont_int_log}")
    
    print("NEMO config details have been set to >>>")
    print(f"model = {model}")
    print(f"fscheme = {fscheme}")
    
    if cont_int_log == True:
        print("Contour integration settings are >>>")
        print(f"interp = {interp}")
        print(f"1/interp_res = {1/interp_res}")
        print(f"level_min = {level_min}")
        print(f"level_max = {level_max}")
        
    print(">>>")

    
    

    i = 0
    for cube in data_list:
        if len(cube.shape) > 2:
            data_list[i] = cube[...,jmin:jmax, imin:imax]
        i = i + 1

    i = 0
    for cube in grid_list:
        if len(cube.shape) > 2:
            grid_list[i] = cube[...,jmin:jmax, imin:imax]
        i = i + 1

    # Load velocity and momentum diagnostic data.
    # Variable names vary between models
    if model.lower() == "global":
        u_cube = CubeListExtract(data_list, 'uo')
        v_cube = CubeListExtract(data_list, 'vo')

        if vort_diag_log == True:
            u_keg_cube = CubeListExtract(data_list, 'utrd_keg')
            u_rvo_cube = CubeListExtract(data_list, 'utrd_rvo')
            u_pvo_cube = CubeListExtract(data_list, 'utrd_pvo')
            u_hpg_cube = CubeListExtract(data_list, 'utrd_hpg')
            u_ldf_cube = CubeListExtract(data_list, 'utrd_ldf')
            u_zdf_cube = CubeListExtract(data_list, 'utrd_zdf')
            u_zad_cube = CubeListExtract(data_list, 'utrd_zad')
            u_tot_cube = CubeListExtract(data_list, 'utrd_tot')
            #u_tau_cube = CubeListExtract(data_list, 'tauuo')/1026
            #u_tau_cube = CubeListExtract(data_list, 'utrd_tau')/2
            u_tau_cube = CubeListExtract(data_list, 'utrd_tau2d_hu')
            u_ice_cube = CubeListExtract(data_list, 'utrd_tfr2d_hu')
            
            
            v_keg_cube = CubeListExtract(data_list, 'vtrd_keg')
            v_rvo_cube = CubeListExtract(data_list, 'vtrd_rvo')
            v_pvo_cube = CubeListExtract(data_list, 'vtrd_pvo')
            v_hpg_cube = CubeListExtract(data_list, 'vtrd_hpg')
            v_ldf_cube = CubeListExtract(data_list, 'vtrd_ldf')
            v_zdf_cube = CubeListExtract(data_list, 'vtrd_zdf')
            v_zad_cube = CubeListExtract(data_list, 'vtrd_zad')
            v_tot_cube = CubeListExtract(data_list, 'vtrd_tot')
            #v_tau_cube = CubeListExtract(data_list, 'tauvo')/
            #v_tau_cube = CubeListExtract(data_list, 'vtrd_tau')/2
            v_tau_cube = CubeListExtract(data_list, 'vtrd_tau2d_hv')
            v_ice_cube = CubeListExtract(data_list, 'vtrd_tfr2d_hv')

    if model.lower() == "gyre":
        # u_cube = CubeListExtract(data_list,'vozocrtx')
        # v_cube = CubeListExtract(data_list,'vomecrty')

        u_cube = CubeListExtract(data_list,'uoce')
        v_cube = CubeListExtract(data_list,'voce')

        if vort_diag_log == True:
            u_keg_cube = CubeListExtract(data_list, 'utrd_swkeg')
            u_rvo_cube = CubeListExtract(data_list, 'utrd_swrvo')
            u_pvo_cube = CubeListExtract(data_list, 'utrd_swpvo')
            u_hpg_cube = CubeListExtract(data_list, 'utrd_swhpg')
            u_ldf_cube = CubeListExtract(data_list, 'utrd_swldf')
            u_zdf_cube = CubeListExtract(data_list, 'utrd_swzdf')
            u_zad_cube = CubeListExtract(data_list, 'utrd_swzad')
            u_tot_cube = CubeListExtract(data_list, 'utrd_swtot')
            # u_tau_cube = CubeListExtract(data_list, 'sozotaux')/1026
            u_tau_cube = CubeListExtract(data_list, 'utrd_swtau')/2
            
            v_keg_cube = CubeListExtract(data_list, 'vtrd_swkeg')
            v_rvo_cube = CubeListExtract(data_list, 'vtrd_swrvo')
            v_pvo_cube = CubeListExtract(data_list, 'vtrd_swpvo')
            v_hpg_cube = CubeListExtract(data_list, 'vtrd_swhpg')
            v_ldf_cube = CubeListExtract(data_list, 'vtrd_swldf')
            v_zdf_cube = CubeListExtract(data_list, 'vtrd_swzdf')
            v_zad_cube = CubeListExtract(data_list, 'vtrd_swzad')
            v_tot_cube = CubeListExtract(data_list, 'vtrd_swtot')
            # v_tau_cube = CubeListExtract(data_list, 'sometauy')/1026
            v_tau_cube = CubeListExtract(data_list, 'vtrd_swtau')/2

        # Outputs from GYRE are not masked, we apply the masks here
        umask_cube = CubeListExtract(grid_list, 'umask')
        vmask_cube = CubeListExtract(grid_list, 'vmask')
        umaskutil_cube = CubeListExtract(grid_list, 'umaskutil')
        vmaskutil_cube = CubeListExtract(grid_list, 'vmaskutil')

        u_cube = apply_mask(u_cube, umask_cube)
        v_cube = apply_mask(v_cube, vmask_cube)

        if vort_diag_log == True:
            u_keg_cube = apply_mask(u_keg_cube, umask_cube)
            u_rvo_cube = apply_mask(u_rvo_cube, umask_cube)
            u_pvo_cube = apply_mask(u_pvo_cube, umask_cube)
            u_hpg_cube = apply_mask(u_hpg_cube, umask_cube)
            u_ldf_cube = apply_mask(u_ldf_cube, umask_cube)
            u_zdf_cube = apply_mask(u_zdf_cube, umask_cube)
            u_zad_cube = apply_mask(u_zad_cube, umask_cube)
            u_tot_cube = apply_mask(u_tot_cube, umask_cube)
            u_tau_cube = apply_mask(u_tau_cube, umaskutil_cube)

            v_keg_cube = apply_mask(v_keg_cube, vmask_cube)
            v_rvo_cube = apply_mask(v_rvo_cube, vmask_cube)
            v_pvo_cube = apply_mask(v_pvo_cube, vmask_cube)
            v_hpg_cube = apply_mask(v_hpg_cube, vmask_cube)
            v_ldf_cube = apply_mask(v_ldf_cube, vmask_cube)
            v_zdf_cube = apply_mask(v_zdf_cube, vmask_cube)
            v_zad_cube = apply_mask(v_zad_cube, vmask_cube)
            v_tot_cube = apply_mask(v_tot_cube, vmask_cube)
            v_tau_cube = apply_mask(v_tau_cube, vmaskutil_cube)
            
            if model == 'global':
                u_ice_cube = apply_mask(u_ice_cube, umaskutil_cube)
                v_ice_cube = apply_mask(v_ice_cube, vmaskutil_cube)

    # Perform a simple repair for the loaded data cubes. Rename the auxilliary time coordinate to "aux_time" instead of "time"
    u_cube = coord_repair(u_cube)
    v_cube = coord_repair(v_cube)

    if vort_diag_log == True:
        u_keg_cube = coord_repair(u_keg_cube)
        u_rvo_cube = coord_repair(u_rvo_cube)
        u_pvo_cube = coord_repair(u_pvo_cube)
        u_hpg_cube = coord_repair(u_hpg_cube)
        u_ldf_cube = coord_repair(u_ldf_cube)
        u_zdf_cube = coord_repair(u_zdf_cube)
        u_zad_cube = coord_repair(u_zad_cube)
        u_tot_cube = coord_repair(u_tot_cube)
        u_tau_cube = coord_repair(u_tau_cube)

        v_keg_cube = coord_repair(v_keg_cube)
        v_rvo_cube = coord_repair(v_rvo_cube)
        v_pvo_cube = coord_repair(v_pvo_cube)
        v_hpg_cube = coord_repair(v_hpg_cube)
        v_ldf_cube = coord_repair(v_ldf_cube)
        v_zdf_cube = coord_repair(v_zdf_cube)
        v_zad_cube = coord_repair(v_zad_cube)
        v_tot_cube = coord_repair(v_tot_cube)
        v_tau_cube = coord_repair(v_tau_cube)
        
        if model == 'global':
            u_ice_cube = coord_repair(u_ice_cube)
            v_ice_cube = coord_repair(v_ice_cube)

    #Load cell widths and thicknesses. Variable names should be consistent between models
    e1u_cube = CubeListExtract(grid_list, 'e1u')
    e2u_cube = CubeListExtract(grid_list, 'e2u')
    e1v_cube = CubeListExtract(grid_list, 'e1v')
    e2v_cube = CubeListExtract(grid_list, 'e2v')
    e1f_cube = CubeListExtract(grid_list, 'e1f')
    e2f_cube = CubeListExtract(grid_list, 'e2f')
    e3u_cube = CubeListExtract(grid_list, 'e3u_0')
    e3v_cube = CubeListExtract(grid_list, 'e3v_0')
    e3t_cube = CubeListExtract(grid_list, 'e3t_0')

    e1u = np.squeeze(e1u_cube.data)
    e2u = np.squeeze(e2u_cube.data)
    e1v = np.squeeze(e1v_cube.data)
    e2v = np.squeeze(e2v_cube.data)
    e1f = np.squeeze(e1f_cube.data)
    e2f = np.squeeze(e2f_cube.data)
    e3u = np.squeeze(e3u_cube.data)
    e3v = np.squeeze(e3v_cube.data)
    e3t = np.squeeze(e3t_cube.data)

    #Load values of the Coriolis parameter centred on f points
    ff_f_cube = CubeListExtract(grid_list, 'ff_f')
    ff_f = np.squeeze(ff_f_cube.data)

    #Load the mask for t points. The location of this data depends on the model used
    if model.lower() == "gyre": 
        tmask_cube = CubeListExtract(grid_list, 'tmask')
        tmask = ~np.ma.make_mask(np.squeeze(tmask_cube.data))

    if model.lower() == "global":
        tmask = np.ma.make_mask(np.squeeze(CubeListExtract(data_list, 'thetao_con').data.mask))

    #Create output folder in data directory
    out_dir = os.path.abspath(f"{data_dir}/OUTPUT_{out_label}/")

    if not os.path.exists(out_dir): os.mkdir(out_dir)


    if sf_zint_log == True:
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #Calculate the depth-integrated stream function 
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        print("Calculating the depth integrated stream function")

        sf_zint_cube = ContourInt.DepIntSF_orca(u_cube, e1u, e2u, e3u, e1f)

        iris.save(sf_zint_cube, out_dir + f'/sf_zint.{out_label}.nc' )

    if vort_diag_log == True:
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #Calculate the barotropic vorticity diagnostics 
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        print("Calculating the barotropic vorticity diagnostics")

        #Calculate components of PVO momentum diagnostic
        u_bet_cube, v_bet_cube = VortDiagnostics.PVO_betcalc(u_cube, v_cube, ff_f, e1u, e1v, e2u, e2v, e1f, e2f, e3u, e3v, tmask, model=model.lower(), fscheme=fscheme.lower() )
        u_prc_cube, v_prc_cube = VortDiagnostics.PVO_prccalc(u_cube, v_cube, ff_f, e1u, e1v, e2u, e2v, e1f, e2f, e3u, e3v, 
                                                            e3t, tmask, fscheme=fscheme.lower(), model=model.lower())
        
        u_nul_cube, v_nul_cube = VortDiagnostics.PVO_nulcalc(u_cube, v_cube, ff_f, e1u, e1v, e2u, e2v, e1f, e2f, e3u, e3v, tmask, model=model.lower(), fscheme=fscheme.lower())
        
        u_pvo2_cube, v_pvo2_cube = VortDiagnostics.PVO_fullcalc(u_cube, v_cube, ff_f, e1u, e1v, e2u, e2v, e1f, e2f, e3u, e3v, 
                                                            e3t, tmask, fscheme=fscheme.lower(), model=model.lower())

        iris.save(u_bet_cube, out_dir + f'/u_bet.{out_label}.nc')
        iris.save(v_bet_cube, out_dir + f'/v_bet.{out_label}.nc')
        iris.save(u_prc_cube, out_dir + f'/u_prc.{out_label}.nc')
        iris.save(v_prc_cube, out_dir + f'/v_prc.{out_label}.nc')
        iris.save(u_nul_cube, out_dir + f'/u_nul.{out_label}.nc')
        iris.save(v_nul_cube, out_dir + f'/v_nul.{out_label}.nc')
        iris.save(u_pvo2_cube,out_dir + f'/u_pvo2.{out_label}.nc')
        iris.save(v_pvo2_cube,out_dir + f'/v_pvo2.{out_label}.nc')

        if model == 'global':
            vort_diag_dict = VortDiagnostics.VortDiagnostic2D(u_cube, v_cube, u_keg_cube, u_rvo_cube, u_pvo_cube, u_hpg_cube,
                                                u_ldf_cube, u_zdf_cube, u_zad_cube, u_tot_cube, u_bet_cube, u_prc_cube, u_pvo2_cube, u_nul_cube, u_tau_cube,
                                                v_keg_cube, v_rvo_cube, v_pvo_cube, v_hpg_cube,
                                                v_ldf_cube, v_zdf_cube, v_zad_cube, v_tot_cube, v_bet_cube, v_prc_cube, v_pvo2_cube, v_nul_cube, v_tau_cube,
                                                ff_f, e3u, e3v, e3t, e1u, e2u, e1v, e2v, e1f, e2f, tmask, icelog = True, uice_cube = u_ice_cube, vice_cube = v_ice_cube)
            
        elif model == 'gyre':
            vort_diag_dict = VortDiagnostics.VortDiagnostic2D(u_cube, v_cube, u_keg_cube, u_rvo_cube, u_pvo_cube, u_hpg_cube,
                                                u_ldf_cube, u_zdf_cube, u_zad_cube, u_tot_cube, u_bet_cube, u_prc_cube, u_pvo2_cube, u_nul_cube, u_tau_cube,
                                                v_keg_cube, v_rvo_cube, v_pvo_cube, v_hpg_cube,
                                                v_ldf_cube, v_zdf_cube, v_zad_cube, v_tot_cube, v_bet_cube, v_prc_cube, v_pvo2_cube, v_nul_cube, v_tau_cube,
                                                ff_f, e3u, e3v, e3t, e1u, e2u, e1v, e2v, e1f, e2f, tmask, icelog = False)
        
        for vort_label in vort_diag_dict:
            iris.save(vort_diag_dict[vort_label], out_dir + f"/vort_2D_{vort_label}.{out_label}.nc")
                                                            

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

            vort_diag_list = iris.load(out_dir + "/vort_2D_*.nc")

            Ni_input_dict = { 'KEG' :CubeListExtract(vort_diag_list, 'curl_keg_zint'),
                    'RVO' :CubeListExtract(vort_diag_list, 'curl_rvo_zint'),
                    'PVO' :CubeListExtract(vort_diag_list, 'curl_pvo_zint'),
                    'HPG' :CubeListExtract(vort_diag_list, 'curl_hpg_zint'),
                    'LDF' :CubeListExtract(vort_diag_list, 'curl_ldf_zint'),
                    'ZAD' :CubeListExtract(vort_diag_list, 'curl_zad_zint'),
                    'TOT' :CubeListExtract(vort_diag_list, 'curl_tot_zint'),
                    'WND' :CubeListExtract(vort_diag_list, 'curl_wnd_zint'),
                    'FRC' :CubeListExtract(vort_diag_list, 'curl_frc_zint'),
                    'FDU' :CubeListExtract(vort_diag_list, 'curl_fdu_zint'),
                    'MLV' :CubeListExtract(vort_diag_list, 'curl_mlv_zint'),
                    'BET' :CubeListExtract(vort_diag_list, 'curl_bet_zint'),
                    'PRC' :CubeListExtract(vort_diag_list, 'curl_prc_zint')}
            
            if model == 'global': Ni_input_dict['ICE'] = CubeListExtract(vort_diag_list, 'curl_ice_zint')
            
        else:
            
            Ni_input_dict = { 'KEG' :vort_diag_dict['KEG'],
                    'RVO' :vort_diag_dict['RVO'],
                    'PVO' :vort_diag_dict['PVO'],
                    'HPG' :vort_diag_dict['HPG'],
                    'LDF' :vort_diag_dict['LDF'],
                    'ZAD' :vort_diag_dict['ZAD'],
                    'TOT' :vort_diag_dict['TOT'],
                    'WND' :vort_diag_dict['WND'],
                    'FRC' :vort_diag_dict['FRC'],
                    'FDU' :vort_diag_dict['FDU'],
                    'MLV' :vort_diag_dict['MLV'],
                    'BET' :vort_diag_dict['BET'],
                    'PRC' :vort_diag_dict['PRC']}
            
            if model == 'global': Ni_input_dict['ICE'] = vort_diag_dict['ICE']


        keywords = {'lonlatbounds':lonlatbounds, 'interpolation':interp, 'res':interp_res, 'nlevels':nlevels, 'level_min':level_min, 'level_max':level_max }
        area_weights = e1f*e2f
        
        Ni_cubes_out_dict, contour_masks = ContourInt.niiler_integral2D(Ni_input_dict, sf_zint_cube, area_weights, **keywords)

        
        f = open(f"{out_dir}/NI_contour_masks.{out_label}.pkl",'wb')
        pickle.dump(contour_masks, f)
        f.close()

        for Ni_label in Ni_cubes_out_dict:
            iris.save(Ni_cubes_out_dict[Ni_label], out_dir + f'/NI_2D_vort_{Ni_label}.{out_label}.nc' )
            
        #Also calculate integrals that are linear combinations of others
        print("Calculating ADV")
        NI_ADV_cube = ContourInt.NI_ADV_calc(Ni_cubes_out_dict['KEG'], Ni_cubes_out_dict['RVO'], Ni_cubes_out_dict['ZAD'])
        
        print("Calculating ZDF")
        if model == 'global':
            NI_ZDF_cube = ContourInt.NI_ZDF_calc(Ni_cubes_out_dict['WND'], Ni_cubes_out_dict['FRC'], icelog=True, Ni_ice_cube=Ni_cubes_out_dict['ICE'] )
        else:
            NI_ZDF_cube = ContourInt.NI_ZDF_calc(Ni_cubes_out_dict['WND'], Ni_cubes_out_dict['FRC'], icelog=False)
        
        print("Calculating RES")
        NI_RES_cube = ContourInt.NI_RES_calc(NI_ADV_cube             , Ni_cubes_out_dict['PVO'], Ni_cubes_out_dict['HPG'],
                                             Ni_cubes_out_dict['LDF'], NI_ZDF_cube             , Ni_cubes_out_dict['TOT'])


        
        iris.save(NI_ADV_cube, out_dir + f'/NI_2D_vort_ADV.{out_label}.nc')
        iris.save(NI_ZDF_cube, out_dir + f'/NI_2D_vort_ZDF.{out_label}.nc')
        iris.save(NI_RES_cube, out_dir + f'/NI_2D_vort_RES.{out_label}.nc')
        

    return


def save_namelist(data_dir, nn_res, out_label, sf_zint_log, vort_diag_log, cont_int_log, model, domcfg_path, fscheme, imin, imax, jmin, jmax, nlevels, lonlatbounds, interp, interp_res, level_min, level_max, R):
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
    f.write(f"sf_zint_log = {sf_zint_log}\n")
    f.write(f"vort_diag_log = {vort_diag_log}\n")
    f.write(f"cont_int_log = {cont_int_log}\n")
    f.write(f"model = {model}\n")
    f.write(f"domcfg_path = {domcfg_path}\n")
    f.write(f"fscheme = {fscheme}\n")
    f.write(f"imin = {imin}\n")
    f.write(f"imax = {imax}\n")
    f.write(f"jmin = {jmin}\n")
    f.write(f"jmax = {jmax}\n")
    f.write(f"nlevels = {nlevels}\n")
    f.write(f"lonlatbounds = {lonlatbounds}\n")
    f.write(f"interp = {interp}\n")
    f.write(f"interp_res = {interp_res}\n")
    f.write(f"1/interp_res = {1/interp_res}\n")
    f.write(f"level_min = {level_min}\n")
    f.write(f"level_max = {level_max}\n")
    f.write(f"R = {R}\n")
    f.write(f" \n")
    f.write(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    f.close()




    return

