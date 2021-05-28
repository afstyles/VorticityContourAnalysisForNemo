#  _   _            _  ______ _                             _   _          
# | | | |          | | |  _  (_)                           | | (_)         
# | | | | ___  _ __| |_| | | |_  __ _  __ _ _ __   ___  ___| |_ _  ___ ___ 
# | | | |/ _ \| '__| __| | | | |/ _` |/ _` | '_ \ / _ \/ __| __| |/ __/ __|
# \ \_/ / (_) | |  | |_| |/ /| | (_| | (_| | | | | (_) \__ \ |_| | (__\__ \
#  \___/ \___/|_|   \__|___/ |_|\__,_|\__, |_| |_|\___/|___/\__|_|\___|___/
#                                      __/ |                               
#                                     |___/                                
"""
VortDiagnostics.py

Calculation of vorticity diagnostics in NEMO and components of the PVO momentum diagnostic.

Contains methods:
    VortDiagnostic2D --> Calculates the barotropic vorticity diagnostics from a complete set of momentum diagnostics
    PVO_divcalc --> Calculates the component of the PVO vorticity diagnostic == f divh(U)
    PVO_fullcalc --> Offline calculation of the PVO momentum diagnostics
    PVO_nulcalc --> Calculates PVO momentum diagnostic assuming no variation of the Coriolis parameter or cell thicknesses
    PVO_betcalc --> Calculates PVO momentum diagnostic assuming no variation of cell thicknesses
    PVO_prccalc --> Calculates PVO momentum diagnostic assuming no variation in the Coriolis parameter
"""

import iris
from iris.cube import Cube
from iris.coords import AuxCoord
import numpy as np
import CGridOperations as CGO
from CGridOperations import im1, ip1, jm1, jp1, roll_and_mask

#  _   _            _  ______ _                             _   _      _____ ______ 
# | | | |          | | |  _  (_)                           | | (_)    / __  \|  _  \
# | | | | ___  _ __| |_| | | |_  __ _  __ _ _ __   ___  ___| |_ _  ___`' / /'| | | |
# | | | |/ _ \| '__| __| | | | |/ _` |/ _` | '_ \ / _ \/ __| __| |/ __| / /  | | | |
# \ \_/ / (_) | |  | |_| |/ /| | (_| | (_| | | | | (_) \__ \ |_| | (__./ /___| |/ / 
#  \___/ \___/|_|   \__|___/ |_|\__,_|\__, |_| |_|\___/|___/\__|_|\___\_____/|___/  
#                                      __/ |                                        
#                                     |___/                                    
     
def VortDiagnostic2D(u_cube, v_cube, 
                    ukeg_cube, urvo_cube, upvo_cube, uhpg_cube, uldf_cube, uzdf_cube, uzad_cube, utot_cube, ubet_cube, uprc_cube, upvo2_cube, unul_cube,
                    vkeg_cube, vrvo_cube, vpvo_cube, vhpg_cube, vldf_cube, vzdf_cube, vzad_cube, vtot_cube, vbet_cube, vprc_cube, vpvo2_cube, vnul_cube,
                    ff_f, e3u, e3v, e3t, e1u , e2u, e1v, e2v, e1f, e2f, tmask):
    """
    VortDiagnostic2D(u_cube, v_cube, 
                    ukeg_cube, urvo_cube, upvo_cube, uhpg_cube, uldf_cube, uzdf_cube, uzad_cube, utot_cube, ubet_cube, unvs_cube, upvo2_cube, unul_cube,
                    vkeg_cube, vrvo_cube, vpvo_cube, vhpg_cube, vldf_cube, vzdf_cube, vzad_cube, vtot_cube, vbet_cube, vnvs_cube, vpvo2_cube, vnul_cube,
                    ff_f, e3u, e3v, e3t, e1u , e2u, e1v, e2v, e1f, e2f, tmask):

    Depth integrates the NEMO momentum trends and then calculates the k component of the curl.
    This gives the corresponding depth-integrated vorticity trends.

    The curl is calculated so it is centred on the f point at {i,j} using the following calculation:

    curl(Tx,Ty) = ( Ty[i+1,j] - Ty[i,j] )/dx  - ( Tx[i,j+1] - Tx[i,j] )/dy

    INPUT VARIABLES:

    u_cube    - IRIS cube of x-velocities                                       (t,z,y,x) [m/s]
    v_cube    - IRIS cube of y-velocities                                       (t,z,y,x) [m/s]

    ukeg_cube  - IRIS cube containing Kinetic Energy Gradient x component       (t,z,y,x) [m/s2]
    urvo_cube  - IRIS cube containing Relative Vorticity x component            (t,z,y,x) [m/s2]
    upvo_cube  - IRIS cube containing Planetary Vorticty x component            (t,z,y,x) [m/s2]
    uhpg_cube  - IRIS cube containing Horizontal Pressure Gradient x component  (t,z,y,x) [m/s2]
    uldf_cube  - IRIS cube containing Lateral Diffusion x component             (t,z,y,x) [m/s2]
    uzdf_cube  - IRIS cube containing Vertical Diffusion x component            (t,z,y,x) [m/s2]
    uzad_cube  - IRIS cube containing Vertical Advection x component            (t,z,y,x) [m/s2]
    utot_cube  - IRIS cube containing Total before time stepping x component    (t,z,y,x) [m/s2]
    ubet_cube  - IRIS cube containing PVO assuming no changes in cell thickness x component              (t,z,y,x) [m/s2]
    uprc_cube  - IRIS cube containing PVO assuming no changes in the Coriolis parameter x component      (t,z,y,x) [m/s2]
    upvo2_cube - IRIS cube containing an offline calculation of PVO x component                          (t,z,y,x) [m/s2]
    unul_cube  - IRIS cube containing PVO assuming no changes in the Coriolis parameter or cell thickness x component   (t,z,y,x) [m/s2]

    vkeg_cube - IRIS cube containing Kinetic Energy Gradient y component        (t,z,y,x) [m/s2]
    vrvo_cube - IRIS cube containing Relative Vorticity y component             (t,z,y,x) [m/s2]
    vpvo_cube - IRIS cube containing Planetary Vorticty y component             (t,z,y,x) [m/s2]
    vhpg_cube - IRIS cube containing Horizontal Pressure Gradient y component   (t,z,y,x) [m/s2]
    vldf_cube - IRIS cube containing Lateral Diffusion y component              (t,z,y,x) [m/s2]
    vzdf_cube - IRIS cube containing Vertical Diffusion y component             (t,z,y,x) [m/s2]
    vzad_cube - IRIS cube containing Vertical Advection y component             (t,z,y,x) [m/s2]
    vtot_cube - IRIS cube containing Total before time stepping y component     (t,z,y,x) [m/s2]
    vbet_cube  - IRIS cube containing PVO assuming no changes in cell thickness y component              (t,z,y,x) [m/s2]
    vprc_cube  - IRIS cube containing PVO assuming no changes in the Coriolis parameter y component      (t,z,y,x) [m/s2]
    vpvo2_cube - IRIS cube containing an offline calculation of PVO y component                          (t,z,y,x) [m/s2]
    vnul_cube  - IRIS cube containing PVO assuming no changes in the Coriolis parameter or cell thickness y component   (t,z,y,x) [m/s2]

    ff_f - Array of Coriolis parameter values centred on f points (x,y) [1/s]

    e3u - Array of u cell thicknesses in the z direction (z,y,x) [m]    
    e3v - Array of v cell thicknesses in the z direction (z,y,x) [m]

    e1u - Array of u cell widths in the x direction (y,x) [m]
    e2u - Array of u cell widths in the y direction (y,x) [m]
    e1v - Array of v cell widths in the x direction (y,x) [m]
    e2v - Array of v cell widths in the y direction (y,x) [m]
    e1f - Array of f cell widths in the x direction (y,x) [m]
    e2f - Array of f cell widths in the y direction (y,x) [m]

    tmask - Mask for the t points (True = Masked)   (z,y,x) [-]


    OUTPUT variables
    Returns a dictionary of IRIS cubes that are the associated vorticity diagnostics

    Dictionary key | Description
    ---------------|-------------------------------------------------------------------------------
    KEG            - Curl of Kinetic Energy gradient trend                           (t,y,x) [m/s2]
    RVO            - Curl of Relative Vorticity trend                                (t,y,x) [m/s2]
    PVO            - Curl of Planetary Vorticity trend                               (t,y,x) [m/s2]
    HPG            - Curl of Horizontal Pressure gradient trend                      (t,y,x) [m/s2]
    LDF            - Curl of Lateral Diffusion trend                                 (t,y,x) [m/s2]
    ZDF            - Curl of Vertical Diffusion trend                                (t,y,x) [m/s2]
    ZAD            - Curl of Vertical Advection trend                                (t,y,x) [m/s2]
    TOT            - Curl of Total before time stepping trend                        (t,y,x) [m/s2]
    ADV            - Curl of total Advection trend                                   (t,y,x) [m/s2]
    RES            - Curl of Residual trend                                          (t,y,x) [m/s2]
    PVO2           - Curl of the offline calculation of PVO                          (t,y,x) [m/s2]
    FDU            - Component of curl(PVO) = f * divh(U)                            (t,y,x) [m/s2]     
    MLV            - Component of curl(PVO) due to changes in model level            (t,y,x) [m/s2]
    BET            - Component of curl(PVO) due to changes in the Coriolis parameter (t,y,x) [m/s2]
    PRC            - Component of curl(PVO) due to changes in partial cell thickness (t,y,x) [m/s2]

    """

    #Load Coriolis parameter centred on u and v points
    ff_u = uprc_cube.coord("ff_u").points
    ff_v = vprc_cube.coord("ff_v").points
    
    #Depth integrate the momentum trends and velocities
    u_zint     = np.sum(u_cube.data     * e3u, axis = -3)
    ukeg_zint  = np.sum(ukeg_cube.data  * e3u, axis = -3)
    urvo_zint  = np.sum(urvo_cube.data  * e3u, axis = -3)
    upvo_zint  = np.sum(upvo_cube.data  * e3u, axis = -3)
    uhpg_zint  = np.sum(uhpg_cube.data  * e3u, axis = -3)
    uldf_zint  = np.sum(uldf_cube.data  * e3u, axis = -3)
    uzdf_zint  = np.sum(uzdf_cube.data  * e3u, axis = -3)
    uzad_zint  = np.sum(uzad_cube.data  * e3u, axis = -3)
    utot_zint  = np.sum(utot_cube.data  * e3u, axis = -3)
    ubet_zint  = np.sum(ubet_cube.data  * e3u, axis = -3)
    unvs_zint  = np.sum(uprc_cube.data  * e3u, axis = -3)
    upvo2_zint = np.sum(upvo2_cube.data * e3u, axis = -3)
    unul_zint  = np.sum(unul_cube.data  * e3u, axis = -3)

    v_zint     = np.sum(v_cube.data     * e3v, axis = -3)
    vkeg_zint  = np.sum(vkeg_cube.data  * e3v, axis = -3)
    vrvo_zint  = np.sum(vrvo_cube.data  * e3v, axis = -3)
    vpvo_zint  = np.sum(vpvo_cube.data  * e3v, axis = -3)
    vhpg_zint  = np.sum(vhpg_cube.data  * e3v, axis = -3)
    vldf_zint  = np.sum(vldf_cube.data  * e3v, axis = -3)
    vzdf_zint  = np.sum(vzdf_cube.data  * e3v, axis = -3)
    vzad_zint  = np.sum(vzad_cube.data  * e3v, axis = -3)
    vtot_zint  = np.sum(vtot_cube.data  * e3v, axis = -3)
    vbet_zint  = np.sum(vbet_cube.data  * e3v, axis = -3)
    vnvs_zint  = np.sum(vprc_cube.data  * e3v, axis = -3)
    vpvo2_zint = np.sum(vpvo2_cube.data * e3v, axis = -3)
    vnul_zint  = np.sum(vnul_cube.data  * e3v, axis = -3)
    
    #Approximately separate zdf into wind and friction by integrating over partial depth >>>
    depthu = np.cumsum(e3u, axis=0) - np.broadcast_to(0.5*e3u[0,...], e3u.shape)
    depthv = np.cumsum(e3v, axis=0) - np.broadcast_to(0.5*e3v[0,...], e3v.shape)
    
    depthu = np.broadcast_to(depthu, uzdf_cube.shape)
    depthv = np.broadcast_to(depthv, vzdf_cube.shape)

    uwnd_mask = (depthu > 100)
    vwnd_mask = (depthv > 100)
    
    uwnd = uzdf_cube.data
    mask = np.ma.mask_or(uwnd.mask, uwnd_mask)
    uwnd = np.ma.masked_array(uwnd, mask=mask)
    
    vwnd = vzdf_cube.data
    mask = np.ma.mask_or(vwnd.mask, vwnd_mask)
    vwnd = np.ma.masked_array(vwnd, mask=mask)
    
    uwnd_zint = np.sum(uwnd*e3u, axis = -3)
    vwnd_zint = np.sum(vwnd*e3v, axis = -3)
    
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    curl_keg =      CGO.kcurl_orca(ukeg_zint, vkeg_zint, e1u, e2v, e1f, e2f)
    curl_rvo =      CGO.kcurl_orca(urvo_zint, vrvo_zint, e1u, e2v, e1f, e2f)
    curl_pvo =      CGO.kcurl_orca(upvo_zint, vpvo_zint, e1u, e2v, e1f, e2f)
    curl_hpg =      CGO.kcurl_orca(uhpg_zint, vhpg_zint, e1u, e2v, e1f, e2f)
    curl_ldf =      CGO.kcurl_orca(uldf_zint, vldf_zint, e1u, e2v, e1f, e2f)
    curl_zdf =      CGO.kcurl_orca(uzdf_zint, vzdf_zint, e1u, e2v, e1f, e2f)
    curl_zad =      CGO.kcurl_orca(uzad_zint, vzad_zint, e1u, e2v, e1f, e2f)
    curl_tot =      CGO.kcurl_orca(utot_zint, vtot_zint, e1u, e2v, e1f, e2f)
    curl_wnd =      CGO.kcurl_orca(uwnd_zint, vwnd_zint, e1u, e2v, e1f, e2f)
    curl_bet =      CGO.kcurl_orca(ubet_zint, vbet_zint, e1u, e2v, e1f, e2f )
    curl_pvo2 =     CGO.kcurl_orca(upvo2_zint, vpvo2_zint, e1u, e2v, e1f, e2f )

    #Divide the momentum diagnostics by the u/v point value of f so no beta effects emerge from the curl calculation
    #Then multiply by the value of f at the f point afterwards
    curl_nvs = ff_f*CGO.kcurl_orca(unvs_zint/ff_u, vnvs_zint/ff_v, e1u, e2v, e1f, e2f )
    curl_nul = ff_f*CGO.kcurl_orca(unul_zint/ff_u, vnul_zint/ff_v, e1u, e2v, e1f, e2f )
    
    #Calculate the friction contribution from remainder of ZDF
    curl_frc = curl_zdf - curl_wnd
    
    #Calculate total advective contribution
    curl_adv = curl_keg + curl_rvo + curl_zad

    #Calculate residual of vorticity budget:
    # Budget :  TOT = KEG + RVO + PVO + HPG + LDF + ZDF + ZAD
    # Residual: RES = KEG + RVO + PVO + HPG + LDF + ZDF + ZAD - TOT

    curl_res = ( curl_keg + curl_rvo + curl_pvo + curl_hpg 
                + curl_ldf + curl_zdf + curl_zad - curl_tot )
        
    #Decompose PVO into 4 meaningful parts
    curl_fdu = PVO_divcalc(u_cube, v_cube, ff_f, e1u, e1v, e2u, e2v, e1f, e2f, e3u, e3v, curl_pvo.mask )  #f divh(U)
    curl_mlv = curl_nul - curl_fdu  #Changes in model level
    curl_bet = curl_bet - curl_nul  #Beta effect
    curl_prc = curl_nvs - curl_nul  #Partial cells

    #Save outputs as IRIS cubes >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    time_coord  = ukeg_cube.coord('time')
    
    lat = ukeg_cube.coord('latitude')
    lon = ukeg_cube.coord('longitude')

    curl_keg_cube = Cube(curl_keg, dim_coords_and_dims=[(time_coord,0)])
    curl_keg_cube.long_name     = 'k-curl of depth-integrated kinetic energy gradient trend'
    curl_keg_cube.var_name      = 'curl_keg_zint'
    curl_keg_cube.units         = 'm/s2'
    curl_keg_cube.add_aux_coord(lat, [-2,-1])
    curl_keg_cube.add_aux_coord(lon, [-2,-1])

    curl_rvo_cube = Cube(curl_rvo, dim_coords_and_dims=[(time_coord,0)])
    curl_rvo_cube.long_name     = 'k-curl of depth-integrated relative vorticity trend'
    curl_rvo_cube.var_name      = 'curl_rvo_zint'
    curl_rvo_cube.units         = 'm/s2'
    curl_rvo_cube.add_aux_coord(lat, [-2,-1])
    curl_rvo_cube.add_aux_coord(lon, [-2,-1])

    curl_pvo_cube = Cube(curl_pvo, dim_coords_and_dims=[(time_coord,0)])
    curl_pvo_cube.long_name     = 'k-curl of depth-integrated planetary vorticity trend'
    curl_pvo_cube.var_name      = 'curl_pvo_zint'
    curl_pvo_cube.units         = 'm/s2'
    curl_pvo_cube.add_aux_coord(lat, [-2,-1])
    curl_pvo_cube.add_aux_coord(lon, [-2,-1])
    
    curl_hpg_cube = Cube(curl_hpg, dim_coords_and_dims=[(time_coord,0)])
    curl_hpg_cube.long_name     = 'k-curl of depth-integrated horizontal pressure gradient trend'
    curl_hpg_cube.var_name      = 'curl_hpg_zint'
    curl_hpg_cube.units         = 'm/s2'
    curl_hpg_cube.add_aux_coord(lat, [-2,-1])
    curl_hpg_cube.add_aux_coord(lon, [-2,-1])

    curl_ldf_cube = Cube(curl_ldf, dim_coords_and_dims=[(time_coord,0)])
    curl_ldf_cube.long_name     = 'k-curl of depth-integrated lateral diffusion trend'
    curl_ldf_cube.var_name      = 'curl_ldf_zint'
    curl_ldf_cube.units         = 'm/s2'
    curl_ldf_cube.add_aux_coord(lat, [-2,-1])
    curl_ldf_cube.add_aux_coord(lon, [-2,-1])

    curl_zdf_cube = Cube(curl_zdf, dim_coords_and_dims=[(time_coord,0)])
    curl_zdf_cube.long_name     = 'k-curl of depth-integrated vertical diffusion trend'
    curl_zdf_cube.var_name      = 'curl_zdf_zint'
    curl_zdf_cube.units         = 'm/s2'
    curl_zdf_cube.add_aux_coord(lat, [-2,-1])
    curl_zdf_cube.add_aux_coord(lon, [-2,-1])

    curl_zad_cube = Cube(curl_zad, dim_coords_and_dims=[(time_coord,0)])
    curl_zad_cube.long_name     = 'k-curl of depth-integrated vertical advection trend'
    curl_zad_cube.var_name      = 'curl_zad_zint'
    curl_zad_cube.units         = 'm/s2'
    curl_zad_cube.add_aux_coord(lat, [-2,-1])
    curl_zad_cube.add_aux_coord(lon, [-2,-1])

    curl_tot_cube = Cube(curl_tot, dim_coords_and_dims=[(time_coord,0)])
    curl_tot_cube.long_name     = 'k-curl of depth-integrated total before time stepping trend'
    curl_tot_cube.var_name      = 'curl_tot_zint'
    curl_tot_cube.units         = 'm/s2'
    curl_tot_cube.add_aux_coord(lat, [-2,-1])
    curl_tot_cube.add_aux_coord(lon, [-2,-1])
    
    curl_wnd_cube = Cube(curl_wnd, dim_coords_and_dims=[(time_coord,0)])
    curl_wnd_cube.long_name     = 'k-curl of depth-integrated wind stress (partial ZDF) trend'
    curl_wnd_cube.var_name      = 'curl_wnd_zint'
    curl_wnd_cube.units         = 'm/s2'
    curl_wnd_cube.add_aux_coord(lat, [-2,-1])
    curl_wnd_cube.add_aux_coord(lon, [-2,-1])
    
    curl_frc_cube = Cube(curl_frc, dim_coords_and_dims=[(time_coord,0)])
    curl_frc_cube.long_name     = 'k-curl of depth-integrated lateral friction (partial ZDF) trend'
    curl_frc_cube.var_name      = 'curl_frc_zint'
    curl_frc_cube.units         = 'm/s2'
    curl_frc_cube.add_aux_coord(lat, [-2,-1])
    curl_frc_cube.add_aux_coord(lon, [-2,-1])

    curl_adv_cube = Cube(curl_adv, dim_coords_and_dims=[(time_coord,0)])
    curl_adv_cube.long_name     = 'k-curl of depth-integrated total advection trend'
    curl_adv_cube.var_name      = 'curl_adv_zint'
    curl_adv_cube.units         = 'm/s2'
    curl_adv_cube.add_aux_coord(lat, [-2,-1])
    curl_adv_cube.add_aux_coord(lon, [-2,-1])

    curl_res_cube = Cube(curl_res, dim_coords_and_dims=[(time_coord,0)])
    curl_res_cube.long_name     = 'k-curl of depth-integrated residual trend'
    curl_res_cube.var_name      = 'curl_res_zint'
    curl_res_cube.units         = 'm/s2'
    curl_res_cube.add_aux_coord(lat, [-2,-1])
    curl_res_cube.add_aux_coord(lon, [-2,-1])
    
    curl_pvo2_cube = Cube(curl_pvo2, dim_coords_and_dims=[(time_coord,0)])
    curl_pvo2_cube.long_name     = 'Recreation of PVO'
    curl_pvo2_cube.var_name      = 'curl_pvo2_zint'
    curl_pvo2_cube.units         = 'm/s2'
    curl_pvo2_cube.add_aux_coord(lat, [-2,-1])
    curl_pvo2_cube.add_aux_coord(lon, [-2,-1])
    curl_pvo2_cube.attributes = upvo2_cube.attributes
    
    curl_fdu_cube = Cube(curl_fdu, dim_coords_and_dims=[(time_coord,0)])
    curl_fdu_cube.long_name     = 'Calculation of f divh(U)'
    curl_fdu_cube.var_name      = 'curl_fdu_zint'
    curl_fdu_cube.units         = 'm/s2'
    curl_fdu_cube.add_aux_coord(lat, [-2,-1])
    curl_fdu_cube.add_aux_coord(lon, [-2,-1])
    
    curl_mlv_cube = Cube(curl_mlv, dim_coords_and_dims=[(time_coord,0)])
    curl_mlv_cube.long_name     = 'PVO contribution due to changes in lowest model level'
    curl_mlv_cube.var_name      = 'curl_mlv_zint'
    curl_mlv_cube.units         = 'm/s2'
    curl_mlv_cube.add_aux_coord(lat, [-2,-1])
    curl_mlv_cube.add_aux_coord(lon, [-2,-1])
    curl_mlv_cube.attributes = unul_cube.attributes
    
    curl_bet_cube = Cube(curl_bet, dim_coords_and_dims=[(time_coord,0)])
    curl_bet_cube.long_name     = 'PVO contribution due to variations in Coriolis parameter'
    curl_bet_cube.var_name      = 'curl_bet_zint'
    curl_bet_cube.units         = 'm/s2'
    curl_bet_cube.add_aux_coord(lat, [-2,-1])
    curl_bet_cube.add_aux_coord(lon, [-2,-1])
    curl_bet_cube.attributes = ubet_cube.attributes
    
    curl_prc_cube = Cube(curl_prc, dim_coords_and_dims=[(time_coord,0)])
    curl_prc_cube.long_name     = 'PVO contribution due to varying cell thickness'
    curl_prc_cube.var_name      = 'curl_prc_zint'
    curl_prc_cube.units         = 'm/s2'
    curl_prc_cube.add_aux_coord(lat, [-2,-1])
    curl_prc_cube.add_aux_coord(lon, [-2,-1])
    curl_prc_cube.attributes = uprc_cube.attributes



    #Save outputs in a dictionary for easy extraction >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    output_dict = { 'KEG' :curl_keg_cube,
                    'RVO' :curl_rvo_cube,
                    'PVO' :curl_pvo_cube,
                    'HPG' :curl_hpg_cube,
                    'LDF' :curl_ldf_cube,
                    'ZDF' :curl_zdf_cube,
                    'ZAD' :curl_zad_cube,
                    'TOT' :curl_tot_cube,
                    'WND' :curl_wnd_cube,
                    'FRC' :curl_frc_cube,
                    'ADV' :curl_adv_cube,
                    'RES' :curl_res_cube,
                    'PVO2':curl_pvo2_cube,
                    'FDU' :curl_fdu_cube,
                    'MLV' :curl_mlv_cube,
                    'BET' :curl_bet_cube,
                    'PRC' :curl_prc_cube }


    return output_dict

# ______ _   _ _____     _ _                _      
# | ___ \ | | |  _  |   | (_)              | |     
# | |_/ / | | | | | | __| |___   _____ __ _| | ___ 
# |  __/| | | | | | |/ _` | \ \ / / __/ _` | |/ __|
# | |   \ \_/ | \_/ / (_| | |\ V / (_| (_| | | (__ 
# \_|    \___/ \___/ \__,_|_| \_/ \___\__,_|_|\___|
#                ______                            
#               |______|                           

def PVO_divcalc(u_cube, v_cube, ff_f, e1u, e1v, e2u, e2v, e1f, e2f, e3u, e3v, fmask ):
    """
    PVO_divcalc(u_cube, v_cube, ff_f, e1u, e1v, e2u, e2v, e1f, e2f, e3u, e3v, fmask )

    Calculates the component of PVO due to the divergence of the depth integrated velocity field

    Returns f * divh(U)
    where U is the depth integrated velocity field and divh(U) is the horizontal divergence centred on the 
    fpoints of the grid

    u_cube - IRIS cube of x-velocities    (t,z,y,x)
    v_cube - IRIS cube of y-velocities    (t,z,y,x)

    ff_f   - Array of coriolis parameter values centred on f points (y,x) 

    e1u    - Array of u cell widths in the x direction        (y,x) [m]
    e1v    - Array of v cell widths in the x direction        (y,x) [m]
    e2u    - Array of u cell widths in the y direction        (y,x) [m]
    e2v    - Array of v cell widths in the y direction        (y,x) [m]
    e1f    - Array of f cell widths in the x direction        (y,x) [m]
    e2f    - Array of f cell widths in the y direction        (y,x) [m]
    e3u    - Array of u cell thicknesses in the z direction (z,y,x) [m]
    e3v    - Array of v cell thicknesses in the z direction (z,y,x) [m]

    tmask  - Mask for the t points (True=Masked)            (z,y,x) [-]

    Returns

    PVO_div - Array of f*divh(U) centred on f points         (t,y,x) [m/s2]



    """    
    uflux = u_cube.data * e2u * e3u
    vflux = v_cube.data * e1v * e3v
    
    uflux[uflux.mask] = 0
    vflux[vflux.mask] = 0
    
    uflux_zint = np.sum(uflux, axis=-3)
    vflux_zint = np.sum(vflux, axis=-3)
    
    divhU = ( ip1(jp1(uflux_zint)) + ip1(uflux_zint) - im1(uflux_zint) - im1(jp1(uflux_zint)) 
             +jp1(vflux_zint) + ip1(jp1(vflux_zint)) - jm1(vflux_zint) - jm1(ip1(vflux_zint)))/(4*e1f*e2f)

    divhU[...,: ,0 ] = 0
    divhU[...,: ,-1] = 0
    divhU[...,0 ,: ] = 0
    divhU[...,-1,: ] = 0
    
    PVO_div = - ff_f * divhU
    PVO_div = np.ma.masked_array(PVO_div, mask=fmask)
    
    return PVO_div

# ______ _   _ _____   __       _ _           _      
# | ___ \ | | |  _  | / _|     | | |         | |     
# | |_/ / | | | | | || |_ _   _| | | ___ __ _| | ___ 
# |  __/| | | | | | ||  _| | | | | |/ __/ _` | |/ __|
# | |   \ \_/ | \_/ /| | | |_| | | | (_| (_| | | (__ 
# \_|    \___/ \___/ |_|  \__,_|_|_|\___\__,_|_|\___|
#                ______                              
#               |______|                             

def PVO_fullcalc(u_cube, v_cube, ff_f, e1u, e1v, e2u, e2v, e1f, e2f, e3u, e3v, e3t, tmask, fscheme='een_0', model='global'):
    """
    PVO_fullcalc(u_cube, v_cube, ff_f, e1u, e1v, e2u, e2v, e1f, e2f, e3u, e3v, e3t, tmask, fscheme='een_0', model='global')

    Calculates the x and y component of the PVO momentum diagnostic outside of the model. This demonstrates the exact method used in NEMO to calculate the diagnostic.

    u_cube - IRIS cube of x-velocities    (t,z,y,x)
    v_cube - IRIS cube of y-velocities    (t,z,y,x)

    ff_f   - Array of coriolis parameter values centred on f points (y,x) 

    e1u    - Array of u cell widths in the x direction        (y,x) [m]
    e1v    - Array of v cell widths in the x direction        (y,x) [m]
    e2u    - Array of u cell widths in the y direction        (y,x) [m]
    e2v    - Array of v cell widths in the y direction        (y,x) [m]
    e1f    - Array of f cell widths in the x direction        (y,x) [m]
    e2f    - Array of f cell widths in the y direction        (y,x) [m]
    e3u    - Array of u cell thicknesses in the z direction (z,y,x) [m]
    e3v    - Array of v cell thicknesses in the z direction (z,y,x) [m]
    e3t    - Array of t cell thicknesses in the z direction (z,y,x) [m]

    tmask  - Mask for the t points (True=Masked)            (z,y,x) [-]

    fscheme- String to describe the coriolis scheme method
                "een_0" - describes the EEN method which determines f cell thickness by
                          e3f = (e3t[i,j] + e3t[i+1,j] + e3t[i,j+1] + e3t[i+1,j+1])/4
                          Where masked values of e3t = 0

                "een_1" - describes the EEN method which determines f cell thickness by
                          e3f = (e3t[i,j] + e3t[i+1,j] + e3t[i,j+1] + e3t[i+1,j+1])/(num_unmasked)
                          Where masked values of e3t = 0 and num_unmasked is the number of unmasked values 
                          in the average


    model  - String to designate which model the data originates from. This is necessary as GYRE and the global model have different masking conventions.
                "global" - Results are from the NEMO global model
                "gyre"   - Results are from GYRE_PISCES
    
    Returns - u_pvo_cube, v_pvo_cube

    u_pvo_cube - IRIS cube, x component of the recreated PVO momentum diagnostic  (t,z,y,x) [m/s2]
    v_pvo_cube - IRIS cube, y component of the recreated PVO momentum diagnostic  (t,z,y,x) [m/s2]

    """
    #First determine the f cell thicknesses
    e3t_copy = np.ma.copy(e3t)
    e3t_copy[tmask] = 0.0
    
    if fscheme == 'een_0':
        e3f = (e3t_copy + ip1(e3t_copy) + jp1(e3t_copy) + ip1(jp1(e3t_copy)))/4

    else:
        #Calculate number of masked t points surrounding the f point
        num_masked_t = np.sum([tmask,ip1(tmask),jp1(tmask), ip1(jp1(tmask))], axis=0)

        e3f  = (e3t_copy + ip1(e3t_copy) + jp1(e3t_copy) + ip1(jp1(e3t_copy)))/(4-num_masked_t)

        #F points that are completely surrounded by masked t points are zeroed
        e3f[num_masked_t == 4] = 0.0

    
    #Edge values of e3f are zeroed as done in NEMO
    e3f[...,-1,:] = 0.0
    e3f[...,:,-1] = 0.0

    #Calculate f/e3f
    f_e3f = ff_f/e3f

    #Set areas divided by zero to zero
    f_e3f[e3f==0.0] = 0.0

    #Calculate f triads (neglecting variations in f)
    f3_ne = (    f_e3f  + im1(f_e3f) +     jm1(f_e3f))
    f3_nw = (    f_e3f  + im1(f_e3f) + im1(jm1(f_e3f))) 
    f3_se = (    f_e3f  + jm1(f_e3f) + im1(jm1(f_e3f))) 
    f3_sw = (im1(f_e3f) + jm1(f_e3f) + im1(jm1(f_e3f))) 

    #Then set first row and column to zero as done in NEMO
    f3_ne[...,0,:] = 0
    f3_nw[...,0,:] = 0
    f3_se[...,0,:] = 0
    f3_sw[...,0,:] = 0

    f3_ne[...,:,0] = 0
    f3_nw[...,:,0] = 0
    f3_se[...,:,0] = 0
    f3_sw[...,:,0] = 0

    #Calculate x and y volume fluxes
    uflux = u_cube.data * e2u * e3u
    vflux = v_cube.data * e1v * e3v

    uflux[uflux.mask] = 0.0
    vflux[vflux.mask] = 0.0


    #Calculate x component of PVO momentum diagnostic
    uPVO = (1/12.0)*(1/e1u)*(   f3_ne      * vflux 
                                + ip1(f3_nw) * ip1(vflux)
                                + f3_se      * jm1(vflux)
                                + ip1(f3_sw) * ip1(jm1(vflux)) )

    #Set edge values to zero as done in NEMO
    uPVO[...,0 ,: ] = 0.0
    uPVO[...,-1,: ] = 0.0
    uPVO[...,: ,-1] = 0.0
    
    
    # If data is from the global model, adjust mask to maintain a consistent
    # masking convention with diagnostics from the model
    if model == 'global':
        umask = u_cube.data.mask
        uPVO_zeroing_mask = tmask + ip1(tmask)
        uPVO_zeroing_mask[...,:,0] = True
        uPVO_zeroing_mask[...,:,-1] = True
        uPVO_zeroing_mask = np.broadcast_to(uPVO_zeroing_mask, uPVO.shape)
        
        uPVO[uPVO_zeroing_mask] = 0.0
        

    uPVO = np.ma.masked_array(uPVO.data, mask=u_cube.data.mask)


    #Calculate y component of PVO momentum diagnostic
    vPVO = -(1/12.0)*(1/e2v)*(  jp1(f3_sw) * im1(jp1(uflux))
                             + jp1(f3_se) * jp1(uflux)
                                 + f3_nw  * im1(uflux)
                                 + f3_ne  * uflux )

    #Set edge values to zero as done in NEMO
    vPVO[...,0 ,: ] = 0
    vPVO[...,-1,: ] = 0
    vPVO[...,: ,0 ] = 0

    # If data is from the global model, adjust mask to maintain a consistent
    # masking convention with diagnostics from the model
    if model == 'global':
        vmask = v_cube.data.mask
        vPVO_zeroing_mask = tmask + jp1(tmask)
        vPVO_zeroing_mask[...,0 , :] = True
        vPVO_zeroing_mask[...,-1, :] = True
        vPVO_zeroing_mask = np.broadcast_to(vPVO_zeroing_mask, vPVO.shape)
        
        vPVO[vPVO_zeroing_mask] = 0.0
    
    vPVO = np.ma.masked_array(vPVO.data, mask=v_cube.data.mask)

    #Store uPVO and vPVO as IRIS cubes
    time_coord = u_cube.coord("time")
    lat = u_cube.coord("latitude")
    lon = u_cube.coord("longitude")
    e3f_coord = AuxCoord(e3f.data, long_name='e3f', units='m')

    u_pvo_cube = Cube(uPVO, dim_coords_and_dims=[(time_coord,0)])
    u_pvo_cube.long_name     = 'x component of topographic coriolis acceleration'
    u_pvo_cube.var_name      = 'u_pvo2'
    u_pvo_cube.units         = 'm/s2'
    u_pvo_cube.add_aux_coord(lat, [2,3])
    u_pvo_cube.add_aux_coord(lon, [2,3])
    u_pvo_cube.add_aux_coord(e3f_coord, [1,2,3])
    u_pvo_cube.attributes = {'fscheme':fscheme, 'model':model}


    v_pvo_cube = Cube(vPVO, dim_coords_and_dims=[(time_coord,0)])
    v_pvo_cube.long_name     = 'y component of topographic coriolis acceleration'
    v_pvo_cube.var_name      = 'v_pvo2'
    v_pvo_cube.units         = 'm/s2'
    v_pvo_cube.add_aux_coord(lat, [2,3])
    v_pvo_cube.add_aux_coord(lon, [2,3])
    v_pvo_cube.add_aux_coord(e3f_coord, [1,2,3])
    v_pvo_cube.attributes = {'fscheme':fscheme, 'model':model}

    return u_pvo_cube, v_pvo_cube

# ______ _   _ _____              _           _      
# | ___ \ | | |  _  |            | |         | |     
# | |_/ / | | | | | | _ __  _   _| | ___ __ _| | ___ 
# |  __/| | | | | | || '_ \| | | | |/ __/ _` | |/ __|
# | |   \ \_/ | \_/ /| | | | |_| | | (_| (_| | | (__ 
# \_|    \___/ \___/ |_| |_|\__,_|_|\___\__,_|_|\___|
#                ______                              
#               |______|                             

def PVO_nulcalc(u_cube, v_cube, ff_f, e1u, e1v, e2u, e2v, e1f, e2f, e3u, e3v, tmask, model='global'):
    """
    PVO_nullcalc(u_cube, v_cube, ff_f, e1u, e1v, e2u, e2v, e1f, e2f, e3u, e3v, tmask, model='global')

    Calculates the x and y component of the PVO momentum diagnostic while assuming no variation in the 
    coriolis parameter or variations in cell thicknesses.

    u_cube - IRIS cube of x-velocities    (t,z,y,x)
    v_cube - IRIS cube of y-velocities    (t,z,y,x)

    ff_f   - Array of coriolis parameter values centred on f points (y,x) 

    e1u    - Array of u cell widths in the x direction        (y,x) [m]
    e1v    - Array of v cell widths in the x direction        (y,x) [m]
    e2u    - Array of u cell widths in the y direction        (y,x) [m]
    e2v    - Array of v cell widths in the y direction        (y,x) [m]
    e1f    - Array of f cell widths in the x direction        (y,x) [m]
    e2f    - Array of f cell widths in the y direction        (y,x) [m]
    e3u    - Array of u cell thicknesses in the z direction (z,y,x) [m]
    e3v    - Array of v cell thicknesses in the z direction (z,y,x) [m]

    tmask  - Mask for the t points (True=Masked)            (z,y,x) [-]

    model  - String to designate which model the data originates from. This is necessary as GYRE and the global model have different masking conventions.
                "global" - Results are from the NEMO global model
                "gyre"   - Results are from GYRE_PISCES
    
    Returns - u_pvo_cube, v_pvo_cube

    u_nul_cube - IRIS cube, x component of the recreated PVO momentum diagnostic  (t,z,y,x) [m/s2]
    v_nul_cube - IRIS cube, y component of the recreated PVO momentum diagnostic  (t,z,y,x) [m/s2]

    """

    #Calculate x and y volume fluxes
    uflux = u_cube.data * e2u * e3u
    vflux = v_cube.data * e1v * e3v

    uflux[uflux.mask] = 0.0
    vflux[vflux.mask] = 0.0

    #Calculate the Coriolis parameter centred on the u point
    ff_u = (jm1(ff_f)*(e2u-0.5*jm1(e2f)) + 0.5*jm1(e2f)*ff_f )/e2u
    ff_u[...,0,:] = ff_f[...,0,:]
    
    #Calculate the x component of PVO
    uPVO = (1/12.0)*(3*ff_u/e1u)*(1/e3u)*(     vflux 
                                + ip1(vflux)
                                + jm1(vflux)
                                + ip1(jm1(vflux)) )

    #Set edge values to zero as done in NEMO
    uPVO[...,0 ,: ] = 0.0
    uPVO[...,-1,: ] = 0.0
    uPVO[...,: ,-1] = 0.0
    uPVO[...,: ,0 ] = 0.0
    
    # If data is from the global model, adjust mask to maintain a consistent
    # masking convention with diagnostics from the model
    if model == 'global':
        umask = u_cube.data.mask
        uPVO_zeroing_mask = tmask + ip1(tmask)
        uPVO_zeroing_mask[...,:,0] = True
        uPVO_zeroing_mask[...,:,-1] = True
        uPVO_zeroing_mask = np.broadcast_to(uPVO_zeroing_mask, uPVO.shape)
        
        uPVO[uPVO_zeroing_mask] = 0.0
    
    uPVO = np.ma.masked_array(uPVO.data, mask=u_cube.data.mask)

    #Calculate the Coriolis parameter centred on the v point
    ff_v = (im1(ff_f)*(e1v-0.5*im1(e1f)) + 0.5*im1(e1f)*ff_f )/e1v
    ff_v[...,:,0] = ff_f[...,:,0]
    
    #Calculate the y component of PVO
    vPVO = -(1/12.0)*(3*ff_v/e2v)*(1/e3v)*( im1(jp1(uflux))
                                             + jp1(uflux)
                                             + im1(uflux)
                                             + uflux )

    #Set edge values to zero as done in NEMO
    vPVO[...,0 ,: ] = 0
    vPVO[...,-1,: ] = 0
    vPVO[...,: ,0 ] = 0
    vPVO[...,: ,-1] = 0
    
    # If data is from the global model, adjust mask to maintain a consistent
    # masking convention with diagnostics from the model
    if model == 'global':
        vmask = v_cube.data.mask
        vPVO_zeroing_mask = tmask + jp1(tmask)
        vPVO_zeroing_mask[...,0 , :] = True
        vPVO_zeroing_mask[...,-1, :] = True
        vPVO_zeroing_mask = np.broadcast_to(vPVO_zeroing_mask, vPVO.shape)
        
        vPVO[vPVO_zeroing_mask] = 0.0

    vPVO = np.ma.masked_array(vPVO.data, mask=v_cube.data.mask)


    time_coord = u_cube.coord("time")
    lat = u_cube.coord("latitude")
    lon = u_cube.coord("longitude")
    ff_u_coord = AuxCoord(ff_u, long_name='ff_u', units='1/s')
    ff_v_coord = AuxCoord(ff_v, long_name='ff_v', units='1/s')

    #Store uPVO and vPVO as IRIS cubes
    u_nul_cube = Cube(uPVO, dim_coords_and_dims=[(time_coord,0)])
    u_nul_cube.long_name     = 'x component of PVO (assuming constant f and cell thickness)'
    u_nul_cube.var_name      = 'u_nul'
    u_nul_cube.units         = 'm/s2'
    u_nul_cube.add_aux_coord(lat, [2,3])
    u_nul_cube.add_aux_coord(lon, [2,3])
    u_nul_cube.add_aux_coord(ff_u_coord, [2,3])
    u_nul_cube.attributes = {'model':model}

    v_nul_cube = Cube(vPVO, dim_coords_and_dims=[(time_coord,0)])
    v_nul_cube.long_name     = 'y component of PVO (assuming constant f and cell thickness)'
    v_nul_cube.var_name      = 'v_nul'
    v_nul_cube.units         = 'm/s2'
    v_nul_cube.add_aux_coord(lat, [2,3])
    v_nul_cube.add_aux_coord(lon, [2,3])
    v_nul_cube.add_aux_coord(ff_v_coord, [2,3])
    v_nul_cube.attributes = {'model':model}

    return u_nul_cube, v_nul_cube

# ______ _   _ _____  _          _            _      
# | ___ \ | | |  _  || |        | |          | |     
# | |_/ / | | | | | || |__   ___| |_ ___ __ _| | ___ 
# |  __/| | | | | | || '_ \ / _ \ __/ __/ _` | |/ __|
# | |   \ \_/ | \_/ /| |_) |  __/ || (_| (_| | | (__ 
# \_|    \___/ \___/ |_.__/ \___|\__\___\__,_|_|\___|
#                ______                              
#               |______|                             

def PVO_betcalc(u_cube, v_cube, ff_f, e1u, e1v, e2u, e2v, e1f, e2f, e3u, e3v, tmask, model='global'):
    """
    PVO_betacalc(u_cube, v_cube, ff_f, e1u, e1v, e2u, e2v, e1f, e2f, e3u, e3v, tmask, model='global')

    Calculates the x and y component of the PVO momentum diagnostic while assuming no variation in cell thicknesses.

    u_cube - IRIS cube of x-velocities    (t,z,y,x)
    v_cube - IRIS cube of y-velocities    (t,z,y,x)

    ff_f   - Array of coriolis parameter values centred on f points (y,x) 

    e1u    - Array of u cell widths in the x direction        (y,x) [m]
    e1v    - Array of v cell widths in the x direction        (y,x) [m]
    e2u    - Array of u cell widths in the y direction        (y,x) [m]
    e2v    - Array of v cell widths in the y direction        (y,x) [m]
    e1f    - Array of f cell widths in the x direction        (y,x) [m]
    e2f    - Array of f cell widths in the y direction        (y,x) [m]
    e3u    - Array of u cell thicknesses in the z direction (z,y,x) [m]
    e3v    - Array of v cell thicknesses in the z direction (z,y,x) [m]

    tmask  - Mask for the t points (True=Masked)            (z,y,x) [-]

    model  - String to designate which model the data originates from. This is necessary as GYRE and the global model have different masking conventions.
                "global" - Results are from the NEMO global model
                "gyre"   - Results are from GYRE_PISCES
    
    Returns - u_bet_cube, v_bet_cube

    u_bet_cube - IRIS cube, x component of the recreated PVO momentum diagnostic  (t,z,y,x) [m/s2]
    v_bet_cube - IRIS cube, y component of the recreated PVO momentum diagnostic  (t,z,y,x) [m/s2]

    """

    from iris.cube import Cube
    from iris.coords import AuxCoord
    
    #Calculate f triads (neglecting variations in e3f)
    f3_ne = (    ff_f  + im1(ff_f) +     jm1(ff_f))
    f3_nw = (    ff_f  + im1(ff_f) + im1(jm1(ff_f))) 
    f3_se = (    ff_f  + jm1(ff_f) + im1(jm1(ff_f))) 
    f3_sw = (im1(ff_f) + jm1(ff_f) + im1(jm1(ff_f))) 

    #Then set first row and column to zero as done in NEMO
    f3_ne[...,0,:] = 0
    f3_nw[...,0,:] = 0
    f3_se[...,0,:] = 0
    f3_sw[...,0,:] = 0

    f3_ne[...,:,0] = 0
    f3_nw[...,:,0] = 0
    f3_se[...,:,0] = 0
    f3_sw[...,:,0] = 0

    #Calculate x and y volume fluxes
    uflux = u_cube.data * e2u * e3u
    vflux = v_cube.data * e1v * e3v

    uflux[uflux.mask] = 0.0
    vflux[vflux.mask] = 0.0

    #Calculate the x component of PVO
    uPVO = (1/12.0)*(1/e1u)*(1/e3u)*(   f3_ne      * vflux 
                                + ip1(f3_nw) * ip1(vflux)
                                + f3_se      * jm1(vflux)
                                + ip1(f3_sw) * ip1(jm1(vflux)) )

    #Set edge values to zero as done in NEMO
    uPVO[...,0 ,: ] = 0.0
    uPVO[...,-1,: ] = 0.0
    uPVO[...,: ,-1] = 0.0
    uPVO[...,: ,0 ] = 0.0
    
    # If data is from the global model, adjust mask to maintain a consistent
    # masking convention with diagnostics from the model
    if model == 'global':
        umask = u_cube.data.mask
        uPVO_zeroing_mask = tmask + ip1(tmask)
        uPVO_zeroing_mask[...,:,0] = True
        uPVO_zeroing_mask[...,:,-1] = True
        uPVO_zeroing_mask = np.broadcast_to(uPVO_zeroing_mask, uPVO.shape)
        
        uPVO[uPVO_zeroing_mask] = 0.0
    
    uPVO = np.ma.masked_array(uPVO.data, mask=u_cube.data.mask)

    #Calculate the y component of PVO
    vPVO = -(1/12.0)*(1/e2v)*(1/e3v)*(  jp1(f3_sw) * im1(jp1(uflux))
                             + jp1(f3_se) * jp1(uflux)
                                 + f3_nw  * im1(uflux)
                                 + f3_ne  * uflux )

    #Set edge values to zero as done in NEMO
    vPVO[...,0 ,: ] = 0
    vPVO[...,-1,: ] = 0
    vPVO[...,: ,0 ] = 0
    vPVO[...,: ,-1] = 0

    # If data is from the global model, adjust mask to maintain a consistent
    # masking convention with diagnostics from the model
    if model == 'global':
        vmask = v_cube.data.mask
        vPVO_zeroing_mask = tmask + jp1(tmask)
        vPVO_zeroing_mask[...,0 , :] = True
        vPVO_zeroing_mask[...,-1, :] = True
        vPVO_zeroing_mask = np.broadcast_to(vPVO_zeroing_mask, vPVO.shape)
        
        vPVO[vPVO_zeroing_mask] = 0.0

    vPVO = np.ma.masked_array(vPVO.data, mask=v_cube.data.mask)

    time_coord = u_cube.coord("time")
    lat = u_cube.coord("latitude")
    lon = u_cube.coord("longitude")

    #Store uPVO and vPVO as IRIS cubes
    u_bet_cube = Cube(uPVO, dim_coords_and_dims=[(time_coord,0)])
    u_bet_cube.long_name     = 'x component of PVO (assuming constant cell thickness)'
    u_bet_cube.var_name      = 'u_bet'
    u_bet_cube.units         = 'm/s2'
    u_bet_cube.add_aux_coord(lat, [2,3])
    u_bet_cube.add_aux_coord(lon, [2,3])
    u_bet_cube.attributes = {'model':model}

    v_bet_cube = Cube(vPVO, dim_coords_and_dims=[(time_coord,0)])
    v_bet_cube.long_name     = 'y component of PVO (assuming constant cell thickness)'
    v_bet_cube.var_name      = 'v_bet'
    v_bet_cube.units         = 'm/s2'
    v_bet_cube.add_aux_coord(lat, [2,3])
    v_bet_cube.add_aux_coord(lon, [2,3])
    v_bet_cube.attributes = {'model':model}

    return u_bet_cube, v_bet_cube

# ______ _   _ _____                            _      
# | ___ \ | | |  _  |                          | |     
# | |_/ / | | | | | |   _ __  _ __ ___ ___ __ _| | ___ 
# |  __/| | | | | | |  | '_ \| '__/ __/ __/ _` | |/ __|
# | |   \ \_/ | \_/ /  | |_) | | | (_| (_| (_| | | (__ 
# \_|    \___/ \___/   | .__/|_|  \___\___\__,_|_|\___|
#                ______| |                             
#               |______|_|                             

def PVO_prccalc(u_cube, v_cube, ff_f, e1u, e1v, e2u, e2v, e1f, e2f, e3u, e3v, e3t, tmask, fscheme='een_0', model='global'):
    """
    PVO_prccalc(u_cube, v_cube, ff_f, e1u, e1v, e2u, e2v, e1f, e2f, e3u, e3v, e3t, tmask, fscheme='een_0', model='global')

    Calculates the x and y component of the PVO momentum diagnostic while assuming no variation of the Coriolis parameter.

    u_cube - IRIS cube of x-velocities    (t,z,y,x)
    v_cube - IRIS cube of y-velocities    (t,z,y,x)

    ff_f   - Array of coriolis parameter values centred on f points (y,x) 

    e1u    - Array of u cell widths in the x direction        (y,x) [m]
    e1v    - Array of v cell widths in the x direction        (y,x) [m]
    e2u    - Array of u cell widths in the y direction        (y,x) [m]
    e2v    - Array of v cell widths in the y direction        (y,x) [m]
    e1f    - Array of f cell widths in the x direction        (y,x) [m]
    e2f    - Array of f cell widths in the y direction        (y,x) [m]
    e3u    - Array of u cell thicknesses in the z direction (z,y,x) [m]
    e3v    - Array of v cell thicknesses in the z direction (z,y,x) [m]
    e3t    - Array of t cell thicknesses in the z direction (z,y,x) [m]

    tmask  - Mask for the t points (True=Masked)            (z,y,x) [-]

    fscheme- String to describe the coriolis scheme method
                "een_0" - describes the EEN method which determines f cell thickness by
                          e3f = (e3t[i,j] + e3t[i+1,j] + e3t[i,j+1] + e3t[i+1,j+1])/4
                          Where masked values of e3t = 0

                "een_1" - describes the EEN method which determines f cell thickness by
                          e3f = (e3t[i,j] + e3t[i+1,j] + e3t[i,j+1] + e3t[i+1,j+1])/(num_unmasked)
                          Where masked values of e3t = 0 and num_unmasked is the number of unmasked values 
                          in the average


    model  - String to designate which model the data originates from. This is necessary as GYRE and the global model have different masking conventions.
                "global" - Results are from the NEMO global model
                "gyre"   - Results are from GYRE_PISCES
    
    Returns - u_prc_cube, v_prc_cube

    u_prc_cube - IRIS cube, x component of the recreated PVO momentum diagnostic  (t,z,y,x) [m/s2]
    v_prc_cube - IRIS cube, y component of the recreated PVO momentum diagnostic  (t,z,y,x) [m/s2]

    """

    #First determine f cell thicknesses
    e3t_copy = np.ma.copy(e3t)
    e3t_copy[e3t_copy.mask] = 0.0
    
    if fscheme == 'een_0':
        e3f = (e3t_copy + ip1(e3t_copy) + jp1(e3t_copy) + ip1(jp1(e3t_copy)))/4

    else:
        num_masked_t = np.sum([tmask,ip1(tmask),jp1(tmask), ip1(jp1(tmask))], axis=0)
        e3f  = (e3t_copy + ip1(e3t_copy) + jp1(e3t_copy) + ip1(jp1(e3t_copy)))/(4-num_masked_t)
        
        e3f[num_masked_t == 4] = 0.0

    #Set edge values to zero as done in NEMO
    e3f[...,-1,:] = 0.0
    e3f[...,:,-1] = 0.0

    #Calculate 1/e3f
    inv_e3f = 1/e3f

    #Set areas divided by zero to zero
    inv_e3f[e3f==0.0] = 0.0

    if np.ma.is_masked(inv_e3f): inv_e3f = inv_e3f.data

    #Calculate f triads (neglecting variations in f)
    e3_ne = (    inv_e3f  + im1(inv_e3f) +     jm1(inv_e3f))
    e3_nw = (    inv_e3f  + im1(inv_e3f) + im1(jm1(inv_e3f))) 
    e3_se = (    inv_e3f  + jm1(inv_e3f) + im1(jm1(inv_e3f))) 
    e3_sw = (im1(inv_e3f) + jm1(inv_e3f) + im1(jm1(inv_e3f))) 

    #Then set first row and column to zero as done in NEMO
    e3_ne[...,0,:] = 0
    e3_nw[...,0,:] = 0
    e3_se[...,0,:] = 0
    e3_sw[...,0,:] = 0

    e3_ne[...,:,0] = 0
    e3_nw[...,:,0] = 0
    e3_se[...,:,0] = 0
    e3_sw[...,:,0] = 0

    #Calculate x and y fluxes
    uflux = u_cube.data * e2u * e3u
    vflux = v_cube.data * e1v * e3v

    uflux[uflux.mask] = 0.0
    vflux[vflux.mask] = 0.0

    #Calculate the Coriolis parameter centred on u points
    ff_u = (jm1(ff_f)*(e2u-0.5*jm1(e2f)) + 0.5*jm1(e2f)*ff_f )/e2u
    ff_u[...,0,:] = ff_f[...,0,:]

    #Calculate the x component of PVO
    uPVO = (1/12.0)*(ff_u/e1u)*(   e3_ne      * vflux 
                                + ip1(e3_nw) * ip1(vflux)
                                + e3_se      * jm1(vflux)
                                + ip1(e3_sw) * ip1(jm1(vflux)) )

    #Set edge values to zero as done in NEMO
    uPVO[...,0 ,: ] = 0.0
    uPVO[...,-1,: ] = 0.0
    uPVO[...,: ,-1] = 0.0
    
    # If data is from the global model, adjust mask to maintain a consistent
    # masking convention with diagnostics from the model
    if model == 'global':
        umask = u_cube.data.mask
        uPVO_zeroing_mask = tmask + ip1(tmask)
        uPVO_zeroing_mask[...,:,0] = True
        uPVO_zeroing_mask[...,:,-1] = True
        uPVO_zeroing_mask = np.broadcast_to(uPVO_zeroing_mask, uPVO.shape)
        
        uPVO[uPVO_zeroing_mask] = 0.0

    uPVO = np.ma.masked_array(uPVO.data, mask=u_cube.data.mask)

    #Calculate the Coriolis parameter centred on v points
    ff_v = (im1(ff_f)*(e1v-0.5*im1(e1f)) + 0.5*im1(e1f)*ff_f )/e1v
    ff_v[...,:,0] = ff_f[...,:,0]

    #Calculate the y component of PVO
    vPVO = -(1/12.0)*(ff_v/e2v)*(  jp1(e3_sw) * im1(jp1(uflux))
                             + jp1(e3_se) * jp1(uflux)
                                 + e3_nw  * im1(uflux)
                                 + e3_ne  * uflux )

    #Set the edge values to zero as done in NEMO
    vPVO[...,0 ,: ] = 0
    vPVO[...,-1,: ] = 0
    vPVO[...,: ,0 ] = 0

    # If data is from the global model, adjust mask to maintain a consistent
    # masking convention with diagnostics from the model    
    if model == 'global':
        vmask = v_cube.data.mask
        vPVO_zeroing_mask = tmask + jp1(tmask)
        vPVO_zeroing_mask[...,0 , :] = True
        vPVO_zeroing_mask[...,-1, :] = True
        vPVO_zeroing_mask = np.broadcast_to(vPVO_zeroing_mask, vPVO.shape)
        
        vPVO[vPVO_zeroing_mask] = 0.0

    vPVO = np.ma.masked_array(vPVO.data, mask=v_cube.data.mask)

    time_coord = u_cube.coord("time")
    lat = u_cube.coord("latitude")
    lon = u_cube.coord("longitude")

    ff_u_coord = AuxCoord(ff_u, long_name='ff_u', units='1/s')
    ff_v_coord = AuxCoord(ff_v, long_name='ff_v', units='1/s')

    #Store uPVO and vPVO as IRIS cubes
    u_prc_cube = Cube(uPVO, dim_coords_and_dims=[(time_coord,0)])
    u_prc_cube.long_name     = 'x component of PVO (assuming f = f0)'
    u_prc_cube.var_name      = 'u_prc'
    u_prc_cube.units         = 'm/s2'
    u_prc_cube.add_aux_coord(lat, [2,3])
    u_prc_cube.add_aux_coord(lon, [2,3])
    u_prc_cube.add_aux_coord(ff_u_coord, [2,3])
    u_prc_cube.attributes = {'fscheme':fscheme, 'model':model}

    v_prc_cube = Cube(vPVO, dim_coords_and_dims=[(time_coord,0)])
    v_prc_cube.long_name     = 'y component of PVO (assuming f = f0)'
    v_prc_cube.var_name      = 'v_prc'
    v_prc_cube.units         = 'm/s2'
    v_prc_cube.add_aux_coord(lat, [2,3])
    v_prc_cube.add_aux_coord(lon, [2,3])
    v_prc_cube.add_aux_coord(ff_v_coord, [2,3])
    v_prc_cube.attributes = {'fscheme':fscheme, 'model':model}

    return u_prc_cube, v_prc_cube
