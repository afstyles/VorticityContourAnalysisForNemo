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

import sys
import os
sys.path.append(os.path.abspath("../lib"))
from cube_prep import CubeListExtract as CLE

#  _   _            _  ______ _                             _   _      _____ ______ 
# | | | |          | | |  _  (_)                           | | (_)    / __  \|  _  \
# | | | | ___  _ __| |_| | | |_  __ _  __ _ _ __   ___  ___| |_ _  ___`' / /'| | | |
# | | | |/ _ \| '__| __| | | | |/ _` |/ _` | '_ \ / _ \/ __| __| |/ __| / /  | | | |
# \ \_/ / (_) | |  | |_| |/ /| | (_| | (_| | | | | (_) \__ \ |_| | (__./ /___| |/ / 
#  \___/ \___/|_|   \__|___/ |_|\__,_|\__, |_| |_|\___/|___/\__|_|\___\_____/|___/  
#                                      __/ |                                        
#                                     |___/                                    
     
def VortDiagnostic2D( data_list, grid_list, VarDict, MaskDict, PVO_mdecomp_dict, icelog=False, uice_cube=None, vice_cube=None):
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
    
    icelog - Boolean = True if you want to consider sea ice stresses
    uice_cube - IRIS cube containing accelerations associated with sea ice surface stresses in the x direction (t,y,x) [m/s2]
    vice_cube - IRIS cube containing accelerations associated with sea ice surface stresses in the v direction (t,y,x) [m/s2]

    OUTPUT variables
    Returns a dictionary of IRIS cubes that are the associated vorticity diagnostics and the vorticity field of the depth-integrated flow

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

    VORTICITY      - Curl of the depth-integrated flow                               (t,y,x) [m/s]
    """


    ff_f = iris.util.squeeze(CLE(grid_list, VarDict['ff_f'])).data
    e1u = iris.util.squeeze(CLE(grid_list, VarDict['e1u'])).data
    e2u = iris.util.squeeze(CLE(grid_list, VarDict['e2u'])).data
    e1v = iris.util.squeeze(CLE(grid_list, VarDict['e1v'])).data
    e2v = iris.util.squeeze(CLE(grid_list, VarDict['e2v'])).data
    e1f = iris.util.squeeze(CLE(grid_list, VarDict['e1f'])).data
    e2f = iris.util.squeeze(CLE(grid_list, VarDict['e2f'])).data
    e3u = iris.util.squeeze(CLE(grid_list, VarDict['e3u'])).data
    e3v = iris.util.squeeze(CLE(grid_list, VarDict['e3v'])).data
    e3t = iris.util.squeeze(CLE(grid_list, VarDict['e3t'])).data
    umask = MaskDict['umask']
    vmask = MaskDict['vmask']
    tmaskutil = MaskDict['tmaskutil']
    umaskutil = MaskDict['umaskutil']
    vmaskutil = MaskDict['vmaskutil']

    u_cube = CLE(data_list, VarDict['u'])
    v_cube = CLE(data_list, VarDict['v'])


    #Calculate the Coriolis parameter centred on u points
    ff_u = (jm1(ff_f)*(e2u-0.5*jm1(e2f)) + 0.5*jm1(e2f)*ff_f )/e2u
    ff_u[...,0,:] = ff_f[...,0,:]

    #Calculate the Coriolis parameter centred on v points
    ff_v = (im1(ff_f)*(e1v-0.5*im1(e1f)) + 0.5*im1(e1f)*ff_f )/e1v
    ff_v[...,:,0] = ff_f[...,:,0]
    
    #Depth integrate the momentum trends and velocities
    momlist = ['_keg', '_rvo', '_pvo', '_hpg', '_ldf', '_zdf', '_zad', '_tot' ]
    curl_dict = {}
    for label in momlist:
        umom_cube = CLE(data_list, VarDict['u' + label])
        vmom_cube = CLE(data_list, VarDict['v' + label])
        curl_dict[label] = Mom2Vort(umom_cube, vmom_cube, e3u, e3v, umask, vmask, e1u, e2v, e1f, e2f)

    PVO_momlist = ['_bet', '_prc', '_nul', '_pvo2']
    for label in PVO_momlist:
        umom_cube = PVO_mdecomp_dict['u' + label]
        vmom_cube = PVO_mdecomp_dict['v' + label]
        curl_dict[label] = Mom2Vort(umom_cube, vmom_cube, e3u, e3v, umask, vmask, e1u, e2v, e1f, e2f)

    #Also calculate the relative vorticity
    curl_dict['_vorticity'] = Mom2Vort(u_cube, v_cube, e3u, e3v, umask, vmask, e1u, e2v, e1f, e2f)
    
    if icelog == True:
        uice_zint = CLE(data_list, VarDict['u_ice']).data * umaskutil
        vice_zint = CLE(data_list, VarDict['v_ice']).data * vmaskutil
        
        uwnd_zint = CLE(data_list, VarDict['u_tau']).data * umaskutil
        vwnd_zint = CLE(data_list, VarDict['v_tau']).data * vmaskutil

        curl_dict['_ice'] = CGO.kcurl_orca(uice_zint, vice_zint, e1u, e2v, e1f, e2f)
        curl_dict['_wnd'] = CGO.kcurl_orca(uwnd_zint, vwnd_zint, e1u, e2v, e1f, e2f)
        curl_dict['_frc'] = curl_dict['_zdf'] - curl_dict['_wnd'] - curl_dict['_ice']
        
    else:
        uwnd_zint = CLE(data_list, VarDict['u_tau'] ).data*(e3u[0,...])/2
        vwnd_zint = CLE(data_list, VarDict['v_tau'] ).data*(e3v[0,...])/2

        curl_dict['_wnd'] = CGO.kcurl_orca(uwnd_zint, vwnd_zint, e1u, e2v, e1f, e2f)
        curl_dict['_frc'] = curl_dict['_zdf'] - curl_dict['_wnd']

    
    #Calculate total advective contribution
    curl_dict['_adv'] = curl_dict['_keg'] + curl_dict['_rvo'] + curl_dict['_zad']

    #Calculate residual of vorticity budget:
    # Budget :  TOT = KEG + RVO + PVO + HPG + LDF + ZDF + ZAD
    # Residual: RES = KEG + RVO + PVO + HPG + LDF + ZDF + ZAD - TOT

    curl_dict['_res'] = ( curl_dict['_adv'] + curl_dict['_pvo'] + curl_dict['_hpg'] 
                + curl_dict['_ldf'] + curl_dict['_zdf'] - curl_dict['_tot'] )

    #Determine the fmask
    fmask = np.ma.make_mask(tmaskutil * jp1(tmaskutil) * ip1(tmaskutil) * ip1(jp1(tmaskutil)))
    fmask[...,: ,-1]   = False
    fmask[...,-1,: ]   = False

    #Decompose PVO into 5 meaningful parts
    curl_dict['_fdu'] = PVO_divcalc(u_cube, v_cube, ff_f, e1u, e1v, e2u, e2v, e1f, e2f, e3u, e3v, umask, vmask, fmask, f_inside=False)  #f divh(U) effect of divergences 
    curl_dict['_div'] = PVO_divcalc(u_cube, v_cube, ff_f, e1u, e1v, e2u, e2v, e1f, e2f, e3u, e3v, umask, vmask, fmask, f_inside=True )  # divh(fU) analytic form
    curl_dict['_mlv'] = curl_dict['_nul'] - curl_dict['_div']  #Changes in model level
    curl_dict['_bet'] = curl_dict['_bet'] - curl_dict['_nul']  #F displacement term
    curl_dict['_prc'] = curl_dict['_prc'] - curl_dict['_nul']  #Partial cells

    #Also calculate f0 divh(U) for error estimation when contour integrating
    curl_dict['_mfdu'] = np.max(np.abs(ff_f))*np.abs(curl_dict['_fdu']/ff_f)

    #Save outputs as IRIS cubes >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    time_coord  = u_cube.coord('time')
    
    lat = u_cube.coord('latitude')
    lon = u_cube.coord('longitude')

    #Store vorticity diagnostic arrays as IRIS cubes
    output_dict = {}

    for label in curl_dict:
        array = curl_dict[label]
        if np.ma.is_masked(array): array = array.data
        array = np.ma.masked_array( array, mask=np.broadcast_to(~fmask, array.shape)  )
        output_dict[label] = Cube(array, dim_coords_and_dims=[(time_coord,0)])    
        output_dict[label].var_name      = 'curl' + label + '_zint'
        output_dict[label].units         = 'm/s2'
        output_dict[label].add_aux_coord(lat, [-2,-1])
        output_dict[label].add_aux_coord(lon, [-2,-1])

    #Correct units for the relative vorticity Cube
    output_dict['_vorticity'].units = 'm/s'

    #Give IRIS cubes long_names that are more readable
    output_dict['_keg'].long_name = 'k-curl of depth-integrated kinetic energy gradient trend'
    output_dict['_rvo'].long_name = 'k-curl of depth-integrated relative vorticity trend'
    output_dict['_pvo'].long_name = 'k-curl of depth-integrated planetary vorticity trend'
    output_dict['_hpg'].long_name = 'k-curl of depth-integrated horizontal pressure gradient trend'
    output_dict['_ldf'].long_name = 'k-curl of depth-integrated lateral diffusion trend'
    output_dict['_zdf'].long_name = 'k-curl of depth-integrated vertical diffusion trend'
    output_dict['_zad'].long_name = 'k-curl of depth-integrated vertical advection trend'
    output_dict['_tot'].long_name = 'k-curl of depth-integrated total before time stepping trend'
    output_dict['_wnd'].long_name = 'k-curl of depth-integrated wind stress (partial ZDF) trend'
    output_dict['_frc'].long_name = 'k-curl of depth-integrated lateral friction (partial ZDF) trend'
    output_dict['_adv'].long_name = 'k-curl of depth-integrated total advection trend'
    output_dict['_res'].long_name = 'k-curl of depth-integrated residual trend'
    output_dict['_pvo2'].long_name = 'Recreation of PVO'
    output_dict['_fdu'].long_name = 'Calculation of f divh(U)'
    output_dict['_mfdu'].long_name = 'Calculation of |f0 divh(U)|'
    output_dict['_div'].long_name = 'Calculation of divh(fU)'
    output_dict['_mlv'].long_name = 'PVO contribution due to changes in lowest model level'
    output_dict['_bet'].long_name = 'PVO contribution due to f displacement term'
    output_dict['_prc'].long_name = 'PVO contribution due to varying cell thickness'
    output_dict['_vorticity'].long_name = 'Curl of the depth-integrated velocity field'

    if icelog == True:
        output_dict['_ice'].long_name     = 'k-curl of sea ice surface stress trend'

    return output_dict

def Mom2Vort(umom_cube, vmom_cube, e3u, e3v, umask, vmask, e1u, e2v, e1f, e2f):
    """
    Calculate a vorticity diagnostic from the two corresponding horizontal momentum diagnostics.

    The momentum diagnostics are depth-integrated and then the curl is taken

    Returns a numpy array

    """

    umom_zint     = np.sum(umom_cube.data * e3u * umask, axis = -3)
    vmom_zint     = np.sum(vmom_cube.data * e3v * vmask, axis = -3)

    curl_mom  = CGO.kcurl_orca(umom_zint, vmom_zint, e1u, e2v, e1f, e2f )

    return curl_mom

# ______ _   _ _____     _ _                _      
# | ___ \ | | |  _  |   | (_)              | |     
# | |_/ / | | | | | | __| |___   _____ __ _| | ___ 
# |  __/| | | | | | |/ _` | \ \ / / __/ _` | |/ __|
# | |   \ \_/ | \_/ / (_| | |\ V / (_| (_| | | (__ 
# \_|    \___/ \___/ \__,_|_| \_/ \___\__,_|_|\___|
#                ______                            
#               |______|                           

def PVO_divcalc(u_cube, v_cube, ff_f, e1u, e1v, e2u, e2v, e1f, e2f, e3u, e3v, umask, vmask, fmask, f_inside=False ):
    """
    PVO_divcalc(u_cube, v_cube, ff_f, e1u, e1v, e2u, e2v, e1f, e2f, e3u, e3v, fmask, f_inside=False )

    Calculates the component of PVO due to the divergence of the depth integrated velocity field

    Returns f * divh(U) (if f_inside == False)
    Returns divh(f*U)   (if f_inside == True )
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

    f_inside - Logical. 
               Calculate f * divh(U) if f_inside == False (default behaviour)
               Calculate divh(f*U) if f_inside == True

    Returns

    PVO_div - Array of f*divh(U) or divh(f*U) centred on f points         (t,y,x) [m/s2]



    """    
    uflux = u_cube.data * e2u * e3u * umask
    vflux = v_cube.data * e1v * e3v * vmask
    
    uflux_zint = np.sum(uflux, axis=-3)
    vflux_zint = np.sum(vflux, axis=-3)

    if f_inside == False:
    
        divhU = ( ip1(jp1(uflux_zint)) + ip1(uflux_zint) - im1(uflux_zint) - im1(jp1(uflux_zint)) 
                +jp1(vflux_zint) + ip1(jp1(vflux_zint)) - jm1(vflux_zint) - jm1(ip1(vflux_zint)))/(4*e1f*e2f)

    elif f_inside == True:

        #Calculate the Coriolis parameter centred on u points
        ff_u = (jm1(ff_f)*(e2u-0.5*jm1(e2f)) + 0.5*jm1(e2f)*ff_f )/e2u
        ff_u[...,0,:] = ff_f[...,0,:]

        #Calculate the Coriolis parameter centred on v points
        ff_v = (im1(ff_f)*(e1v-0.5*im1(e1f)) + 0.5*im1(e1f)*ff_f )/e1v
        ff_v[...,:,0] = ff_f[...,:,0]

        divhU = ( ip1(jp1(ff_u*uflux_zint)) + ip1(ff_u*uflux_zint) - im1(ff_u*uflux_zint) - im1(jp1(ff_u*uflux_zint)) 
                +jp1(ff_v*vflux_zint) + ip1(jp1(ff_v*vflux_zint)) - jm1(ff_v*vflux_zint) - jm1(ip1(ff_v*vflux_zint)))/(4*e1f*e2f)



    divhU[...,: ,0 ] = 0
    divhU[...,: ,-1] = 0
    divhU[...,0 ,: ] = 0
    divhU[...,-1,: ] = 0
    
    if f_inside == False: PVO_div = - ff_f * divhU
    elif f_inside == True: PVO_div = - divhU

    PVO_div = np.ma.masked_array(PVO_div, mask=~np.broadcast_to(fmask, PVO_div.shape))
    
    return PVO_div

# ______ _   _ _____   __       _ _           _      
# | ___ \ | | |  _  | / _|     | | |         | |     
# | |_/ / | | | | | || |_ _   _| | | ___ __ _| | ___ 
# |  __/| | | | | | ||  _| | | | | |/ __/ _` | |/ __|
# | |   \ \_/ | \_/ /| | | |_| | | | (_| (_| | | (__ 
# \_|    \___/ \___/ |_|  \__,_|_|_|\___\__,_|_|\___|
#                ______                              
#               |______|                             

def PVO_fullcalc( data_list, grid_list, VarDict, MaskDict, fscheme='een_0', no_pen_masked_log=True):
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

    u_cube = CLE(data_list, VarDict['u'])
    v_cube = CLE(data_list, VarDict['v'])

    ff_f = iris.util.squeeze(CLE(grid_list, VarDict['ff_f'])).data
    e1u = iris.util.squeeze(CLE(grid_list, VarDict['e1u'])).data
    e2u = iris.util.squeeze(CLE(grid_list, VarDict['e2u'])).data
    e1v = iris.util.squeeze(CLE(grid_list, VarDict['e1v'])).data
    e2v = iris.util.squeeze(CLE(grid_list, VarDict['e2v'])).data
    e1f = iris.util.squeeze(CLE(grid_list, VarDict['e1f'])).data
    e2f = iris.util.squeeze(CLE(grid_list, VarDict['e2f'])).data
    e3u = iris.util.squeeze(CLE(grid_list, VarDict['e3u'])).data
    e3v = iris.util.squeeze(CLE(grid_list, VarDict['e3v'])).data
    e3t = iris.util.squeeze(CLE(grid_list, VarDict['e3t'])).data
    tmask = MaskDict['tmask']
    umask = MaskDict['umask']
    vmask = MaskDict['vmask']

    if fscheme == 'een_0' or fscheme =='een_1':
        #First determine the f cell thicknesses (only needed for EEN schemes)
        e3t_copy = np.ma.copy(e3t)
        e3t_copy = e3t_copy * tmask
    
        if fscheme == 'een_0':
            e3f = (e3t_copy + ip1(e3t_copy) + jp1(e3t_copy) + ip1(jp1(e3t_copy)))/4

        elif fscheme == 'een_1':
            #Calculate number of masked t points surrounding the f point
            num_unmasked_t = np.sum([tmask,ip1(tmask),jp1(tmask), ip1(jp1(tmask))], axis=0)

            e3f  = (e3t_copy + ip1(e3t_copy) + jp1(e3t_copy) + ip1(jp1(e3t_copy)))/(num_unmasked_t)

            #F points that are completely surrounded by masked t points are zeroed
            e3f[num_unmasked_t == 0] = 0.0

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
        uflux = u_cube.data * e2u * e3u * umask
        vflux = v_cube.data * e1v * e3v * vmask



    else:
        zwz = np.ma.copy(np.broadcast_to(ff_f, e3u.shape))
        uflux = u_cube.data * e2u * umask
        vflux = v_cube.data * e1v * vmask

    #Calculate x component of PVO momentum diagnostic
    if fscheme == 'een_0' or fscheme == 'een_1':
        uPVO = (1/12.0)*(1/e1u)*(   f3_ne      * vflux 
                                    + ip1(f3_nw) * ip1(vflux)
                                    + f3_se      * jm1(vflux)
                                    + ip1(f3_sw) * ip1(jm1(vflux)) )

    elif fscheme == 'ens':
        uPVO = (1/8.0)*(1/e1u)*( jm1(vflux) + ip1(jm1(vflux))
                                   + vflux  +     ip1(vflux))*(jm1(zwz) + zwz)

    elif fscheme == 'ene':
        uPVO = (1/4.0)*(1/e1u)*( jm1(zwz)*( jm1(vflux) + ip1(jm1(vflux)) )
                                +    zwz *(     vflux +      ip1(vflux)  ) )

    else: 
        print(f">> unknown Coriolis scheme >> {fscheme} ")
        return

    #Set edge values to zero as done in NEMO
    uPVO[...,0 ,: ] = 0.0
    uPVO[...,-1,: ] = 0.0
    uPVO[...,: ,-1] = 0.0
    
    
    # If data is from the global model, adjust mask to maintain a consistent
    # masking convention with diagnostics from the model
    if no_pen_masked_log == False:
        uPVO_zeroing_mask = tmask * ip1(tmask)
        uPVO_zeroing_mask[...,:,0] = False
        uPVO_zeroing_mask[...,:,-1] = False
        uPVO = uPVO * uPVO_zeroing_mask
        

    uPVO = np.ma.masked_array(uPVO.data, mask=np.broadcast_to(~umask, uPVO.shape))


    #Calculate y component of PVO momentum diagnostic
    if fscheme == 'een_0' or fscheme == 'een_1':
        vPVO = -(1/12.0)*(1/e2v)*(  jp1(f3_sw) * im1(jp1(uflux))
                                + jp1(f3_se) * jp1(uflux)
                                    + f3_nw  * im1(uflux)
                                    + f3_ne  * uflux )
    
    elif fscheme == 'ens':
        vPVO = -(1/8.0)*(1/e2v)*(   im1(uflux) + im1(jp1(uflux)) 
                                    +   uflux  +     jp1(uflux))*(im1(zwz) + zwz)
    
    elif fscheme == 'ene':
        vPVO = -(1/4.0)*(1/e2v)*( im1(zwz)*( im1(uflux) + im1(jp1(uflux)) )
                                         +    zwz *(     uflux +      jp1(uflux)) )

    else: 
        print(f">> unknown Coriolis scheme >> {fscheme} ")
        return
    

    #Set edge values to zero as done in NEMO
    vPVO[...,0 ,: ] = 0
    vPVO[...,-1,: ] = 0
    vPVO[...,: ,0 ] = 0

    # If data is from the global model, adjust mask to maintain a consistent
    # masking convention with diagnostics from the model
    if no_pen_masked_log == False:
        vPVO_zeroing_mask = tmask * jp1(tmask)
        vPVO_zeroing_mask[...,0 , :] = False
        vPVO_zeroing_mask[...,-1, :] = False
        vPVO = vPVO * vPVO_zeroing_mask
    
    vPVO = np.ma.masked_array(vPVO.data, mask=np.broadcast_to(~vmask, vPVO.shape))

    #Store uPVO and vPVO as IRIS cubes
    time_coord = u_cube.coord("time")
    lat = u_cube.coord("latitude")
    lon = u_cube.coord("longitude")

    if fscheme == 'een_0' or fscheme == 'een_1':
        e3f_coord = AuxCoord(e3f.data, long_name='e3f', units='m')

    u_pvo_cube = Cube(uPVO, dim_coords_and_dims=[(time_coord,0)])
    u_pvo_cube.long_name     = 'x component of topographic coriolis acceleration'
    u_pvo_cube.var_name      = 'u_pvo2'
    u_pvo_cube.units         = 'm/s2'
    u_pvo_cube.add_aux_coord(lat, [2,3])
    u_pvo_cube.add_aux_coord(lon, [2,3])
    if fscheme == 'een_0' or fscheme == 'een_1': 
        u_pvo_cube.add_aux_coord(e3f_coord, [1,2,3])
    u_pvo_cube.attributes = {'fscheme':fscheme, 'no_pen_masked_log':str(no_pen_masked_log)}


    v_pvo_cube = Cube(vPVO, dim_coords_and_dims=[(time_coord,0)])
    v_pvo_cube.long_name     = 'y component of topographic coriolis acceleration'
    v_pvo_cube.var_name      = 'v_pvo2'
    v_pvo_cube.units         = 'm/s2'
    v_pvo_cube.add_aux_coord(lat, [2,3])
    v_pvo_cube.add_aux_coord(lon, [2,3])
    if fscheme == 'een_0' or fscheme == 'een_1':
        v_pvo_cube.add_aux_coord(e3f_coord, [1,2,3])
    v_pvo_cube.attributes = {'fscheme':fscheme, 'no_pen_masked_log':str(no_pen_masked_log)}

    return u_pvo_cube, v_pvo_cube

# ______ _   _ _____              _           _      
# | ___ \ | | |  _  |            | |         | |     
# | |_/ / | | | | | | _ __  _   _| | ___ __ _| | ___ 
# |  __/| | | | | | || '_ \| | | | |/ __/ _` | |/ __|
# | |   \ \_/ | \_/ /| | | | |_| | | (_| (_| | | (__ 
# \_|    \___/ \___/ |_| |_|\__,_|_|\___\__,_|_|\___|
#                ______                              
#               |______|                             

def PVO_nulcalc(data_list, grid_list, VarDict, MaskDict, no_pen_masked_log = True, fscheme='een_0'):
    """
    PVO_nullcalc(u_cube, v_cube, ff_f, e1u, e1v, e2u, e2v, e1f, e2f, e3u, e3v, tmask, model='global', fscheme='een_0')

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

    u_cube = CLE(data_list, VarDict['u'])
    v_cube = CLE(data_list, VarDict['v'])

    ff_f = iris.util.squeeze(CLE(grid_list, VarDict['ff_f'])).data
    e1u = iris.util.squeeze(CLE(grid_list, VarDict['e1u'])).data
    e2u = iris.util.squeeze(CLE(grid_list, VarDict['e2u'])).data
    e1v = iris.util.squeeze(CLE(grid_list, VarDict['e1v'])).data
    e2v = iris.util.squeeze(CLE(grid_list, VarDict['e2v'])).data
    e1f = iris.util.squeeze(CLE(grid_list, VarDict['e1f'])).data
    e2f = iris.util.squeeze(CLE(grid_list, VarDict['e2f'])).data
    e3u = iris.util.squeeze(CLE(grid_list, VarDict['e3u'])).data
    e3v = iris.util.squeeze(CLE(grid_list, VarDict['e3v'])).data
    e3t = iris.util.squeeze(CLE(grid_list, VarDict['e3t'])).data
    tmask = MaskDict['tmask']
    umask = MaskDict['umask']
    vmask = MaskDict['vmask']

    # Use 3d flux even if not EEN scheme
    uflux = u_cube.data * e2u * e3u * umask 
    vflux = v_cube.data * e1v * e3v * vmask

    #Calculate the Coriolis parameter centred on the u point
    ff_u = (jm1(ff_f)*(e2u-0.5*jm1(e2f)) + 0.5*jm1(e2f)*ff_f )/e2u
    ff_u[...,0,:] = ff_f[...,0,:]

    #Calculate the Coriolis parameter centred on the v point
    ff_v = (im1(ff_f)*(e1v-0.5*im1(e1f)) + 0.5*im1(e1f)*ff_f )/e1v
    ff_v[...,:,0] = ff_f[...,:,0]
    
    #Calculate the x component of PVO (valid form for all fschemes)
    uPVO = (1/4.0)*(1/e1u)*(1/e3u)*(    ff_v*vflux 
                                + ip1(ff_v*vflux)
                                + jm1(ff_v*vflux)
                                + ip1(jm1(ff_v*vflux)) )

    #Set edge values to zero as done in NEMO
    uPVO[...,0 ,: ] = 0.0
    uPVO[...,-1,: ] = 0.0
    uPVO[...,: ,-1] = 0.0
    uPVO[...,: ,0 ] = 0.0
    
    # If data is from the global model, adjust mask to maintain a consistent
    # masking convention with diagnostics from the model
    if no_pen_masked_log == False:
        uPVO_zeroing_mask = tmask * ip1(tmask)
        uPVO_zeroing_mask[...,:,0] = False
        uPVO_zeroing_mask[...,:,-1] = False        
        uPVO = uPVO * uPVO_zeroing_mask

    uPVO = np.ma.masked_array(uPVO.data, mask=np.broadcast_to(~umask, uPVO.shape))

    #Calculate the y component of PVO
    vPVO = -(1/4.0)*(1/e2v)*(1/e3v)*( im1(jp1(ff_u*uflux))
                                             + jp1(ff_u*uflux)
                                             + im1(ff_u*uflux)
                                             + ff_u*uflux )

    #Set edge values to zero as done in NEMO
    vPVO[...,0 ,: ] = 0
    vPVO[...,-1,: ] = 0
    vPVO[...,: ,0 ] = 0
    vPVO[...,: ,-1] = 0
    
    # If data is from the global model, adjust mask to maintain a consistent
    # masking convention with diagnostics from the model
    if no_pen_masked_log == False:
        vPVO_zeroing_mask = tmask * jp1(tmask)
        vPVO_zeroing_mask[...,0 , :] = False
        vPVO_zeroing_mask[...,-1, :] = False
        vPVO = vPVO * vPVO_zeroing_mask

    vPVO = np.ma.masked_array(vPVO.data, mask=np.broadcast_to(~vmask, vPVO.shape))


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
    u_nul_cube.attributes = {'no_pen_masked_log':str(no_pen_masked_log), 'fscheme':fscheme}

    v_nul_cube = Cube(vPVO, dim_coords_and_dims=[(time_coord,0)])
    v_nul_cube.long_name     = 'y component of PVO (assuming constant f and cell thickness)'
    v_nul_cube.var_name      = 'v_nul'
    v_nul_cube.units         = 'm/s2'
    v_nul_cube.add_aux_coord(lat, [2,3])
    v_nul_cube.add_aux_coord(lon, [2,3])
    v_nul_cube.add_aux_coord(ff_v_coord, [2,3])
    v_nul_cube.attributes = {'no_pen_masked_log':str(no_pen_masked_log), 'fscheme':fscheme}

    return u_nul_cube, v_nul_cube

# ______ _   _ _____  _          _            _      
# | ___ \ | | |  _  || |        | |          | |     
# | |_/ / | | | | | || |__   ___| |_ ___ __ _| | ___ 
# |  __/| | | | | | || '_ \ / _ \ __/ __/ _` | |/ __|
# | |   \ \_/ | \_/ /| |_) |  __/ || (_| (_| | | (__ 
# \_|    \___/ \___/ |_.__/ \___|\__\___\__,_|_|\___|
#                ______                              
#               |______|                             

def PVO_betcalc(data_list, grid_list, VarDict, MaskDict, no_pen_masked_log = True, fscheme ='een_0'):
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

    u_cube = CLE(data_list, VarDict['u'])
    v_cube = CLE(data_list, VarDict['v'])

    ff_f = iris.util.squeeze(CLE(grid_list, VarDict['ff_f'])).data
    e1u = iris.util.squeeze(CLE(grid_list, VarDict['e1u'])).data
    e2u = iris.util.squeeze(CLE(grid_list, VarDict['e2u'])).data
    e1v = iris.util.squeeze(CLE(grid_list, VarDict['e1v'])).data
    e2v = iris.util.squeeze(CLE(grid_list, VarDict['e2v'])).data
    e1f = iris.util.squeeze(CLE(grid_list, VarDict['e1f'])).data
    e2f = iris.util.squeeze(CLE(grid_list, VarDict['e2f'])).data
    e3u = iris.util.squeeze(CLE(grid_list, VarDict['e3u'])).data
    e3v = iris.util.squeeze(CLE(grid_list, VarDict['e3v'])).data
    tmask = MaskDict['tmask']
    umask = MaskDict['umask']
    vmask = MaskDict['vmask']
    
    if fscheme == 'een_0' or fscheme == 'een_1':
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

    else:
        zwz = np.ma.copy(np.broadcast_to(ff_f, e3u.shape))

    #Calculate x and y volume fluxes
    uflux = u_cube.data * e2u * e3u * umask
    vflux = v_cube.data * e1v * e3v * vmask



    #Calculate the x component of PVO
    if fscheme == 'een_0' or fscheme == 'een_1':
        uPVO = (1/12.0)*(1/e1u)*(1/e3u)*(   f3_ne      * vflux 
                                    + ip1(f3_nw) * ip1(vflux)
                                    + f3_se      * jm1(vflux)
                                    + ip1(f3_sw) * ip1(jm1(vflux)) )

    elif fscheme == 'ens':
        uPVO = (1/8.0)*(1/e1u)*(1/e3u)*( jm1(vflux) + ip1(jm1(vflux))
                                           + vflux  +     ip1(vflux))*(jm1(zwz) + zwz)

    elif fscheme == 'ene':
        uPVO = (1/4.0)*(1/e1u)*(1/e3u)*( jm1(zwz)*( jm1(vflux) + ip1(jm1(vflux)) )
                                        +    zwz *(     vflux +      ip1(vflux)  ) )

    else: 
        print(f">> unknown Coriolis scheme >> {fscheme} ")
        return


    #Set edge values to zero as done in NEMO
    uPVO[...,0 ,: ] = 0.0
    uPVO[...,-1,: ] = 0.0
    uPVO[...,: ,-1] = 0.0
    uPVO[...,: ,0 ] = 0.0
    
    # If data is from the global model, adjust mask to maintain a consistent
    # masking convention with diagnostics from the model
    if no_pen_masked_log == False:
        uPVO_zeroing_mask = tmask * ip1(tmask)
        uPVO_zeroing_mask[...,:,0] = False
        uPVO_zeroing_mask[...,:,-1] = False
        uPVO = uPVO * uPVO_zeroing_mask
    
    uPVO = np.ma.masked_array(uPVO.data, mask=np.broadcast_to(~umask, uPVO.shape))

    #Calculate the y component of PVO
    if fscheme == 'een_0' or fscheme == 'een_1':
        vPVO = -(1/12.0)*(1/e2v)*(1/e3v)*(  jp1(f3_sw) * im1(jp1(uflux))
                                + jp1(f3_se) * jp1(uflux)
                                    + f3_nw  * im1(uflux)
                                    + f3_ne  * uflux )

    elif fscheme == 'ens':
        vPVO = -(1/8.0)*(1/e2v)*(1/e3v)*(   im1(uflux) + im1(jp1(uflux)) 
                                    +   uflux  +     jp1(uflux))*(im1(zwz) + zwz)
    
    elif fscheme == 'ene':
        vPVO = -(1/4.0)*(1/e2v)*(1/e3v)*( im1(zwz)*( im1(uflux) + im1(jp1(uflux)) )
                                         +    zwz *(     uflux +      jp1(uflux)) )

    else: 
        print(f">> unknown Coriolis scheme >> {fscheme} ")
        return

    #Set edge values to zero as done in NEMO
    vPVO[...,0 ,: ] = 0
    vPVO[...,-1,: ] = 0
    vPVO[...,: ,0 ] = 0
    vPVO[...,: ,-1] = 0

    # If data is from the global model, adjust mask to maintain a consistent
    # masking convention with diagnostics from the model
    if no_pen_masked_log == False:
        vPVO_zeroing_mask = tmask * jp1(tmask)
        vPVO_zeroing_mask[...,0 , :] = False
        vPVO_zeroing_mask[...,-1, :] = False
        vPVO = vPVO * vPVO_zeroing_mask

    vPVO = np.ma.masked_array(vPVO.data, mask=np.broadcast_to(~vmask, vPVO.shape))

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
    u_bet_cube.attributes = {'no_pen_masked_log':str(no_pen_masked_log), 'fscheme':fscheme}

    v_bet_cube = Cube(vPVO, dim_coords_and_dims=[(time_coord,0)])
    v_bet_cube.long_name     = 'y component of PVO (assuming constant cell thickness)'
    v_bet_cube.var_name      = 'v_bet'
    v_bet_cube.units         = 'm/s2'
    v_bet_cube.add_aux_coord(lat, [2,3])
    v_bet_cube.add_aux_coord(lon, [2,3])
    v_bet_cube.attributes = {'no_pen_masked_log':str(no_pen_masked_log), 'fscheme':fscheme}

    return u_bet_cube, v_bet_cube

# ______ _   _ _____                            _      
# | ___ \ | | |  _  |                          | |     
# | |_/ / | | | | | |   _ __  _ __ ___ ___ __ _| | ___ 
# |  __/| | | | | | |  | '_ \| '__/ __/ __/ _` | |/ __|
# | |   \ \_/ | \_/ /  | |_) | | | (_| (_| (_| | | (__ 
# \_|    \___/ \___/   | .__/|_|  \___\___\__,_|_|\___|
#                ______| |                             
#               |______|_|                             

def PVO_prccalc( data_list, grid_list, VarDict, MaskDict, fscheme='een_0', no_pen_masked_log = True):
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
    
    u_cube = CLE(data_list, VarDict['u'])
    v_cube = CLE(data_list, VarDict['v'])

    ff_f = iris.util.squeeze(CLE(grid_list, VarDict['ff_f'])).data
    e1u = iris.util.squeeze(CLE(grid_list, VarDict['e1u'])).data
    e2u = iris.util.squeeze(CLE(grid_list, VarDict['e2u'])).data
    e1v = iris.util.squeeze(CLE(grid_list, VarDict['e1v'])).data
    e2v = iris.util.squeeze(CLE(grid_list, VarDict['e2v'])).data
    e1f = iris.util.squeeze(CLE(grid_list, VarDict['e1f'])).data
    e2f = iris.util.squeeze(CLE(grid_list, VarDict['e2f'])).data
    e3u = iris.util.squeeze(CLE(grid_list, VarDict['e3u'])).data
    e3v = iris.util.squeeze(CLE(grid_list, VarDict['e3v'])).data
    e3t = iris.util.squeeze(CLE(grid_list, VarDict['e3t'])).data
    tmask = MaskDict['tmask']
    umask = MaskDict['umask']
    vmask = MaskDict['vmask']

    if fscheme == 'een_0' or fscheme == 'een_1':
        #First determine f cell thicknesses
        e3t_copy = np.ma.copy(e3t)
        e3t_copy = e3t_copy * tmask
        
        if fscheme == 'een_0':
            e3f = (e3t_copy + ip1(e3t_copy) + jp1(e3t_copy) + ip1(jp1(e3t_copy)))/4

        else:
            num_unmasked_t = np.sum([tmask,ip1(tmask),jp1(tmask), ip1(jp1(tmask))], axis=0)
            e3f  = (e3t_copy + ip1(e3t_copy) + jp1(e3t_copy) + ip1(jp1(e3t_copy)))/(num_unmasked_t)
            
            e3f[num_unmasked_t == 0] = 0.0

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
        uflux = u_cube.data * e2u * e3u * umask
        vflux = v_cube.data * e1v * e3v * vmask

    else:
        zwz = np.ma.copy(np.broadcast_to(ff_f, e3u.shape))

        uflux = u_cube.data * e2u * umask
        vflux = v_cube.data * e1v * vmask

    #Calculate the Coriolis parameter centred on u points
    ff_u = (jm1(ff_f)*(e2u-0.5*jm1(e2f)) + 0.5*jm1(e2f)*ff_f )/e2u
    ff_u[...,0,:] = ff_f[...,0,:]

    #Calculate the Coriolis parameter centred on v points
    ff_v = (im1(ff_f)*(e1v-0.5*im1(e1f)) + 0.5*im1(e1f)*ff_f )/e1v
    ff_v[...,:,0] = ff_f[...,:,0]

    #Calculate the x component of PVO
    if fscheme == 'een_0' or fscheme == 'een_1':
        uPVO = (1/12.0)*(1/e1u)*(   e3_ne      * ff_v*vflux 
                                    + ip1(e3_nw) * ip1(ff_v*vflux)
                                    + e3_se      * jm1(ff_v*vflux)
                                    + ip1(e3_sw) * ip1(jm1(ff_v*vflux)) )

    elif fscheme == 'ens':
        uPVO = (1/4.0)*(1/e1u)*( jm1(ff_v*vflux) + ip1(jm1(ff_v*vflux))
                                   + ff_v*vflux  +     ip1(ff_v*vflux))

    elif fscheme == 'ene':
        uPVO = (1/4.0)*(1/e1u)*( jm1(ff_v*vflux) + ip1(jm1(ff_v*vflux)) 
                                +       ff_v*vflux +      ip1(ff_v*vflux)  ) 

    else: 
        print(f">> unknown Coriolis scheme >> {fscheme} ")
        return
                            

    #Set edge values to zero as done in NEMO
    uPVO[...,0 ,: ] = 0.0
    uPVO[...,-1,: ] = 0.0
    uPVO[...,: ,-1] = 0.0
    
    # If data is from the global model, adjust mask to maintain a consistent
    # masking convention with diagnostics from the model
    if no_pen_masked_log == False:
        uPVO_zeroing_mask = tmask * ip1(tmask)
        uPVO_zeroing_mask[...,:,0] = False
        uPVO_zeroing_mask[...,:,-1] = False        
        uPVO = uPVO * uPVO_zeroing_mask

    uPVO = np.ma.masked_array(uPVO.data, mask=np.broadcast_to(~umask, uPVO.shape))

    #Calculate the y component of PVO
    if fscheme == 'een_0' or fscheme =='een_1':
        vPVO = -(1/12.0)*(1/e2v)*(  jp1(e3_sw) * im1(jp1(ff_u*uflux))
                                + jp1(e3_se) * jp1(ff_u*uflux)
                                    + e3_nw  * im1(ff_u*uflux)
                                    + e3_ne  * ff_u*uflux )

    elif fscheme == 'ens':
        vPVO = -(1/4.0)*(1/e2v)*(   im1(ff_u*uflux) + im1(jp1(ff_u*uflux)) 
                                    +   ff_u*uflux  +     jp1(ff_u*uflux))
    
    elif fscheme == 'ene':
        vPVO = -(1/4.0)*(1/e2v)*( ( im1(ff_u*uflux) + im1(jp1(ff_u*uflux)) )
                                    +(     ff_u*uflux +      jp1(ff_u*uflux)) )

    else: 
        print(f">> unknown Coriolis scheme >> {fscheme} ")
        return

    #Set the edge values to zero as done in NEMO
    vPVO[...,0 ,: ] = 0
    vPVO[...,-1,: ] = 0
    vPVO[...,: ,0 ] = 0

    # If data is from the global model, adjust mask to maintain a consistent
    # masking convention with diagnostics from the model    
    if no_pen_masked_log == False:
        vPVO_zeroing_mask = tmask * jp1(tmask)
        vPVO_zeroing_mask[...,0 , :] = False
        vPVO_zeroing_mask[...,-1, :] = False
        vPVO = vPVO * vPVO_zeroing_mask

    vPVO = np.ma.masked_array(vPVO.data, mask=np.broadcast_to(~vmask, vPVO.shape))

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
    u_prc_cube.attributes = {'fscheme':fscheme, 'no_pen_masked_log':str(no_pen_masked_log)}

    v_prc_cube = Cube(vPVO, dim_coords_and_dims=[(time_coord,0)])
    v_prc_cube.long_name     = 'y component of PVO (assuming f = f0)'
    v_prc_cube.var_name      = 'v_prc'
    v_prc_cube.units         = 'm/s2'
    v_prc_cube.add_aux_coord(lat, [2,3])
    v_prc_cube.add_aux_coord(lon, [2,3])
    v_prc_cube.add_aux_coord(ff_v_coord, [2,3])
    v_prc_cube.attributes = {'fscheme':fscheme, 'no_pen_masked_log':str(no_pen_masked_log)}

    return u_prc_cube, v_prc_cube
