#             _                                   
#            | |                                  
#   ___ _   _| |__   ___     _ __  _ __ ___ _ __  
#  / __| | | | '_ \ / _ \   | '_ \| '__/ _ \ '_ \ 
# | (__| |_| | |_) |  __/   | |_) | | |  __/ |_) |
#  \___|\__,_|_.__/ \___|   | .__/|_|  \___| .__/ 
#                     ______| |            | |    
#                    |______|_|            |_|    

"""
cube_prep.py

Methods for loading and manipulating IRIS cubes

Contains methods:
    CubeListExtract --> Extracts a cube with a specific variable name from a cubelist 
    VarDict --> Generate a customizable dictionary of variable names for loading data

"""

import numpy as np
import iris

#  _____       _          _     _     _   _____     _                  _   
# /  __ \     | |        | |   (_)   | | |  ___|   | |                | |  
# | /  \/_   _| |__   ___| |    _ ___| |_| |____  _| |_ _ __ __ _  ___| |_ 
# | |   | | | | '_ \ / _ \ |   | / __| __|  __\ \/ / __| '__/ _` |/ __| __|
# | \__/\ |_| | |_) |  __/ |___| \__ \ |_| |___>  <| |_| | | (_| | (__| |_ 
#  \____/\__,_|_.__/ \___\_____/_|___/\__\____/_/\_\\__|_|  \__,_|\___|\__|
                                                                                                                                       
def CubeListExtract(cube_list, var_name, jmin=None, jmax=None, imin=None, imax=None, timefix=True):
    """
    Extracts a cube with a specific variable name from a cube list. 
    If two cubes have the same variable name in the list then the first occurence in the list is extracted

    cube_list - CubeList object
    var_name - String, variable name of cube to extract

    Returns
    cube - IRIS cube with variable name matching var_name

    """
    
    VarNames = [cube_list[i].var_name for i in range(len(cube_list))]
    
    try:
    
        index = VarNames.index(var_name)

        #Subset the found cube based off jmin, jmax, imin, imax
        cube = cube_list[index]

        
        if (jmin is not None) or (jmax is not None) or (imin is not None) or (imax is not None): 
            cube = cube[...,jmin:jmax,imin:imax]


        #Repair the time coordinate if needed
        if timefix==True:
            try: 
                time_coord = cube.coord("time")
            except:
                if (len(cube.aux_coords) > 0):
                    aux_time = cube.aux_coords[0]
                    aux_time.rename("aux_time")

        return cube
    
    except:
        
        print('Variable name not found')
        
        return 

#                    _                               _    
#                   | |                             | |   
#   __ _ _ __  _ __ | |_   _     _ __ ___   __ _ ___| | __
#  / _` | '_ \| '_ \| | | | |   | '_ ` _ \ / _` / __| |/ /
# | (_| | |_) | |_) | | |_| |   | | | | | | (_| \__ \   < 
#  \__,_| .__/| .__/|_|\__, |   |_| |_| |_|\__,_|___/_|\_\
#       | |   | |       __/ |_____                        
#       |_|   |_|      |___/______|                       

def apply_mask(cube, mask_cube):
    """
    apply_mask(cube, mask_cube)

    Applies a loaded mask IRIS cube to another iris cube. NEMO masks are the opposite to numpy masks
    i.e. 1 = Unmasked, 0 = Masked (NEMO) and True(1) = Masked, False(0) = Unmasked

    cube - IRIS cube to apply mask to (t,...)
    mask_cube - IRIS cube containing mask (t=1,...) if time independent or (t,...) if time dependent
    """

    mask = np.broadcast_to(~np.ma.make_mask(np.squeeze(mask_cube.data)),cube.shape)
    cube.data = np.ma.masked_array(cube.data.data, mask=mask)

    return cube

def coord_repair(cube):
    """
    coord_repair(cube)

    Renames the auxilliary time coordinate to 'aux_time' this prevents confusion when loading the non-auxilliary time coordinate

    INPUT variable
    cube - IRIS cube with a dimension called "time" and an auxilliary coordinate called "time"
    """

    aux_time = cube.aux_coords[0]
    aux_time.rename('aux_time')

    return cube

def GenVarDict(VarDictName='gyre'):

    if VarDictName.lower() == "gyre" :
        output_dict = {
            'u' : 'vozocrtx' ,
            'v' : 'vomecrty',
            'u_keg' : 'utrd_swkeg', 
            'u_rvo' : 'utrd_swrvo',
            'u_pvo' : 'utrd_swpvo',
            'u_hpg' : 'utrd_swhpg',
            'u_ldf' : 'utrd_swldf',
            'u_zdf' : 'utrd_swzdf',
            'u_zad' : 'utrd_swzad',
            'u_tot' : 'utrd_swtot',
            'u_tau' : 'utrd_swtau',
            'v_keg' : 'vtrd_swkeg', 
            'v_rvo' : 'vtrd_swrvo',
            'v_pvo' : 'vtrd_swpvo',
            'v_hpg' : 'vtrd_swhpg',
            'v_ldf' : 'vtrd_swldf',
            'v_zdf' : 'vtrd_swzdf',
            'v_zad' : 'vtrd_swzad',
            'v_tot' : 'vtrd_swtot',
            'v_tau' : 'vtrd_swtau',
            'umask' : 'umask',
            'vmask' : 'vmask',
            'tmask' : 'tmask',
            'tmaskutil' : 'tmaskutil',
            'umaskutil' : 'umaskutil',
            'vmaskutil' : 'vmaskutil',
            'e1u' : 'e1u',
            'e2u' : 'e2u',
            'e1v' : 'e1v',
            'e2v' : 'e2v',
            'e1f' : 'e1f',
            'e2f' : 'e2f',
            'e3u' : 'e3u_0',
            'e3v' : 'e3v_0',
            'e3t' : 'e3t_0',
            'ff_f': 'ff_f' # 
            }
        
    if VarDictName.lower() == "global":
        output_dict = {
            'u' : 'uo' ,
            'v' : 'vo',
            'u_keg' : 'utrd_keg', 
            'u_rvo' : 'utrd_rvo',
            'u_pvo' : 'utrd_pvo',
            'u_hpg' : 'utrd_hpg',
            'u_ldf' : 'utrd_ldf',
            'u_zdf' : 'utrd_zdf',
            'u_zad' : 'utrd_zad',
            'u_tot' : 'utrd_tot',
            'u_tau' : 'utrd_tau2d_hu',
            'u_ice' : 'utrd_tfr2d_hu',
            'v_keg' : 'vtrd_keg', 
            'v_rvo' : 'vtrd_rvo',
            'v_pvo' : 'vtrd_pvo',
            'v_hpg' : 'vtrd_hpg',
            'v_ldf' : 'vtrd_ldf',
            'v_zdf' : 'vtrd_zdf',
            'v_zad' : 'vtrd_zad',
            'v_tot' : 'vtrd_tot',
            'v_tau' : 'vtrd_tau2d_hv',
            'v_ice' : 'vtrd_tfr2d_hv',
            'umask' : 'uo',           #The global model comes with premasked data and the 
            'vmask' : 'vo',           #domcfg file does not provide the masks. In this case,
            'tmask' : 'thetao_con',   # the dictionary will point to an ocean variable that
            'tmaskutil' : 'zos',
            'umaskutil' : 'sozocrtx', # has the correct mask.
            'vmaskutil' : 'somecrty',
            'e1u' : 'e1u',
            'e2u' : 'e2u',
            'e1v' : 'e1v',
            'e2v' : 'e2v',
            'e1f' : 'e1f',
            'e2f' : 'e2f',
            'e3u' : 'e3u_0',
            'e3v' : 'e3v_0',
            'e3t' : 'e3t_0',
            'ff_f': 'ff_f' # 
            }

    return output_dict

def get_mask( data_list, grid_list, maskname, VarDict, method='detached' ):
    """
    Get the mask from a given Nemo data set. 
    Either recover the mask from masked data or load the mask from the mesh_mask file
    """
    varname = VarDict[maskname]

    if method.lower() == 'detached':
        # Mask loaded from grid file follows 1=Unmasked, 0=masked convention
        mask = iris.util.squeeze(CubeListExtract(grid_list, varname)).data
        mask = np.ma.make_mask(mask)
        return mask   #Return boolean mask array with T=Unmasked, F=Masked convention

    elif method.lower() == 'attached':
        # Mask extracted from data file uses T=Masked, F=Unmasked convention
        mask = CubeListExtract(data_list, varname ).data.mask[0,...]
        mask = ~mask  #Switch mask to match NEMO convention.
        return mask   #Return boolean mask array with T=Unmasked, F=Masked convention

    else:
        print("No method selected!!")
        return

