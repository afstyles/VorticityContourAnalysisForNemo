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

"""

import numpy as np
import iris

#  _____       _          _     _     _   _____     _                  _   
# /  __ \     | |        | |   (_)   | | |  ___|   | |                | |  
# | /  \/_   _| |__   ___| |    _ ___| |_| |____  _| |_ _ __ __ _  ___| |_ 
# | |   | | | | '_ \ / _ \ |   | / __| __|  __\ \/ / __| '__/ _` |/ __| __|
# | \__/\ |_| | |_) |  __/ |___| \__ \ |_| |___>  <| |_| | | (_| | (__| |_ 
#  \____/\__,_|_.__/ \___\_____/_|___/\__\____/_/\_\\__|_|  \__,_|\___|\__|
                                                                                                                                       
def CubeListExtract(cube_list, var_name):
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
    
        return cube_list[index]
    
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
