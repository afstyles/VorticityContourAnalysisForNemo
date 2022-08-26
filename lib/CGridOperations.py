#  _____ _____      _     _ _____                      _   _                               
# /  __ \  __ \    (_)   | |  _  |                    | | (_)                              
# | /  \/ |  \/_ __ _  __| | | | |_ __   ___ _ __ __ _| |_ _  ___  _ __  ___   _ __  _   _ 
# | |   | | __| '__| |/ _` | | | | '_ \ / _ \ '__/ _` | __| |/ _ \| '_ \/ __| | '_ \| | | |
# | \__/\ |_\ \ |  | | (_| \ \_/ / |_) |  __/ | | (_| | |_| | (_) | | | \__ \_| |_) | |_| |
#  \____/\____/_|  |_|\__,_|\___/| .__/ \___|_|  \__,_|\__|_|\___/|_| |_|___(_) .__/ \__, |
#                                | |                                          | |     __/ |
#                                |_|                                          |_|    |___/
"""
CGridOperations.py

Standard operations on the C grid

Contains methods:
    im1, ip1, jm1, jp1 --> Shorthand functions for rolling arrays in the i or j direction by one place
    roll_and_mask --> Rolls the array and masks values that wrap around to the other side
    kcurl_orca --> Calculates the vertical component of the curl of u and v points
"""

import numpy as np

def im1(M): 
    """
    im1(M)[..., j ,i ] = M[..., j ,i-1]  for i != 0
    im1(M)[..., j ,0 ] = M[..., j ,-1 ]  for i == 0
    """
    return np.roll(M, 1 , axis=-1)

def ip1(M): 
    """
    ip1(M)[..., j , i ] = M[..., j ,i+1 ] for i != -1
    ip1(M)[..., j ,-1 ] = M[..., j , 0  ] for i == -1 
    """
    return np.roll(M, -1, axis=-1)

def jm1(M):
    """
    jm1(M)[...,  j,i ] = M[..., j-1 , i ] for j != 0
    jm1(M)[..., 0 ,i ] = M[..., -1  , i ] for j == 0
     """
    return np.roll(M, 1 , axis=-2)

def jp1(M):
    """
    jp1(M)[..., j , i ] = M[..., j+1, i ] for j != -1
    jp1(M)[..., -1, i ] = M[..., 0  , i ] for j == -1
    """
    return np.roll(M, -1, axis=-2)

#            _ _                  _                       _    
#           | | |                | |                     | |   
#  _ __ ___ | | |  __ _ _ __   __| |  _ __ ___   __ _ ___| | __
# | '__/ _ \| | | / _` | '_ \ / _` | | '_ ` _ \ / _` / __| |/ /
# | | | (_) | | || (_| | | | | (_| | | | | | | | (_| \__ \   < 
# |_|  \___/|_|_| \__,_|_| |_|\__,_| |_| |_| |_|\__,_|___/_|\_\
#             ______             ______                        
#            |______|           |______|                       

def roll_and_mask(M, shift, axis=-1):
    """
    Roll an array and mask the elements that are wrapped around to the other side of the array. Will work on 
    both masked and unmasked numpy arrays.
    
    M - Array to be rolled and masked                                         (numpy.array or ma.masked_array)
    shift - Number of spaces to be shifted. New index = Original index + shift  (integer)
    axis - Axis to operate on. By default, operate on last axis                 (integer)
    
    Returns a masked numpy array matching shape of M
    M_roll - Masked array                                                     (numpy.ma.masked_array)
    """
    
    #Roll the array using the numpy function
    M_roll = np.roll(M, shift, axis=axis)
    
    if np.ma.is_masked(M_roll):
        mask = M_roll.mask
        
    else:
        mask = np.zeros(M_roll.shape, dtype=bool)
    
    #Calculuate indexes to mask based off shift value
    #e.g. if abs(shift)  = 2 
    # col_to_mask = [1,2]
    col_to_mask = np.array(range(1,np.abs(shift)+1))
       
    if shift > 0:
        #e.g. if shift = 2
        #col_to_mask [1,2] --> [0,1]
        col_to_mask = col_to_mask - 1
        
    elif shift < 0:
        #e.g. if shift = -2
        #col_to_mask [1,2] --> [-1,-2]
        col_to_mask = -col_to_mask
        
    else:
        print("M is unchanged")
        return M
    
    #Mask all values that have wrapped around to the other side of the array
    mask = np.swapaxes(mask, axis, -1)
    mask[...,col_to_mask] = True
    mask = np.swapaxes(mask, axis, -1)
    
    return np.ma.masked_array(M_roll, mask = mask)

#  _                   _                      
# | |                 | |                     
# | | _____ _   _ _ __| |  ___  _ __ ___ __ _ 
# | |/ / __| | | | '__| | / _ \| '__/ __/ _` |
# |   < (__| |_| | |  | || (_) | | | (_| (_| |
# |_|\_\___|\__,_|_|  |_| \___/|_|  \___\__,_|
#                     ______                  
#                    |______|                 

def kcurl_orca(u, v, e1u, e2v, e1f, e2f, xaxis=-1, yaxis=-2):
    """
    Function to calculate the k-component of the curl on a C grid with no assumptions of grid regularity.
    
    Uses the following discretization centred on the f point
    
    dot(k,curl(u,v)) = [ (v*e2v[i+1,j] - v*e2v[ij]) - (u*e1u[i,j+1] - u*e1u[i,j])  ]/(e1f[ij]*e2f[ij])
    
    u - Array of vector x component (numpy array)        [...,y,x]
    v - Array of vector y component (numpy array)        [...,y,x]
    e1u, e2v. e1f, e2f - Cell thicknesses (numpy array)      [y,x]
    
    Returns k-curl of the vector quantity (u,v)
    kcurl - Array of the k-curl                          [...,y,x]
    """

    #Roll and mask weighted velocities
    u_e1u_roll = roll_and_mask(u*e1u, -1, axis=yaxis) # =u*e1u[ i ,j+1]
    v_e2v_roll = roll_and_mask(v*e2v, -1, axis=xaxis) # =v*e2v[i+1, j ]
    
    kcurl = (v_e2v_roll - v*e2v - u_e1u_roll + u*e1u)/(e1f*e2f)
    kcurl[kcurl.mask] = 0.
    
    return kcurl
