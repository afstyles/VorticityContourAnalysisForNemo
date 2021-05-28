#  _____             _                 _____      _   
# /  __ \           | |               |_   _|    | |  
# | /  \/ ___  _ __ | |_ ___  _   _ _ __| | _ __ | |_ 
# | |    / _ \| '_ \| __/ _ \| | | | '__| || '_ \| __|
# | \__/\ (_) | | | | || (_) | |_| | | _| || | | | |_ 
#  \____/\___/|_| |_|\__\___/ \__,_|_| \___/_| |_|\__|

"""
ContourInt.py

Calculation of closed contours and integrations within them

Contains methods:
    niiler_integral2D --> Integrates all vorticity diagnostics over areas enclosed by contours 
    contour_integral --> Identifies contours and integrates fields over the enclosed areas
    interp_to_finegrid --> Interpolates a field to a fine lon, lat grid
    take_largest_contour --> From outputs of niiler_integral2d, take contours that span the largest area 
                             if there are multiple contours of the same isovalue  
    DepIntSF_orca --> Calculates the depth integrated stream function
"""
import numpy as np
import pickle
import iris
from iris.cube import Cube
from iris.coords import DimCoord
from iris.coords import AuxCoord
from skimage.measure import find_contours
from skimage.measure import grid_points_in_poly
from scipy.interpolate import griddata

def niiler_integral2D(vort_2D_dict, sf_zint_cube, area_weights, nlevels=201, level_min=False, level_max=False, 
                      lonlatbounds=None, interpolation=None, res=1/12.0, R=6400.0e3):
    """
    niiler_integral2D(vort_2D_dict, sf_zint_cube, area_weights, nlevels=201, level_min=False, level_max=False, 
                      lonlatbounds=None, interpolation=None, res=1/12.0, R=6400.0e3):

    Integrates the time-averaged vorticity diagnostics over the area enclosed by the time-averaged, depth-integrated streamlines.
    Return a dictionary of area integrals corresponding to each diagnostic in vort_2D_dict

    INPUT variables
    vort_2D_dict - Dictionary of IRIS cubes describing vorticity diagnostics         {(t,y,x),...} [m/s]
    sf_zint_cube - IRIS cube of the depth-integrate streamlines                       (t,y,x) [Sv]
    area_weights - F cell horizontal areas == e1f*e2f. Only used if interpolation==None (y,x) [m2]
    nlevels - Number of streamline values to integrate within
    level_min - Minimum value of streamline to integrate within.   [Sv]
                Set to False to use maximum of sf_zint_cube
    level_max - Maximum value of streamline to integrate within.   [Sv]
                Set to False to use minimum of sf_zint_cube
    lonlatbounds - Tuple of longitudes and latitudes to describe a rectangular subregion of interest
                   (lon_min, lon_max, lat_min, lat_max)  [deg]
                   Set to None if you wish to use the full region
    interpolation - Method of interpolation to use: 'linear' for linear interpolation
                                                      None   for no interpolation
    res - Resolution to interpolate to in degrees. Not used if interpolation == None.  [deg]   
    R   - Radius of the earth, used to determine interpolated cell areas               [m]

    OUTPUT variables
    cubes_out_dict - Dictionary of IRIS cubes containing area integrations.   {(ncont), ...} [m3/s2]
                     Keys for dictionary entries match those in vort_2D_dict

    contour_masks_out - Array of masks used for each contour integration      (ncont, y, x) [-] 

    """

    tm_sf_zint = np.mean(sf_zint_cube.data, axis=0)
    
    tm_vort_2D_dict = {}
    
    for label in vort_2D_dict:
        vc_zint_cube = vort_2D_dict[label]
        #Time average vorticity components
        if len(vc_zint_cube.shape) == 2:
            tm_vc_zint = vc_zint_cube.data
            tm_vort_2D_dict = {**tm_vort_2D_dict, label:tm_vc_zint}

        else:
            tm_vc_zint = np.mean(vc_zint_cube.data, axis=0)
            tm_vort_2D_dict = {**tm_vort_2D_dict, label:tm_vc_zint}
    
    if interpolation == 'linear':
        print("Interpolating to fine grid")
        lon = sf_zint_cube.coord("longitude").points
        lat = sf_zint_cube.coord("latitude").points

        tm_sf_zint, lon_finegrid, lat_finegrid = interp_to_finegrid(tm_sf_zint, lon, lat, res, lonlatbounds=lonlatbounds)
        
        for label in tm_vort_2D_dict:
            tm_vc_zint = tm_vort_2D_dict[label]
            tm_vc_zint, lon_finegrid, lat_finegrid = interp_to_finegrid(tm_vc_zint, lon, lat, res, lonlatbounds=lonlatbounds)
            tm_vort_2D_dict[label] = tm_vc_zint

        area_weights = (R**2)*((res*(np.pi/180))**2)*np.cos(lat_finegrid*np.pi/180.0)
        
    integrals_out_dict, levels_out, areas_out, mask_out, contour_masks_out = contour_integral( tm_vort_2D_dict, tm_sf_zint,
                                                    area_weights, level_min=level_min, level_max=level_max, nlevels=nlevels)
    
    
    #Save output as IRIS cube >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    sf_level_coord = AuxCoord(levels_out, long_name='sf_zint_level', units=sf_zint_cube.units)
    area_coord = AuxCoord(areas_out, long_name='enclosed_area', units='m2')
    cubes_out_dict = {}
    
    for label in integrals_out_dict:
        output_cube = Cube(integrals_out_dict[label])
        output_cube.long_name = 'NiilerIntegral2D(' + label + ')'
        output_cube.var_name = 'Ni2D_' + label
        output_cube.units = 'm3/s2'
        output_cube.add_aux_coord(sf_level_coord, [0])
        output_cube.add_aux_coord(area_coord, [0])
        
        cubes_out_dict[label] = output_cube
        
    
    contour_masks_out = np.array(contour_masks_out, dtype=bool)
#     Annoyingly IRIS does not allow the saving of boolean data as a cube. A pickle file will have to do    
#     contour_masks_cube = Cube(np.array(contour_masks_out,dtype='bool'))
#     contour_masks_cube.long_name = 'contour_masks'
#     contour_masks_cube.var_name = 'cont_masks'
#     contour_masks_cube.units = 'no_unit'
#     contour_masks_cube.add_aux_coord(sf_level_coord, [0])
            
    return cubes_out_dict, contour_masks_out

#                  _                   _       _                       _ 
#                 | |                 (_)     | |                     | |
#   ___ ___  _ __ | |_ ___  _   _ _ __ _ _ __ | |_ ___  __ _ _ __ __ _| |
#  / __/ _ \| '_ \| __/ _ \| | | | '__| | '_ \| __/ _ \/ _` | '__/ _` | |
# | (_| (_) | | | | || (_) | |_| | |  | | | | | ||  __/ (_| | | | (_| | |
#  \___\___/|_| |_|\__\___/ \__,_|_|  |_|_| |_|\__\___|\__, |_|  \__,_|_|
#                                ______                 __/ |            
#                               |______|               |___/             

def contour_integral(integrand_dict, contour_field, area_weights, level_min=False, level_max=False, nlevels = 101):
    """
    contour_integral(integrand_dict, contour_field, area_weights, level_min=False, level_max=False, nlevels = 101)

    Calculates the area inetgral enclosed withing ncont contours of a field.

    INPUT variables
    integrand_dict - Dictionary of 2D arrays to be area integrated          {(x,y),...}
    contour_field  - Field containing the contours to be integrated within  (x,y)
    area_weights - Areas of each horizontal cell                            (x,y)  [m2]
    level_min - Float, Minimum contour level (defaults to minimum of field)
    level_max - Float, Maximum contour level (defaults to maximum of field)
    nlevels - Integer, number of levels to evaluate (evenly spaced)

    integrals_out_dict, levels_out, areas_out, mask_out, contour_masks_out

    OUTPUT variables
    integrals_out_dict - Dictionary of area integrals for each integrand in integrand_dict  {(ncont),...}
                         The key for each entry matches the key in integrand_dict
                         Each entry in the dictionary is a 1D numpy array of area integrals

    levels_out - 1D array of contour values cooresponding to area integrals                (ncont)
    areas_out - 1D array of areas enclosed by contours                                     (ncont)
    mask_out - Mask for specific contours (e.g. if not a closed contour == True)           (ncont)
    contour_masks_out - List of 2D masks corresponding to eaach area integral carried out  (ncont, y, x)
                        True = Outside of enclosed area 

    """
    from skimage.measure import find_contours
    from skimage.measure import grid_points_in_poly
    import numpy as np


    #If min/max levels are not given, use the min/max of the contour field
    if level_min == False:
        level_min = np.min(contour_field)

    if level_max == False:
        level_max = np.max(contour_field)

    #Define a linear space of levels
    levels = np.linspace(level_min, level_max, num=nlevels)
    
    #Create empty output objects
    integrals_out_dict = {}
    
    for label in integrand_dict:
        integrals_out_dict[label] = []

    contours_out = []
    levels_out = []
    areas_out = []
    mask_out = []
    contour_masks_out = []
    
    for level in levels:

        #For the given level identify the coordinates of all matching contours
        contours = find_contours(contour_field.data, level, mask=~contour_field.mask )

        #Count the number of distinct contours
        ncontours = len(contours)

        #Test if contours exist
        if ncontours == 0:
            #If not, move to the next level
            continue
        
        for contour in contours:
            
            contours_out = contours_out + [contour]
            levels_out = levels_out + [level]
            
            #If contour is not closed continue to next contour for this level
            if not np.allclose(contour[0,:],contour[-1,:]):
                areas_out = areas_out + [-999999]
                mask_out = mask_out + [True]
                contour_masks_out = contour_masks_out + [np.ones(contour_field.shape, dtype=bool)]
                
                continue
            

            mask_out = mask_out + [False]
            
            #Define contour_mask that masks all variables outside of the closed area
            contour_mask =  ~grid_points_in_poly(contour_field.shape, contour)
            contour_masks_out = contour_masks_out + [contour_mask]
            
            #Determine enclosed area
            enclosed_area = np.sum((~contour_mask)*area_weights)
            areas_out = areas_out + [enclosed_area]


    #Now we have the contour information, we carry out the integrals over these common contours                
    #For each integrand in the dictionary
    for label in integrand_dict:
        
        print(f"Integrating {label}")
        
        integrand = integrand_dict[label]
        integrals_out = []
        
        i = 0
        
        for mask in contour_masks_out:
            
            if mask_out[i] == True:
                #If contour is masked, add fill value and move on
                integrals_out = integrals_out + [-999999]
                i = i+1
                continue

            #Add contour_mask to integrand mask
            mask_tmp = np.ma.mask_or(mask, integrand.mask)     

            #Temporarily mask the integrand
            integrand_tmp = np.ma.masked_array(integrand, mask=mask_tmp)

            #Carry out area integral of newly masked integrands
            integral = np.sum( integrand_tmp * area_weights )     
            integrals_out = integrals_out + [integral]
            
            i = i+1
        
        #Save the list of integrals
        integrals_out = np.ma.masked_array(integrals_out, mask = mask_out)
        
        #Save into output dictionary
        integrals_out_dict[label] = integrals_out
        
    return integrals_out_dict, levels_out, areas_out, mask_out, contour_masks_out

#  _       _                  _          __ _                       _     _ 
# (_)     | |                | |        / _(_)                     (_)   | |
#  _ _ __ | |_ ___ _ __ _ __ | |_ ___  | |_ _ _ __   ___  __ _ _ __ _  __| |
# | | '_ \| __/ _ \ '__| '_ \| __/ _ \ |  _| | '_ \ / _ \/ _` | '__| |/ _` |
# | | | | | ||  __/ |  | |_) | || (_) || | | | | | |  __/ (_| | |  | | (_| |
# |_|_| |_|\__\___|_|  | .__/ \__\___/ |_| |_|_| |_|\___|\__, |_|  |_|\__,_|
#                      | |______   ______                 __/ |             
#                      |_|______| |______|               |___/              

def interp_to_finegrid(array, lon, lat, res, lonlatbounds=None):
    """
    interp_to_finegrid(array, lon, lat, res, lonlatbounds=None):

    Interpolate an array to a fine regular grid. This is used so that identified contours are smoother.
    
    array   - Array to be interpolated (Masked numpy array)                          [...,x,y]
    lon,lat - Array of longitude/latitude                                                [x,y]
    res - Desired angular resolution (float)
    lonlatbounds - Tuple, optional bounds of area of interest = (lon_min, lon_max, lat_min, lat_max)
    
    Returns
    output_values - Array linearly interpolated on to a fine regular grid           [...,lon,lat]
    lon_finegrid, lat_finegrid - lon/lat values of the new fine grid                    [lon,lat]
    """
    import numpy as np
    from scipy.interpolate import griddata
    
    lon_datagrid = np.broadcast_to(lon, array.shape)
    lat_datagrid = np.broadcast_to(lat, array.shape)
    
    #Mask variables outside of bounds if specified
    if isinstance(lonlatbounds, tuple):
        lon_min = lonlatbounds[0]
        lon_max = lonlatbounds[1]
        lat_min = lonlatbounds[2]
        lat_max = lonlatbounds[3]
        
        extramask = (lon < lon_min) + (lon > lon_max) + (lat < lat_min) + (lat > lat_max)
        extramask = np.broadcast_to(extramask, array.shape)
        new_mask = np.ma.mask_or(array.mask, extramask)
        array = np.ma.masked_array(array, mask=new_mask)
        
    else:
        lon_min = np.min(lon)
        lon_max = np.max(lon)
        lat_min = np.min(lat)
        lat_max = np.max(lat)
        
        extramask = np.zeros(array.shape, dtype=bool)
    
    #Construct fine grid to interpolate on to
    lon_finecoord = np.linspace( lon_min , lon_max, num=int((lon_max-lon_min)/res))
    lat_finecoord = np.linspace( lat_min , lat_max, num=int((lat_max-lat_min)/res))
    
    lon_finegrid, lat_finegrid = np.meshgrid(lon_finecoord, lat_finecoord)
    
    #Interpolate the data values using scipy griddata for interpolating to gridded data
    points = (lon_datagrid[~array.mask], lat_datagrid[~array.mask])
    values = array[~array.mask]
    interp = griddata(points, values, (lon_finegrid, lat_finegrid), method = 'linear')
    interp = np.ma.masked_invalid(interp)
    
    #Interpolate the mask so that any non-zero values implies the value ought to be masked
    mask_points = (lon_datagrid[~extramask], lat_datagrid[~extramask])
    mask_values = array.mask[~extramask]
    interp_mask = griddata(mask_points, mask_values, (lon_finegrid, lat_finegrid), method='linear')

    #Add the interpolated mask to the already existing one
    final_mask = interp.mask + interp_mask
    output_values = np.ma.masked_array(interp, mask=final_mask)
    
    return output_values, lon_finegrid, lat_finegrid

def DepIntSF_orca(u_cube, e1u, e2u, e3u, e1f):
    """
    DepIntSF_orca(u_cube, e1u, e2u, e3u, e1f)

    Calculuates the depth-integrated stream function, making no assumptions about the regularity of the grid. The function integrates
    northwards to calculuate the stream function.

    INPUT variables
    u_cube   - IRIS cube of x-velocities                  (t,z,y,x)  [m/s]
    e1u, e2u -  Array of u cell widths                        (y,x)  [m]
    e3u      - Array of u cell thicknesses                (t,z,y,x)  [m]
    e1f      - Array of f cell x width                        (y,x)  [m]
    
    Returns depth-integrated stream function centred on f points of the C grid as an IRIS cube
    sf_zint_cube - IRIS cube of depth-integrated stream function  (t,y,x)  [Sv] 
    """
       
    u_zint = np.sum(u_cube.data*e3u, axis=1)
    
    e1u = np.broadcast_to(e1u, u_zint.shape)
    e2u = np.broadcast_to(e2u, u_zint.shape)
    e1f = np.broadcast_to(e1f, u_zint.shape)

    integrand = -u_zint*e2u

    sf_zint = np.cumsum(integrand, axis=-2)
    
    #swap axis order termporarily for easier broadcasting (put y axis at start)
    sf_zint = np.swapaxes(sf_zint, -2, 0)
    
    sf_zint = sf_zint - np.broadcast_to(sf_zint[0,...],sf_zint.shape)
    
    #swap axis order back to original state
    sf_zint = np.swapaxes(sf_zint,-2, 0)
    
    #Save stream function as an iris cube
    time = u_cube.coord("time")
    
    sf_zint_cube = Cube(sf_zint/(10**6), dim_coords_and_dims=[(time,0)])
    
    sf_zint_cube.standard_name = 'ocean_barotropic_streamfunction'
    sf_zint_cube.long_name = 'Depth integrated stream function'
    sf_zint_cube.var_name = 'sf_zint'
    sf_zint_cube.units = 'Sv'
    sf_zint_cube.description = 'ocean streamfunction variables'
    
    lat = u_cube.coord("latitude")
    lon = u_cube.coord("longitude")
    
    sf_zint_cube.add_aux_coord(lat, [1,2])
    sf_zint_cube.add_aux_coord(lon, [1,2])
    
    return sf_zint_cube

def NI_ADV_calc(Ni_keg_cube, Ni_rvo_cube, Ni_zad_cube):
    """
    Combine contour integrals of the following diagnostics to calculate the total advective contribution
    """
    
    from iris.cube import Cube
    
    sf_level_coord = Ni_keg_cube.coord("sf_zint_level")
    area_coord = Ni_keg_cube.coord("enclosed_area")
    
    Ni_adv_cube = Cube(Ni_keg_cube.data + Ni_rvo_cube.data + Ni_zad_cube.data)
    Ni_adv_cube.long_name = 'NiilerIntegral2D(ADV)'
    Ni_adv_cube.var_name = 'Ni2D_ADV'
    Ni_adv_cube.units = 'm3/s2'
    Ni_adv_cube.add_aux_coord(sf_level_coord, [0])
    Ni_adv_cube.add_aux_coord(area_coord, [0])
    
    return Ni_adv_cube

def NI_ZDF_calc(Ni_wnd_cube, Ni_frc_cube):
    """
    Combine contour integrals of the following diagnostics to calculate the vertical diffusion contribution
    """
    
    from iris.cube import Cube
    
    sf_level_coord = Ni_wnd_cube.coord("sf_zint_level")
    area_coord = Ni_wnd_cube.coord("enclosed_area")
    
    Ni_zdf_cube = Cube(Ni_wnd_cube.data + Ni_frc_cube.data)
    Ni_zdf_cube.long_name = 'NiilerIntegral2D(ZDF)'
    Ni_zdf_cube.var_name = 'Ni2D_ZDF'
    Ni_zdf_cube.units = 'm3/s2'
    Ni_zdf_cube.add_aux_coord(sf_level_coord, [0])
    Ni_zdf_cube.add_aux_coord(area_coord, [0])
    
    return Ni_zdf_cube
    
def NI_RES_calc(Ni_adv_cube, Ni_pvo_cube, Ni_hpg_cube, Ni_ldf_cube, Ni_zdf_cube, Ni_tot_cube):
    """
    Combine contour integrals of the following diagnostics to calculate the residual contribution
    """
    
    from iris.cube import Cube
    
    sf_level_coord = Ni_adv_cube.coord("sf_zint_level")
    area_coord = Ni_adv_cube.coord("enclosed_area")
    
    Ni_res_cube = Cube(Ni_adv_cube.data + Ni_pvo_cube.data + Ni_hpg_cube.data
                       + Ni_ldf_cube.data + Ni_zdf_cube.data - Ni_tot_cube.data)
    
    Ni_res_cube.long_name = 'NiilerIntegral2D(RES)'
    Ni_res_cube.var_name = 'Ni2D_RES'
    Ni_res_cube.units = 'm3/s2'
    Ni_res_cube.add_aux_coord(sf_level_coord, [0])
    Ni_res_cube.add_aux_coord(area_coord, [0])
    
    return Ni_res_cube

def take_largest_contour(int_out, level_out, area_out):
    
    """
    Function that looks at outputs from contour_integral and filters any contours that aren't that largest at a given value
    
    int_out - 1D Array of integrated values
    level_out - 1D Array of contour levels  
    area_out - 1D Array of enclosed contour area
    
    returns output_value, output_level
    """

    import numpy as np
    
    levels_unique = np.unique(level_out)
    
    output_value = []
    output_level = []
    
    for level in levels_unique:
        values = int_out[(level_out == level)]
        areas = area_out[(level_out == level)]
        large_area_value = values[ areas == np.max(areas) ]
        
        if not np.ma.is_masked(large_area_value):
            output_value = output_value + [ np.mean(large_area_value) ]
            output_level = output_level + [level]
        
    output_value = np.squeeze(np.array(output_value))
    output_level = np.squeeze(np.array(output_level))
    
    return output_value, output_level
