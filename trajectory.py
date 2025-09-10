import numpy as np
import math
import utils

def clean_coords_tuple(x_dist, y_dist):
    
    # Zip and filter out any pair where either element is NaN
    filtered = [(x, y) for x, y in zip(x_dist, y_dist) if not (math.isnan(x) or math.isnan(y))]

    # Unzip the filtered list back into two tuples
    x_clean, y_clean = zip(*filtered) if filtered else ((), ())
    return x_clean, y_clean

def polish_coordinates(dict_cds, theta_trj):
    dict_dists = {}
    for k, v in dict_cds.items():
        print(k)
        pos_in_am = list()

        for i in v:
            x_dist, y_dist            = clean_coords_tuple(*i.distribution_positions)
            x_dist_rot, y_dist_rot, _ = rotate_distribution(x_dist, y_dist, theta = theta_trj)        
            pos_in_am.append((x_dist_rot, y_dist_rot))

        dict_dists[k] = pos_in_am
    return dict_dists


def distribution_coords_normalize(points_distribution, unity, center, rotation_theta):
    '''
    The method rotates, normalizes and recenters a distribution of points.
    Input:
        points_distribution: a list of two tuples, first x coordinates and second y coordinates of a distribution
        of points. 
        unity: float, itrepresents the unite against which normalize. 
        center: tuple of two elements, respectively x and y coordinates of a point. 
                It is used for recentering the distribution.
        rotation theta: np.array with a float element inside, the corrective orientation to apply to the points.
    Output:
        x_dva_rotated, y_dva_rotated: list of float. The coordinates for the distribution of points, normalized and
                                      recentered. 
    Example of usage: 
        x_test, y_test = distance_converter_pixel2dva(distribution_positions, x[0]-x[1], (x[-1], y[-1]), theta_h)
    '''
    # Linearize coordinates
    xs = points_distribution[0]
    ys = points_distribution[1]
    
    x_clean, y_clean = clean_coords_tuple(xs, ys)

    # Rotate distribution according the rotation_theta provided
    x_to_normalize, y_to_normalize, _ = rotate_distribution(x_clean, y_clean, theta = rotation_theta)

    # Normalization of the coordinates for their center and the picked unity
    x_dva_rotated = [(i-center[0])/unity for i in x_to_normalize]
    y_dva_rotated = [(i-center[1])/unity for i in y_to_normalize]    
    
    return x_dva_rotated, y_dva_rotated

def get_angle_distribution(points_distribution, dim_frame):
    '''
    Input:
    points_distribution: list of tuple, each of which contains x and y for each point.
    dim_frame: tuple of two elements, respectively, y and x dimension.
    Output:
    theta_h: np.array of one element: degree in rad
    '''
    # Linearization of x and y coords for each point
    xs = list(list(zip(*points_distribution))[0])
    ys = list(list(zip(*points_distribution))[1])
    # It performs a fitting
    a, b = get_trajectory(xs, ys, (0, dim_frame[1]))
    # Then extraction of slope of the fitting
    theta = get_rad(a, b)
    # Return the detected angle of the distribution -in rad-
    return theta, a, b

def get_cond_names(retino_pos_am):
    cd_am         = list(retino_pos_am.keys())
    cd_pos        = list(set([pos for i in retino_pos_am.values() for pos in i]))
    sorted_cd_pos = utils.cardinal_sort(cd_pos)
    return cd_am, sorted_cd_pos 

def get_direction(single_pos_coords, list_pos):
    tmp  = [single_pos_coords[j] for j in list_pos]
    tmp  = list(list(zip(*tmp))[0])
    sign = get_sign(tmp, 0, -1)
    if sign > 0:
        tmp_dir = 'up'
    else:
        tmp_dir = 'dw'
    return tmp_dir, sign

def get_mask_on_trajectory(dims, xs, ys, radius = 2):
    up = int(np.max(xs))
    bottom = int(np.min(xs))
    x, y = get_trajectory(xs, ys, (bottom, up))
    xs_fit_lins = np.round(np.linspace(bottom,up,(up-bottom)*2))
    masks_small = [utils.sector_mask(dims, (int(round(np.interp(i, x, y))), i), radius, (0,360)) for i in xs_fit_lins]
    small_mask = sum(masks_small)
    small_mask[np.where(small_mask>1)] = 1
    return small_mask

def get_rad(xs, ys):
    '''
    Given two sets of coordinates, -x and y-, it finds the angle of the distribution.
    It returns an np.array with a float value. It works better with fitted distributions
    or set of points.
    '''
    return -(np.arctan2(np.array([ys[-1]-ys[0]]), np.array([xs[-1] - xs[0]])))

def get_sign(xs_real, id_first, id_last):
    return np.sign(xs_real[id_last] - xs_real[id_first])

def get_spacing_dva(dict_metadata_session, sorted_cd_pos):
    
    tmp_am_cds           = list(dict_metadata_session['pos metadata'].keys())
    
    for k in tmp_am_cds:
        tmp_list = dict_metadata_session['pos metadata'][k]['conditions']
        if (sorted_cd_pos[-1] in tmp_list) and (sorted_cd_pos[0] in tmp_list):
            last_id  = tmp_list.index(sorted_cd_pos[-1])
            first_id = tmp_list.index(sorted_cd_pos[0])
            spacing_between_strokes = int(abs((last_id - first_id)*dict_metadata_session['pos metadata'][k]['inter stimulus space']))

    return spacing_between_strokes 

def get_spacing_pixel(point_on_trajectory, stepping):
    xs = list(list(zip(*point_on_trajectory))[0])
    ys = list(list(zip(*point_on_trajectory))[1])
    return (abs((xs[-1] - xs[0])/stepping), abs((ys[-1] - ys[0])/stepping))


def get_trajectory(xs, ys, limits):
    """
    Get a trajectory by fitting a line to data points and resampling it.

    Args:
    xs (array-like): X-coordinates of the data points.
    ys (array-like): Y-coordinates of the data points.
    limits (tuple): A tuple containing the start and end limits for resampling.

    Returns:
    tuple: A tuple containing the resampled X and Y coordinates of the fitted trajectory.
    """
    xs_to_fit =  np.round(np.linspace(limits[0], limits[1]-1, limits[1]-limits[0])) 
    ys_to_fit = np.poly1d(np.polyfit(xs, ys, 1))(np.unique(xs_to_fit))
    #ys_fitted = [int(round(np.interp(i, xs_to_fit, ys_to_fit))) for i in xs_fitted]
    return xs_to_fit, ys_to_fit

def get_trajectory_mask(points_in_space, frame_dimension, extremities = (0,0)):
    '''
    points_in_space: list of tuples with coordinates (x,y).
    frame_dimension: tuple with shape dimension
    extremities: tuple with amount to subtract from the frame_dimension
    '''
    xs = list(list(zip(*points_in_space))[0])
    ys = list(list(zip(*points_in_space))[1])
    a, b = get_trajectory(xs, ys, (0 + extremities[0], frame_dimension[1]- extremities[1]))
    traject_mask = get_mask_on_trajectory(frame_dimension, a, b, radius = 15)
    return traject_mask

def get_unit_n_center(distributions_pos, metadata_conds_dict, list_pos, theta):
    max_distance =  get_spacing_dva(metadata_conds_dict, list_pos)
    print(f'Max distance {max_distance}')
    id_first = 0
    id_last  = len(list_pos)-1

    x1, y1, _ = rotate_distribution(distributions_pos[id_first][0], 
                                        distributions_pos[id_first][1], 
                                        theta = theta)

    x2, y2, _ = rotate_distribution(distributions_pos[id_last][0], 
                                        distributions_pos[id_last][1], 
                                        theta = theta)

    unit   = np.abs(np.nanmedian(x1)-np.nanmedian(x2))/max_distance
    center = (np.nanmedian(x1), np.nanmedian(y1))
    return unit, center

def normalize_distribution(raw_dist, unit, center, theta, ref_dist, flip_sign):
    x, y = distribution_coords_normalize(raw_dist, unit, center, theta)
    x = (np.array(x) - np.nanmedian(ref_dist[0])) * flip_sign
    y = np.array(y) - np.nanmedian(ref_dist[1])
    return [x, y]

def rotate_distribution(xs, ys, theta = None, frame_width = None, frame_height = None):
    
    if theta is None:
        theta = get_rad(xs, ys)

    # subtracting mean from original coordinates and saving result to X_new and Y_new 
    if (frame_width is None) and (frame_height is None):
        frame_center_x = np.nanmean(xs)
        frame_center_y = np.nanmean(ys)
    else:
        frame_center_x = frame_width / 2
        frame_center_y = frame_height / 2

    X_new = xs - frame_center_x
    Y_new = ys - frame_center_y

    # Apply rotation
    X_rot = np.cos(theta) * X_new - np.sin(theta) * Y_new
    Y_rot = np.sin(theta) * X_new + np.cos(theta) * Y_new

    # Translate back to original reference (still around frame center)
    X_final = X_rot + frame_center_x
    Y_final = Y_rot + frame_center_y

    return X_final, Y_final, theta