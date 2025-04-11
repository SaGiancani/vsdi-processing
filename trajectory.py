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

    # Rotate distribution according the rotation_theta provided
    x_to_normalize, y_to_normalize, _ = rotate_distribution(xs, ys, theta = rotation_theta)

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
    return theta

def get_cond_names(retino_pos_am, ss_label = 'pos'):
    cd_am         = list(retino_pos_am.keys())
    cd_pos        = list(set([pos for i in retino_pos_am.values() for pos in i]))

    tmp           = sorted([int(i.split(ss_label)[1]) for i in cd_pos])
    sorted_cd_pos = [f'{ss_label}{i}' for i in tmp]    
    return cd_am, sorted_cd_pos 

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

def get_spacing_dva(dict_metadata_session, sorted_cd_pos, retino_pos_am, ss_label = 'pos'):
    
    cd_am, sorted_cd_pos = get_cond_names(retino_pos_am, ss_label = ss_label)
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

def normalize_distributions(dict_metadata_session, cond_am, dict_dists, unity, theta_trj):
    
    single_stroke = dict_metadata_session['pos metadata'][cond_am]['conditions']
    print(single_stroke)
    p0    = single_stroke[0]
    pLast = single_stroke[-1]
#     print(f'p0 {p0}: {(np.nanmean(dict_dists[p0][0][0]), np.nanmean(dict_dists[p0][0][1]) )} pLast {pLast}: {(np.nanmean(dict_dists[pLast][0][0]), np.nanmean(dict_dists[pLast][0][1]) )} step {step}')
    
    center = (np.nanmean(dict_dists[pLast][0][0]), np.nanmean(dict_dists[pLast][0][1]))
    sub_cd = [i for i in list(dict_dists.keys()) if f'{cond_am}-' in i][0]
    print(f'Unity {unity} center {center} Sub condition {sub_cd}')

    x_norm_am, y_norm_am   = distribution_coords_normalize([(np.array([dict_dists[cond_am][-1][0]]))[0], dict_dists[cond_am][-1][1]], 
                                                           unity, 
                                                           center, 
                                                           theta_trj)
    x_norm_am  = [i[0] for i in x_norm_am]
    y_norm_am  = [i[0] for i in y_norm_am]

    x_norm, y_norm         = distribution_coords_normalize([(np.array([dict_dists[pLast][-1][0]]))[0], dict_dists[pLast][-1][1]], 
                                                           unity, 
                                                           center, 
                                                           theta_trj)
    x_norm     = [i[0] for i in x_norm]
    y_norm     = [i[0] for i in y_norm]

    x_norm_sub, y_norm_sub = distribution_coords_normalize([(np.array([dict_dists[sub_cd][-1][0]]))[0], dict_dists[sub_cd][-1][1]],
                                                           unity, 
                                                           center, 
                                                           theta_trj)
    x_norm_sub = [i[0] for i in x_norm_sub]
    y_norm_sub = [i[0] for i in y_norm_sub]    
    return (x_norm_am, y_norm_am), (x_norm, y_norm), (x_norm_sub, y_norm_sub)


def rotate_distribution(xs, ys, theta = None):
    if theta is None:
        theta = get_rad(xs, ys)
    # subtracting mean from original coordinates and saving result to X_new and Y_new 
    X_new = xs - np.mean(xs)
    Y_new = ys - np.mean(ys)

    X_apu = [np.cos(theta)*i-np.sin(theta)*j for i, j in zip(X_new, Y_new) ]
    Y_apu = [np.sin(theta)*i+np.cos(theta)*j for i, j in zip(X_new, Y_new) ]

    # adding mean back to rotated coordinates
    return X_apu + np.mean(xs), Y_apu + np.mean(ys), theta