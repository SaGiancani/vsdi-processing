import numpy as np
import utils

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


def rotate_distribution(xs, ys, theta = None):
    if theta is None:
        theta = get_rad(xs, ys)
    print(theta)
    # subtracting mean from original coordinates and saving result to X_new and Y_new 
    X_new = xs - np.mean(xs)
    Y_new = ys - np.mean(ys)

    X_apu = [np.cos(theta)*i-np.sin(theta)*j for i, j in zip(X_new, Y_new) ]
    Y_apu = [np.sin(theta)*i+np.cos(theta)*j for i, j in zip(X_new, Y_new) ]

    # adding mean back to rotated coordinates
    return X_apu + np.mean(xs), Y_apu + np.mean(ys), theta