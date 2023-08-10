import cv2 as cv
import numpy as np
import scipy.io as scio
import datetime, fnmatch, logging, os, pickle, sys, struct

from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import datetime


cm_data = [[0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905], 
 [0.212252381, 0.2137714286, 0.6269714286], [0.2081, 0.2386, 0.6770857143], 
 [0.1959047619, 0.2644571429, 0.7279], [0.1707285714, 0.2919380952, 
  0.779247619], [0.1252714286, 0.3242428571, 0.8302714286], 
 [0.0591333333, 0.3598333333, 0.8683333333], [0.0116952381, 0.3875095238, 
  0.8819571429], [0.0059571429, 0.4086142857, 0.8828428571], 
 [0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571, 
  0.8719571429], [0.0498142857, 0.4585714286, 0.8640571429], 
 [0.0629333333, 0.4736904762, 0.8554380952], [0.0722666667, 0.4886666667, 
  0.8467], [0.0779428571, 0.5039857143, 0.8383714286], 
 [0.079347619, 0.5200238095, 0.8311809524], [0.0749428571, 0.5375428571, 
  0.8262714286], [0.0640571429, 0.5569857143, 0.8239571429], 
 [0.0487714286, 0.5772238095, 0.8228285714], [0.0343428571, 0.5965809524, 
  0.819852381], [0.0265, 0.6137, 0.8135], [0.0238904762, 0.6286619048, 
  0.8037619048], [0.0230904762, 0.6417857143, 0.7912666667], 
 [0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381, 
  0.7607190476], [0.0383714286, 0.6742714286, 0.743552381], 
 [0.0589714286, 0.6837571429, 0.7253857143], 
 [0.0843, 0.6928333333, 0.7061666667], [0.1132952381, 0.7015, 0.6858571429], 
 [0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429, 
  0.6424333333], [0.2178285714, 0.7250428571, 0.6192619048], 
 [0.2586428571, 0.7317142857, 0.5954285714], [0.3021714286, 0.7376047619, 
  0.5711857143], [0.3481666667, 0.7424333333, 0.5472666667], 
 [0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524, 
  0.5033142857], [0.4871238095, 0.7490619048, 0.4839761905], 
 [0.5300285714, 0.7491142857, 0.4661142857], [0.5708571429, 0.7485190476, 
  0.4493904762], [0.609852381, 0.7473142857, 0.4336857143], 
 [0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333], 
 [0.7184095238, 0.7411333333, 0.3904761905], 
 [0.7524857143, 0.7384, 0.3768142857], [0.7858428571, 0.7355666667, 
  0.3632714286], [0.8185047619, 0.7327333333, 0.3497904762], 
 [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217], 
 [0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857, 
  0.2886428571], [0.9738952381, 0.7313952381, 0.266647619], 
 [0.9937714286, 0.7454571429, 0.240347619], [0.9990428571, 0.7653142857, 
  0.2164142857], [0.9955333333, 0.7860571429, 0.196652381], 
 [0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857], 
 [0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309], 
 [0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333, 
  0.0948380952], [0.9661, 0.9514428571, 0.0755333333], 
 [0.9763, 0.9831, 0.0538]]

PARULA_MAP = LinearSegmentedColormap.from_list('parula', cm_data)

COLORS = colors_a = [  'forestgreen', 'purple', 'orange', 'blue',
                 'aqua', 'plum', 'tomato', 'lightslategray', 'orangered','gainsboro',
                 'yellowgreen', 'aliceblue', 'mediumvioletred', 'gold', 'sandybrown',
                 'aquamarine', 'black','lime', 'pink', 'limegreen', 'royalblue','yellow']

def nonlinear_map():
# nonlincolormap, colormap for displaying suppression/facilitation in a
# matrix 
# fredo 2011
# Salvatore Giancani 2023
    m = len(cm_data)
    n = int(np.ceil(m/6))

    u=np.linspace(0,n-1,n)/(n-1)

    o=int(round(n/2))
    v=u[(n-o+1):n]

    J = np.zeros((m,3))
    tmp = [0]*o + [0]*n + list(u) + [1]*n + [1]*n + list(np.flip(v))
    J[:len(tmp),0] = tmp  #  %r
    tmp = [0]*o + list(u) + [1]*n + [1]*n + list(np.flip(u)) + [0]*o  
    J[:len(tmp),1] = tmp  #  %r
    tmp = list(v) +[1]*n + [1]*n + list(np.flip(u)) + [0]*n + [0]*o   
    J[:len(tmp),2] = tmp  
    return J

class DrawLineWidget(object):
    def __init__(self, image):
        self.drawing = False
        self.original_image = image
        self.clone = self.original_image.copy()
        self.mask = np.zeros(np.shape(self.clone))

        cv.namedWindow('image')
        cv.setMouseCallback('image', self.extract_coordinates)

        # List to store start/end points
        self.image_coordinates = []

    def extract_coordinates(self, event, x, y, flags, parameters):
        # Record starting (x,y) coordinates on left mouse button click
        if event == cv.EVENT_LBUTTONDOWN:
            self.drawing=True
            self.image_coordinates = [(x,y)]

        elif event==cv.EVENT_MOUSEMOVE:
            if self.drawing==True:
                cv.line(self.clone, self.image_coordinates[-1] ,(x,y),color=(255,255,0),thickness=2)
                cv.line(self.mask, self.image_coordinates[-1] ,(x,y),color=(255,0,255),thickness=3)
                self.image_coordinates = [(x,y)]

        # Record ending (x,y) coordintes on left mouse bottom release
        elif event == cv.EVENT_LBUTTONUP:            
            self.drawing=False
            cv.line(self.clone, self.image_coordinates[-1],(x,y),color=(255,255,0),thickness=2) 
            cv.line(self.mask, self.image_coordinates[-1],(x,y),color=(255,0,255),thickness=3) 
            self.image_coordinates.append((x,y))
            cv.imshow("image", self.clone) 

        # Clear drawing boxes on right mouse button click
        elif event == cv.EVENT_RBUTTONDOWN:
            self.clone = self.original_image.copy()

    def show_image(self):
        return self.clone



def detrending(signal):
    """
    Alternative method to scipy.signal.detrending. 
    It is computed fitting a second order polynom 
    ----------
    Parameter
    ----------
    signal: numpy.array 1D
    Returns
    ----------
    dtrnd_signal: numpy.array 1D. The detrended signal
    """
    x = np.linspace(0, 1, len(signal))
    signal = np.nan_to_num(signal)# for avoiding numpy.linalg.LinAlgError raising
    coeff = np.polyfit(x, signal, 2) #creation of a polynom 2-order
    trend = np.polyval(coeff, x)
    dtrnd_signal = signal - trend
    return dtrnd_signal

def datetime_as_string(raw_bytes):
    tup = struct.unpack("<2l", raw_bytes[:8])
    print(tup)
    days_since_1900 = tup[0]
    print(datetime.datetime.fromtimestamp(days_since_1900 / 1e3))
    partial_day = round(tup[1] / 300.0, 3)
    print(datetime.datetime.fromtimestamp(partial_day))
    #date_time = datetime.datetime(1900, 1, 1) + datetime.timedelta(days=days_since_1900) + datetime.timedelta(seconds=partial_day)
    date_time_ = datetime.timedelta(days=days_since_1900) + datetime.timedelta(seconds=partial_day)
    date_time_ = datetime.datetime.strptime(str(date_time_), '%Y-%m-%d %H:%M:%S.%f')
    print(date_time_)
    return date_time_
    #date_time_.strftime([:23]

def find_thing(pattern, path, what ='file'):
    result = []
    for root, dirs, files in os.walk(path):
        if what == 'file':
            for name in files:
                if fnmatch.fnmatch(name, pattern):
                    result.append(os.path.join(root, name))
        elif what == 'dir':
            for name in dirs:
                if fnmatch.fnmatch(name, pattern):
                    result.append(os.path.join(root, name))           
    return result

def get_sessions(path_storage, exp_type = ['VSDI'], sessions = None, subs = None, experiments = None):
    
    if sessions is not None:
        sessions = [s.lower() for s in sessions]
    elif subs is not None:
        subs = [s.lower() for s in subs]
    elif experiments is not None:
        experiments = [s.lower() for s in experiments]

    if (experiments is None) and (exp_type is not None):
        if 'VSDI' in exp_type and 'BEHAV' in exp_type:
            ending_string = "BEHAV+VSDI"
        elif 'VSDI' in exp_type and 'BEHAV' not in exp_type:
            ending_string = "_VSDI"
        elif 'BEHAV' in exp_type and 'VSDI' not in exp_type:
            ending_string = "_BEHAV"
        elif 'IOI' in exp_type:
            ending_string = "_IOI"
        elif 'ALL' in exp_type:
            ending_string = ""
        exps_list = [f.name for f in os.scandir(path_storage) if (f.name.endswith(ending_string))]
    
    elif (experiments is not None):
        exps_list = list()
        for exp in experiments:
            for f in os.scandir(path_storage):
                if exp.lower() in f.name.lower():
                    exps_list.append(f.name)

    exps = dict()
    for exp in exps_list:
        print(exp)
        exps[exp.split('exp-')[1]] = dict()
        path_exp = os.path.join(path_storage, exp)
        #subjs_list = [f.name for f in os.scandir(path_exp) if (subs is not None) and (f.name.split('sub-')[1].lower() in subs)]
        subjs_list = list()
        for f in  os.scandir(path_exp):
            try:
                if (subs is not None) and (f.name.split('sub-')[1].lower() in subs):
                    subjs_list.append(f.name)
                elif (subs is None):
                    subjs_list.append(f.name)        #print(subjs_list)
            except:
                pass
        for sub in subjs_list:
            print(sub)
            #exps[exp.split('exp-')[1]][sub.split('sub-')[1]] = dict() 
            path_sub = os.path.join(path_exp, sub)
            sess_list = list()
            for f in  os.scandir(path_sub):
                if (sessions is not None) and (f.name.split('sess-')[1].lower() in sessions):
                    sess_list.append(f.name)
                elif (sessions is None):
                    sess_list.append(f.name)
            #print(sess_list)
            paths = list()
            for sess in sess_list:  
                print(sess)
                path_sess = os.path.join(path_sub, sess)
                paths.append(path_sess)
            print(sub.split('sub-')) 
            exps[exp.split('exp-')[1]][sub.split('sub-')[1]] = paths

    return exps
    
def inputs_load(filename):
    '''
    ---------------------------------------------------------------------------------------------------------
    The method allows to load pickle extension files, preserving python data_structure formats
    ---------------------------------------------------------------------------------------------------------
    '''
    a = datetime.datetime.now().replace(microsecond=0)
    with open(filename + '.pickle', 'rb') as f:
        t = pickle.load(f)
        print(datetime.datetime.now().replace(microsecond=0)-a)
        return t    
    
def inputs_save(inputs, filename):
    '''
    ---------------------------------------------------------------------------------------------------------
    The method allows to save python data_structure preserving formats
    ---------------------------------------------------------------------------------------------------------
    '''
    with open(filename+'.pickle', 'wb') as f:
        pickle.dump(inputs, f, pickle.HIGHEST_PROTOCOL)

def sector_mask(shape,centre,radius,angle_range):
    """
    From: https://stackoverflow.com/questions/18352973/mask-a-circular-sector-in-a-numpy-array
    Return a boolean mask for a circular sector. The start/stop angles in  
    `angle_range` should be given in clockwise order.
    angle_range: tuple of integers: it has to give the angular coordinates of the cord of interest
    """

    x,y = np.ogrid[:shape[0],:shape[1]]
    cx,cy = centre
    tmin,tmax = np.deg2rad(angle_range)

    # ensure stop angle > start angle
    if tmax < tmin:
            tmax += 2*np.pi

    # convert cartesian --> polar coordinates
    r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
    theta = np.arctan2(x-cx,y-cy) - tmin

    # wrap angles between 0 and 2*pi
    theta %= (2*np.pi)

    # circular mask
    circmask = r2 <= radius*radius

    # angular mask
    anglemask = theta <= (tmax-tmin)

    return circmask*anglemask

def socket_numpy2matlab(path, matrix, substring = ''):
    '''
    ---------------------------------------------------------------------------------------------------------
    Utility method for converting numpy array into a Matlab structure, with field "signal".
    The method saves a .mat matlab matrix variable, in the path folder, containing the matrix data.
    ---------------------------------------------------------------------------------------------------------    
    '''
    scio.savemat(os.path.join(path, substring+'_signal.mat'), {'signal': matrix}, do_compression=True)
    return

def setup_custom_logger(name):
    '''
    -------------------------------------------------------------------------------------------------------------
    Logger for printing and debugging
    
    It is used for log files for background processes.
    -------------------------------------------------------------------------------------------------------------
    '''
    PATH_LOGS = r'./logs'
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    tmp = str(datetime.datetime.now().replace(microsecond=0))
    lista = tmp.split(':')
    second = lista[1]+lista[2]
    lista = lista[0].split('-')
    first = lista[0]+lista[1]
    data = first + lista[2].split(' ')[0] + lista[2].split(' ')[1] + second
    handler = logging.FileHandler(os.path.join(PATH_LOGS, 'log_'+ data +'.txt'), mode='w')
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(screen_handler)
    return logger