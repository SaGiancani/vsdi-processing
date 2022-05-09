import numpy as np
import scipy.io as scio
import datetime, fnmatch, os, pickle, struct


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

def find_file(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

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

def socket_numpy2matlab(path, matrix, substring = ''):
    '''
    ---------------------------------------------------------------------------------------------------------
    Utility method for converting numpy array into a Matlab structure, with field "signal".
    The method saves a .mat matlab matrix variable, in the path folder, containing the matrix data.
    ---------------------------------------------------------------------------------------------------------    
    '''
    scio.savemat(os.path.join(path, substring+'_signal.mat'), {'signal': matrix})
    return