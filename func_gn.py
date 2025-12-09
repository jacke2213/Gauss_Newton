
import json
import numpy as np
from typing import Tuple


def phi0(x : np.ndarray, t : np.ndarray) -> np.ndarray:
    """
    Dummy function for test purposes.
    Computes    [x[0] + t[0] * x[1]]
                [x[0] + t[1] * x[1]]
                    |       |
                [x[0] + t[n-1] * x[1]].
    """
    val = x[0] + x[1] * t
    return val


def phi1(x : np.ndarray, t : np.ndarray) ->  np.ndarray:
    """
    Computes the function values of
    phi1(x,t) = x1*exp(-x2*t)
    where x = parameters.
    Note that t can be a scalar or a vector. The function np.exp() 
    evaluates the exponential function component-wise.
    The return value can be a scalar or a vector (depending on t).
    """
    val = x[0] * np.exp(-x[1] * t)
    return val


def phi2(x : np.ndarray, t : np.ndarray) ->  np.ndarray:
    """
    Computes the function values of
    phi2(x,t) = x1*exp(-x2*t) + x3*exp(-x4*t)
    where x = parameters.
    Note that t can be a scalar or a vector. The function np.exp() 
    evaluates the exponential function component-wise.
    The return value can be a scalar or a vector (depending on t).
    """
    val = x[0]*np.exp(-x[1]*t) + x[2]*np.exp(-x[3]*t)
    return val


def get_data_json(filename : str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads vectors t and y from prepared json-files.
    Set filename equal to "data1.json" or "data2.json"
    when calling the function 
    (if they are in another folder, add the path).
    """
    try:
        with open(filename, 'r') as openfile:
            json_object = json.load(openfile)
    except FileNotFoundError as e:
        print("It seems like you have passed an invalid filename.")
        print(f"Python's error message: {e}")
        exit()
    
    t = np.asarray(json_object["t"])
    y = np.asarray(json_object["y"])

    return t, y
    