import numpy as np

def A_x(degree: float):
    radian = np.radians(degree) # = (degree / 180) * np.pi;
    return np.array([ [1,              0,               0],
                      [0, np.cos(radian), -np.sin(radian)],
                      [0, np.sin(radian),  np.cos(radian)]])

def A_y(degree: float):
    radian = np.radians(degree) # = (degree / 180) * np.pi;
    return np.array([ [np.cos(radian),  0, np.sin(radian)],
                      [0,               1, 0             ],
                      [-np.sin(radian), 0, np.cos(radian)]])

def A_z(degree: float):
    radian = np.radians(degree) # = (degree / 180) * np.pi;
    return np.array([ [np.cos(radian), -np.sin(radian), 0],
                      [np.sin(radian),  np.cos(radian), 0],
                      [0,               0,              1]])