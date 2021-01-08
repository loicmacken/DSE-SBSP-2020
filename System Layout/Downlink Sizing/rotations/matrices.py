import numpy as np

def rotx(degrees):
    cos = np.cos(np.radians(degrees))
    sin = np.sin(np.radians(degrees))
    Rx = np.mat([[1,0,0],[0,cos,-sin],[0,sin,cos]])
    return Rx

def roty(degrees):
    cos = np.cos(np.radians(degrees))
    sin = np.sin(np.radians(degrees))
    Ry = np.mat([[cos,0,sin],[0,1,0],[-sin,0,cos]])
    return Ry

def rotz(degrees):
    cos = np.cos(np.radians(degrees))
    sin = np.sin(np.radians(degrees))
    Rz = np.mat([[cos,-sin,0],[sin,cos,0],[0,0,1]])
    return Rz

def unit(vec):
    unitvec = np.array(vec)/np.sqrt(sum(np.array(vec)**2))
    return unitvec