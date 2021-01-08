import numpy as np

def rotx(radians):
    cos = np.cos(radians)
    cos = cos * 0**(abs(cos)<10**-16)
    sin = np.sin(radians)
    sin = sin * 0**(abs(sin)<10**-16)
    Rx = np.mat([[1,0,0],[0,cos,-sin],[0,sin,cos]])
    return Rx

def roty(radians):
    cos = np.cos(radians)
    cos = cos * 0**(abs(cos)<10**-16)
    sin = np.sin(radians)
    sin = sin * 0**(abs(sin)<10**-16)
    Ry = np.mat([[cos,0,sin],[0,1,0],[-sin,0,cos]])
    return Ry

def rotz(radians):
    cos = np.cos(radians)
    cos = cos * 0**(abs(cos)<10**-16)
    sin = np.sin(radians)
    sin = sin * 0**(abs(sin)<10**-16)
    Rz = np.mat([[cos,-sin,0],[sin,cos,0],[0,0,1]])
    return Rz

def unit(vec):
    unitvec = np.array(vec)/np.sqrt(sum(np.array(vec)**2))
    return unitvec