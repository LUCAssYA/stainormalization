import numpy as np
import cv2
import math
import pandas as pd
import scipy.sparse.linalg as la
from PIL import Image

def conv(x):
    m, n = x.shape
    if m > 1: denom = m -1
    else: denom = m

    xc = x - np.sum(x, axis = 0)/m
    c = np.dot(np.transpose(xc), xc)/denom
    c = c.conjugate()
    return c


def normalizeStaining(I, Io = 240, beta=0.15, alpha = 1, HERef = np.array([[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]]), maxCRef = np.array([1.9705, 1.0308])):
    I = np.array(I, dtype = np.int16)
    h = I.shape[0]
    w = I.shape[1]

    I = np.reshape(I, (-1, 3), order = 'F')

    OD = -np.log((I+1)/Io)

    ODhat = np.array([])

    aa = np.all(OD > beta, axis = 1)
    ODhat = OD[np.where(aa == True)[0], :]
    temp = conv(ODhat)
    _, v = np.linalg.eig(temp)
    temp = -1*v
    v[:, 0], v[:, 1], v[:, 2] = temp[:, 2], temp[:, 1], temp[:, 0]


    that = np.dot(ODhat, v[:, 1:3])
    phi = np.arctan2(that[:, 1], that[:, 0])
    #phi = math.atan2(that[:, 1], that[:, 0])

    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100-alpha)

    vMin = np.dot(v[:, 1:3], np.vstack((np.cos(minPhi), np.sin(minPhi))))
    vMax = np.dot(v[:, 1:3], np.vstack((np.cos(maxPhi), np.sin(maxPhi))))

    if vMin[0] > vMax[0]:
        HE = np.hstack((vMin, vMax))
    else:
        HE = np.hstack((vMax, vMin))

    Y = np.transpose(OD)
    Y = Y.conjugate()

    C = np.linalg.lstsq(HE, Y)[0]

    maxC = np.percentile(C, 99, axis = 1)
    maxC = np.transpose(np.array([maxC]*C.shape[1]))
    C = C/maxC
    maxCRef = np.transpose(np.array([maxCRef]*C.shape[1]))
    C = C*maxCRef

    maxC = None
    maxCRef = None


    Inorm = Io*np.exp(np.dot((-1*HERef), C))
    Inorm = np.array([np.reshape(Inorm[0, :], (h, w), order = 'F'),
                     np.reshape(Inorm[1, :], (h, w), order='F'),
                     np.reshape(Inorm[2, :], (h, w), order='F')])
    Inorm = Inorm.transpose(1, 2, 0)
    Inorm = np.around(Inorm)
    Inorm = np.array(Inorm, dtype = np.uint8)



    return Inorm

I1 = cv2.imread('example1.tif')

norm1 = normalizeStaining(I1)
norm1 = cv2.cvtColor(norm1, cv2.COLOR_RGB2BGR)
cv2.imwrite("aaaa.tif", norm1)