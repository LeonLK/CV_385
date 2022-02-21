import cv2 as cv
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def eight_point(p1, p2, M):
    # dividing each coordinate by M using a transformation matrix T
    # T = np.array([[1./M,0],
    #               [0,1./M]])
    # p1 = p1/M
    # p2 = p2/M
    T = np.array([[1./M,0,0],
                  [0,1./M,0],
                   [0,0,1]])
    # p1 = T @ np.transpose(p1) #2x2 X 2xN = 2xN
    # p2 = T @ np.transpose(p2) #2x2 X 2xN = 2xN
    # p1x = p1[0,:]
    # p1y = p1[1,:]
    # p2x = p2[0,:]
    # p2y = p2[1,:]
    p1x = p1[:,0]
    p1y = p1[:,1]
    p2x = p2[:,0]
    p2y = p2[:,1]
    p1x,p1y,p2x,p2y = p1x/M,p1y/M,p2x/M,p2y/M
    # p1xp2x = p1x*p2x
    # p1xp2y = p1x*p2y
    # p1yp2x = p1y*p2x
    # p1yp2y = p1y*p2y
    # A = []
    # for i in range(p1.shape[1]):
    #     p1x = p1[0,i]
    #     p1y = p1[1,i]
    #     p2x = p2[0,i]
    #     p2y = p2[1,i]
    #     p1xp2x = p1x*p2x
    #     p1xp2y = p1x*p2y
    #     p1yp2x = p1y*p2x
    #     p1yp2y = p1y*p2y
    #     A.append(np.array([p1xp2x, p1xp2y, p1x, p1yp2x, p1yp2y, p1y, p2x, p2y, 1]))
    # A = np.array(A)
    # A = np.transpose(np.vstack((p1xp2x, p1xp2y, p1x, p1yp2x, p1yp2y, p1y, p2x, p2y, np.ones(p1x.shape))))
    A = np.transpose(np.vstack((p2x*p1x, p2x*p1y, p2x, p2y*p1x, p2y*p1y, p2y, p1x, p1y, np.ones(p1x.shape))))
    # print(A.shape)

    _,_,v = np.linalg.svd(A)
    # minIdx = np.unravel_index(np.argmin(s, axis=None), s.shape)
    # s[minIdx] = 0
    f = np.reshape(v[-1],(3,3))
    #Enforce rank 2
    uf,sf,vf = np.linalg.svd(f)
    sf[-1] = 0 #svd returns s in descending order
    sf = np.diag(sf)
    F = uf @ sf @ vf
    F = np.transpose(T) @ F @ T
    return F
