"""
Homework 5
Submission Functions
"""

# import packages here
import cv2 as cv
import numpy as np
from numpy.core.numeric import cross
from numpy.core.shape_base import vstack
import numpy.linalg as la
import matplotlib.pyplot as plt
import helper
import scipy
import scipy.ndimage as ndimage
from scipy import signal
from scipy import misc

"""
Q3.1.1 Eight Point Algorithm
       [I] pts1, points in image 1 (Nx2 matrix)
           pts2, points in image 2 (Nx2 matrix)
           M, scalar value computed as max(H1,W1)
       [O] F, the fundamental matrix (3x3 matrix)
"""
def eight_point(p1, p2, M):
    # dividing each coordinate by M using a transformation matrix T
    T = np.array([[1./M,0,0],
                  [0,1./M,0],
                   [0,0,1]])
    p1 = p1/M
    p2 = p2/M
    p1x = p1[:,0]
    p1y = p1[:,1]
    p2x = p2[:,0]
    p2y = p2[:,1]
    # p1x,p1y,p2x,p2y = p1x/M,p1y/M,p2x/M,p2y/M
    A = np.transpose(np.vstack((p2x*p1x, p2x*p1y, p2x, p2y*p1x, p2y*p1y, p2y, p1x, p1y, np.ones(p1x.shape))))
    # print(A.shape)
    _,_,v = np.linalg.svd(A)
    f = np.reshape(v[-1,:],(3,3))
    # f = helper.refineF(f,p1/M,p2/M)
    f = helper.refineF(f,p1,p2) #this
    #Enforce rank 2
    F = helper._singularize(f)
    # F = helper.refineF(F,p1,p2)
    #Un-Normalize
    F = np.transpose(T) @ F @ T
    return F
  

"""
Q3.1.2 Epipolar Correspondences
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           F, fundamental matrix from image 1 to image 2 (3x3 matrix)
           pts1, points in image 1 (Nx2 matrix)
       [O] pts2, points in image 2 (Nx2 matrix)
"""
def epipolar_correspondences(im1, im2, F, pts1):
    sigma = 20
    im1_blurr = ndimage.gaussian_filter(im1,sigma=sigma,output=np.float64)
    im2_blurr = ndimage.gaussian_filter(im2,sigma=sigma,output=np.float64)
    # im1_blurr = im1
    # im2_blurr = im2
    # im1_blurr = im1.copy()
    # im2_blurr = im2.copy()
    delta = 30
    patchSize = 30
    pts2 = np.zeros((pts1.shape[0:2]))
    h,w = im2_blurr.shape[0:2]
    # print(pts2.shakpe)
    for i in range(0,pts1.shape[0]):
    # for i in range(0,):
        x1,y1 = pts1[i,0],pts1[i,1]
        p1 = np.array([[x1],[y1],[1]])
        line2 = F @ p1
        a,b,c = line2[0],line2[1],line2[2]
        # y2_range = np.arange(y1 - delta, y1 + delta)
        y2_range = np.arange(0,h)
        # print('y2_range:\n',y2_range)
        x2_range = (-(b*y2_range+c)/a) #Both x2_r and y2)r have size delta*2
        x2_range = x2_range.astype(np.int64)
        # inRangeX = (x2_range >= patchSize) & (x2_range < w-patchSize)
        # inRangeY = (y2_range >= patchSize) & (y2_range < h-patchSize)
        
        patch1 = im1_blurr[y1 - patchSize : y1 + patchSize + 1, x1 - patchSize : x1 + patchSize + 1]
        min_dist = np.inf
        # print('patch1 SHape:\n',patch1.shape)
        for j in range(0,h):
            y2 = y2_range[j]
            x2 = x2_range[j]
            if (x2 <= patchSize or x2 >= w - patchSize):
                continue
            elif (y2 <= patchSize or y2 >= h - patchSize):
                continue
            # print(x2,y2) ##FIX USE THE VALID ONE 
            patch2 = im2_blurr[y2 - patchSize : y2 + patchSize + 1, x2 - patchSize : x2 + patchSize + 1]
            # print('patch2 SHape:\n',patch2.shape)
            curr_dist = np.sum(np.abs(patch1-patch2)**2*sigma)
            if (curr_dist < min_dist):
                min_dist = curr_dist
                pts2[i,0] = x2
                pts2[i,1] = y2
    return pts2

"""
Q3.1.3 Essential Matrix
       [I] F, the fundamental matrix (3x3 matrix)
           K1, camera matrix 1 (3x3 matrix)
           K2, camera matrix 2 (3x3 matrix)
       [O] E, the essential matrix (3x3 matrix)
"""
def essential_matrix(F, K1, K2):
    # replace pass by your implementation
    # E = np.dot(np.transpose(K2), np.dot(F, K1))
    E = np.transpose(K2) @ F @ K1
    return E


"""
Q3.1.4 Triangulation
       [I] P1, camera projection matrix 1 (3x4 matrix)
           pts1, points in image 1 (Nx2 matrix)
           P2, camera projection matrix 2 (3x4 matrix)
           pts2, points in image 2 (Nx2 matrix)
       [O] pts3d, 3D points in space (Nx3 matrix)
"""
def triangulate(P1, pts1, P2, pts2):
    x1,y1 = pts1[:,0],pts1[:,1]
    x2,y2 = pts2[:,0],pts2[:,1]
    N = pts1.shape[0]
    pts3d = np.zeros((N,3))
    p1 = P1[0,:]
    p2 = P1[1,:]
    p3 = P1[2,:]
    q1 = P2[0,:]
    q2 = P2[1,:]
    q3 = P2[2,:]

    A1 = np.transpose(np.vstack((y1*p3[0]-p2[0], y1*p3[1]-p2[1], y1*p3[2]-p2[2], y1*p3[3]-p2[3])))
    A2 = np.transpose(np.vstack((p1[0]-x1*p3[0], p1[1]-x1*p3[1], p1[2]-x1*p3[2], p1[3]-x1*p3[3])))
    A3 = np.transpose(np.vstack((y2*q3[0]-q2[0], y2*q3[1]-q2[1], y2*q3[2]-q2[2], y2*q3[3]-q2[3])))
    A4 = np.transpose(np.vstack((q1[0]-x2*q3[0], q1[1]-x2*q3[1], q1[2]-x2*q3[2], q1[3]-x2*q3[3])))
    print('N = ',N)
    for i in range(N):
        #For each p1,p2 pair, get the 4x4 A matrix
        A = np.vstack((A1[i,:],A2[i,:],A3[i,:],A4[i,:]))
        #SVD!!
        _,_,v = np.linalg.svd(A)
        p3d = v[-1,:]
        s = p3d[-1]
        pts3d[i,:] = p3d[0:3]/s
    P_H = np.hstack((pts3d, np.ones((N, 1))))
    print('P_H Shape: ',P_H.shape)
    err = []
    for i in range(N):
        proj1 = np.dot(P1, np.transpose(P_H[i, :]))
        proj1 = np.transpose(proj1[:2]/proj1[-1])
        proj2 = np.dot(P2, np.transpose(P_H[i, :]))
        proj2 = np.transpose(proj2[:2]/proj2[-1])
        # compute error
        err.append(np.sum((proj1-pts1[i])**2 + (proj2-pts2[i])**2))
    avg = np.average(err)
    print('avg:',avg)
    return pts3d
# def triangulate(P1, pts1, P2, pts2):
#     N = pts1.shape[0]
#     pts3d = np.zeros((N,3))
#     p1_a, p2_a, p3_a = P1
#     p1_b, p2_b, p3_b = P2
#     for i in range(N):
#         x_a, y_a = pts1[i,:]
#         x_b, y_b = pts2[i,:]
#         A = np.array([y_a*p3_a - p2_a,
#         p1_a - x_a*p3_a,
#         y_b*p3_b - p2_b,
#         p1_b - x_b*p3_b])
#         _,_,vh = np.linalg.svd(A)
#         Xi = vh[-1,:]
#         X_resized = (Xi / Xi[-1])[:3]
#         pts3d[i,:] = X_resized
#     return pts3d

"""
Q3.2.1 Image Rectification
       [I] K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] M1 M2, rectification matrices (3x3 matrix)
           K1p K2p, rectified camera matrices (3x3 matrix)
           R1p R2p, rectified rotation matrices (3x3 matrix)
           t1p t2p, rectified translation vectors (3x1 matrix)
"""
def rectify_pair(K1, K2, R1, R2, t1, t2):
    #1. Compute the optical centers c1 and c2 of each camera
    c1 = -np.linalg.inv(K1@R1) @ (K1@t1)
    c2 = -np.linalg.inv(K2@R2) @ (K2@t2)
    # print(c1)
    # print(c1.shape)
    #2. Compute the new rotation matrix R = [r1 r2 r3]^T
    # r1,r2,r3 are all 3x1 column vector 
    # r1 = (c1-c2)/||c1-c2||
    r1 = (c1-c2)/np.linalg.norm(c1-c2)
    # r2 = R1(3,:)^T x r1
    R1_3 = np.reshape(R1[:,-1],(3,1))
    r2 = np.cross(R1_3,r1,axis=0)
    # r3 = r2 x r1
    r3 = np.cross(r2,r1,axis=0)
    #transpose r's
    r1 = np.transpose(r1)
    r2 = np.transpose(r2)
    r3 = np.transpose(r3)
    R = np.vstack((r1,r2,r3))
    R1p = R
    R2p = R
    # print('R',R.shape)
    #3. Compute the new intrinsic parameters as K1p=K2p=K2
    K1p = K2
    K2p = K2
    #4. Compute the new translation vectors
    t1p = -R@c1
    t2p = -R@c2
    #5. The rectification matrices of the cameras
    M1 = (K1p @ R1p) @ np.linalg.inv(K1 @ R1)
    M2 = (K2p @ R2p) @ np.linalg.inv(K2 @ R2)
    return M1,M2,K1p,K2p,R1p,R2p,t1p,t2p


"""
Q3.2.2 Disparity Map
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           max_disp, scalar maximum disparity value
           win_size, scalar window size value
       [O] dispM, disparity map (H1xW1 matrix)
"""
def get_disparity(im1, im2, max_disp, win_size):
    h,w = im2.shape[0:2]
    mask = np.ones((win_size,win_size))

    minVal = np.inf
    minDist = None

    # im1_reduced = im1.copy()
    # im1_reduced = im1_reduced[:,max_disp:]
    # im1_reduced = im1.copy()
    # im1_reduced = im1_reduced[:,max_disp:]
    im2Padded = im2.copy()
    pad = np.zeros((h,max_disp))
    # im2Padded = np.hstack((im2Padded,pad))
    im2PaddedL = np.hstack((pad,im2Padded))
    # print('im1shape : ',im1.shape)
    # print('im2Padded : ',im2Padded.shape)
    results = []
    for d in  range(0,max_disp):
        #img1 same size, im2 shifted to right by d pad other to 0
        # currIm2 = im2Padded[:,d:w+d]
        currim2L = im2PaddedL[:,max_disp-d:w+max_disp-d]
        diff = np.square(im1 - currim2L)
        currDist = scipy.signal.convolve2d(diff,mask,mode='same', boundary='fill', fillvalue=0)
        results.append(currDist)
    dispM = np.argmin(results,axis=0)

#argmin axis 0
        # if (currVal < minVal):
        #     minVal = currVal
        #     minDist = currDist
# np.argmin([M1,M2,M3],axis=0)
    # minDist = np.concatenate((im1[:,:max_disp],minDist),axis=1)
    return dispM
        


"""
Q3.2.3 Depth Map
       [I] dispM, disparity map (H1xW1 matrix)
           K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] depthM, depth map (H1xW1 matrix)
"""
def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    depthM = None
    c1 = -np.linalg.inv(K1@R1) @ (K1@t1)
    c2 = -np.linalg.inv(K2@R2) @ (K2@t2)
    f = K1[0,0]
    b = np.linalg.norm(c1-c2)

    bf = b*f
    depthM = np.divide(bf, dispM, out=np.zeros((dispM.shape[0],dispM.shape[1])), where=dispM!=0)
    # depthM = b*f/dispM[dispM!=0
    # c = np.divide(a, b, out=np.zeros((b.shape[0],b.shape[1])), where=b!=0)]
    return depthM


"""
Q3.3.1 Camera Matrix Estimation
       [I] x, 2D points (Nx2 matrix)
           X, 3D points (Nx3 matrix)
       [O] P, camera matrix (3x4 matrix)
"""
def estimate_pose(x, X):
    # replace pass by your implementation
    pass


"""
Q3.3.2 Camera Parameter Estimation
       [I] P, camera matrix (3x4 matrix)
       [O] K, camera intrinsics (3x3 matrix)
           R, camera extrinsics rotation (3x3 matrix)
           t, camera extrinsics translation (3x1 matrix)
"""
def estimate_params(P):
    # replace pass by your implementation
    pass
