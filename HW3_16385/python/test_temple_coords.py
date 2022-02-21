import numpy as np
from eight_point import eight_point
import helper as hlp
import skimage.io as io
import submission as sub
import matplotlib.pyplot as plt
import cv2


# 1. Load the two temple images and the points from data/some_corresp.npz
#Might need to change the path, my pc is different than my laptop for some reason... :(
temple1 = cv2.imread('../data/im1.png')
temple2 = cv2.imread('../data/im2.png') 
data = np.load("../data/some_corresp.npz")
pts1 = data['pts1']
pts2 = data['pts2']
M = max(temple1.shape[0:2]) # Q: should this be max temple1 && temple2? Are they same size?

# 2. Run eight_point to compute F
F = sub.eight_point(pts1,pts2,M)
print(F)
# hlp.displayEpipolarF(temple1,temple2,F)

# 3. Load points in image 1 from data/temple_coords.npz
dataTempCoord = np.load("../data/temple_coords.npz")
# 4. Run epipolar_correspondences to get points in image 2
pts_im2 = sub.epipolar_correspondences(temple1,temple2,F,dataTempCoord['pts1'])
# print(dataTempCoord['pts1'].shape)
# testC = sub.epipolarCorrespondence1(temple1,temple2,F,dataTempCoord['pts1'][:,0],dataTempCoord['pts1'][:,1])
# hlp.epipolarMatchGUI(temple1,temple2,F)
intrinsic = np.load("../data/intrinsics.npz")
K1 = intrinsic['K1']
K2 = intrinsic['K2']
# print('K@@@@',K2.shape)
E = sub.essential_matrix(F,K1,K2)
print(E)
# 5. Compute the camera projection matrix P1
M1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
P1 = K1 @ M1
# 6. Use camera2 to get 4 camera projection matrices P2
M2_4 = hlp.camera2(E)
print(M2_4.shape)

# 7. Run triangulate using the projection matrices
P2 = None
pts3d = None
M2 = None
for i in range(4):
    M2_i = M2_4[:, :, i]
    P2_i = K2 @ M2_i
    # P2_i = np.dot(K2,M2)
    pts3d_i = sub.triangulate(P1, dataTempCoord['pts1'], P2_i, pts_im2)
    # print('3dSHape:',pts3d_i.shape)
    if np.min(pts3d_i[:, 2]) > 0:
        print('Valid')
        P2 = P2_i
        M2 = M2_i
        pts3d = pts3d_i
# 8. Figure out the correct P2
#see above
# print(pts3d.shape)
# 9. Scatter plot the correct 3D points
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(correct_pts3d[:,0],correct_pts3d[:,1],correct_pts3d[:,2])
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# plt.show()
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
# 10. Save the computed extrinsic parameters (R1,R2,t1,t2) to data/extrinsics.npz
# R1,t1 = np.linalg.qr(M1)
# R2,t2 = np.linalg.qr(M2)
R1,t1 = M1[0:3,0:3],M1[:,-1]
print(R1)
print(t1)
R2,t2 = M2[0:3,0:3],M2[:,-1]
R2 = R2/R2[-1,-1]
print(R2)
print(t2)
# print(R1.shape)
# print(t1.shape)

t1 = np.reshape(t1,(3,1))
t2 = np.reshape(t2,(3,1))

np.savez('../data/extrinsics.npz', R1 = R1, R2 = R2, t1 = t1, t2 = t2)


# # 3. Load points in image 1 from data/temple_coords.npz
# # data = np.load(os.path.join(datadir, "temple_coords.npz"))
# temple_pts = data['pts1']
# print(temple_pts.shape)

# # 4. Run epipolar_correspondences to get points in image 2
# # temple_pts_2 = sub.epipolar_correspondences(img1, img2, F, temple_pts)
# # hlp.epipolarMatchGUI(img1, img2, F)
# temple_pts_2 = data['pts2']

# # Loading intrinsic matrices
# # data = np.load(os.path.join(datadir, "intrinsics.npz"))
# # K1 = data['K1']
# # K2 = data['K2']

# # Calculating essential matrix
# E = sub.essential_matrix(F, K1, K2)

# # 5. Compute the camera projection matrix P1
# P1 = np.zeros((3,4))
# P1[:3, :3] = K1

# # 6. Use camera2 to get 4 camera projection matrices P2
# extrinsics = hlp.camera2(E)

# # 7. Run triangulate using the projection matrices
# max_positives = 0
# correct_pts3d = []
# correct_P2 = []
# for i in range(extrinsics.shape[2]):
#     P2 = np.matmul(K2, extrinsics[:, :, i])
#     pts3d = sub.triangulate(P1, temple_pts, P2, pts_im2)
#     positives = np.sum(pts3d[:,-1]>0)
#     # 8. Figure out the correct P2
#     if positives > max_positives:
#         correct_P2 = P2
#         max_positives = positives
#         correct_pts3d = pts3d
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(correct_pts3d[:,0],correct_pts3d[:,1],correct_pts3d[:,2])
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# plt.show()