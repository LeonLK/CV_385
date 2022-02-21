import numpy as np

# testArray = np.array([[1,2,3],[4,5,0]])

# minIdx = np.unravel_index(np.argmin(testArray, axis=None), testArray.shape)
# # print(minIdx)
# # print(testArray[minIdx])
# a = np.array([[3,2,2],[2,3,-2]])
# print(a.shape)
# u, s, vh = np.linalg.svd(a)
# print(u.shape)
# print(s.shape)
# print(vh.shape)

# print(a)
# print(vh)
# minIdx = np.unravel_index(np.argmin(a, axis=None), a.shape)
# result = u@s@vh
# print(result)

# a1 = 1
# a2 = 2
# a3 = 3
# a4 = 4
# p1 = np.transpose(np.array([[1,2],[3,4],[5,6]]))
# p2 = np.transpose(np.array([[1,2],[3,4],[5,6]])*10)
# print(p1.shape)
# p1x = p1[0,:]
# p1y = p1[1,:]
# p2x = p2[0,:]
# p2y = p2[1,:]
# p1xp2x = p1x*p2x
# p1xp2y = p1x*p2y
# p1yp2x = p1y*p2x
# p1yp2y = p1y*p2y
# # A = np.vstack((p1xp2x, p1xp2y, p1x, p1yp2x, p1yp2y, p1y, p2x, p2y, 1))
# A = np.vstack((p1x, p2x, p1y, p2y, np.ones(p1x.shape)*100, np.ones(p1x.shape)*200,np.ones(p1x.shape)*300, np.ones(p1x.shape)*400, np.ones(p1x.shape)))
# # print(np.transpose(A))

# ##2.2
# # y1 = np.array([[20,30,40]])
# # print(y1)
# # search_range = np.ones(y1.shape)*5
# # search_range = search_range.astype(np.int32)
# # print(search_range)
# # # Y = np.array(range(y1-search_range, y1+search_range))
# # diffM = y1-search_range
# # diffP = y1+search_range
# # print(diffM)
# # print(diffP)
# # # Y = np.arange(y1-search_range, y1+search_range)
# # # print(Y)
# # print(y1[0:2])
# # print(y1.shape)
# # a,b,c = y1[0,0],y1[0,1],y1[0,2]
# # print(a,b,c)
# y1 = np.array([[20,30,40]])
# y2 = np.array([[20,30,40]])/10
# yResult = np.vstack((y1,y2))
# print(y1)
# print(y2)
# print(yResult.shape)
# print(yResult)
# diff = np.sum(np.abs(y2-y1))
# print(diff)
# h = 20
# w = 15
# patchSize = 5
# y = np.arange(0,h)
# x = np.arange(0,w)
# inRangeX = (x >= patchSize) & (x < w-patchSize)
# inRangeY = (y >= patchSize) & (y < h-patchSize)
# print(inRangeX)
# print(inRangeY)
# # print(x)
# print(x[inRangeX])
# # print(y)
# print(y[inRangeY])

###TRAINg
# data = np.load("../data/some_corresp.npz")
# pts1 = data['pts1']
# P1 = np.array([[0,1,2,3],[4,5,6,7],[8,9,10,11]])
# print(P1)
# p1 = np.transpose(P1[0,:])
# p2 = np.transpose(P1[1,:])
# p3 = np.transpose(P1[2,:])
# x1,y1 = pts1[:,0],pts1[:,1]

# Temp = y1*p3[2]
# print(Temp.shape)

# disparity
d = 3
a = np.array([[1,2],[3,4]])
b = np.zeros((2,4))
print(a)
aPad = np.pad(a,((0,0),(d,0)),'edge')
# print(aPad)
c = np.hstack((a,b))
print(c)
#depth
# b = np.array([[1,2,0],[1,3,4]])
# a = 10
# print(b)
# c = np.divide(a, b, out=np.zeros((b.shape[0],b.shape[1])), where=b!=0)
# # c = np.divide(a,b,)

# print(c)