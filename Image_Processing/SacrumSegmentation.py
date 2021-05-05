# %%
from skimage.io import imread, imshow
from skimage.color import rgb2gray
from scipy.io import loadmat
from skimage.segmentation import active_contour
from skimage.filters import gaussian
from skimage.measure import find_contours
from skimage.exposure import histogram
from skimage.transform import rescale, resize
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2 as cv


# %%
#os.chdir(r'C:\Users\jihyk\Desktop\SBM')
img = imread('sacrum_sagital.jpg')
img = rgb2gray(img)
mask = imread('sacrum_sagital_mask.jpg')
mask = rgb2gray(mask)
# %%
f1 = plt.figure(1)
imshow(img, cmap = 'bone')

f2 = plt.figure(2)
imshow(mask)
plt.show()
# %%
contours = find_contours(mask, 0.8)

fig, ax = plt.subplots()
ax.imshow(mask)
'''
for contour in contours:
    ax.plot(contour[:,1], contour[:,0], linewidth = 2)
    plt.show()
'''
# %%
line = contours[1]
#imshow(mask)
imshow(img)
plt.plot(line[:,1], line[:,0])
plt.show()
# %%
#print(np.shape(line))

im = cv.imread('sacrum_sagital.jpg')
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
mask = imread('sacrum_sagital_mask.jpg')
maskgray = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)

ret, thresh = cv.threshold(maskgray, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cnt = contours[3]
cnt_scaled = scale_contour(cnt, 2)
cv.drawContours(mask, [cnt_scaled], 0, (0,255,0), 3)
#cv.drawContours(mask, [cnt], 0, (0,255,0), 3)
cv.imshow('image', im)
cv.imshow('Contours', mask)
cv.waitKey(0)
cv.destroyAllWindows()

print("Number of Contours found = " + str(len(contours)))

init1 = cnt_scaled[:, :,0]
init2 = cnt_scaled[:, :,1]
init = np.concatenate((init1, init2), axis = 1)
print(np.shape(init))
print("Size of contour:", np.shape(cnt_scaled))



# %%
imshow(im)

plt.plot(cnt_scaled[:, :, 0], cnt_scaled[:, :, 1])
plt.show()
# %%
def scale_contour(cnt, scale):
    M = cv.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled

# %%
snake = active_contour(gaussian(im, 3),
                       cnt_scaled, alpha=0.015, beta=10, gamma=0.001)

# %%
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(img, cmap=plt.cm.gray)
ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, img.shape[1], img.shape[0], 0])
# %%
