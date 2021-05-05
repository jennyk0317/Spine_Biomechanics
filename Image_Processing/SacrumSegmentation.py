# %%
from skimage.io import imread, imshow
from skimage.color import rgb2gray
from scipy.io import loadmat
from skimage.segmentation import active_contour
from skimage.filters import gaussian
from skimage.measure import find_contours
import numpy as np
import os
import matplotlib.pyplot as plt


# %%
os.chdir(r'C:\Users\jihyk\Desktop\SBM\MATLAB\Spine annotation\EOS Spine Images\Spine_EOS_0050-0103\0051')
mat = loadmat('0051 marked.mat')

# %%
os.chdir(r'C:\Users\jihyk\Desktop\SBM')
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
