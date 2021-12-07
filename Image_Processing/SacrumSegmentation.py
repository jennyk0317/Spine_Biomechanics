# %%
import skimage as ski
from skimage.io import imread, imshow, imsave
from skimage.color import rgb2gray
from scipy.io import loadmat
from skimage.feature import peak_local_max
from skimage.segmentation import active_contour, felzenszwalb, mark_boundaries, watershed
from skimage.segmentation.morphsnakes import inverse_gaussian_gradient, inverse_gaussian_gradient
from skimage.segmentation import (morphological_chan_vese,
                                  morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient,
                                  checkerboard_level_set, chan_vese)
from skimage.filters import gaussian, try_all_threshold, threshold_local, threshold_otsu
from skimage.measure import find_contours
from skimage.exposure import histogram
from skimage.transform import rescale, resize, AffineTransform, warp
from skimage.morphology import disk, dilation, square
from skimage.feature import canny
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage as ndi


# %%
os.chdir(r'C:\Users\jihyk\Desktop\SBM\Sacrum_Seg\03_Colab\Image_Processing')
img = imread('sacrum_sagital_enh.jpg')
img = rgb2gray(img)
mask = imread('sacrum_sagital_mask.jpg')
mask = rgb2gray(mask)
imshow(img, cmap='bone')


skimage.__version__# %%
'''
f1 = plt.figure(1)
imshow(img); plt.axis('off')
plt.show()
img[img < 0.4] = 0
imshow(img); plt.axis('off')
plt.show()'''
img_sacrum = img[100:-150, 200:-150]
local_thresh = threshold_local(img_sacrum, block_size = 51, method = 'gaussian', offset = 0)
otsu = threshold_otsu(gaussian(img_sacrum))

img_local = img_sacrum > local_thresh
img_otsu = img_sacrum > otsu

segments = felzenszwalb(img_sacrum, scale=100, sigma=0.5, min_size=100)

fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

#imshow(img_sacrum); plt.axis('off')
ax[0,0].imshow(img_sacrum, cmap = 'bone')
ax[0,1].imshow(mark_boundaries(img_sacrum, segments), cmap = 'bone')
ax[1,0].imshow(img_local, cmap = 'bone')
ax[1,1].imshow(img_otsu, cmap = 'bone')

#%%
seg_sacrum = imread('sacrum_sagital.jpg', as_gray=True)[::2, ::2]
sacrum = imread('sacrum_sagital.jpg')/255
imshow(sacrum, cmap='bone')
plt.show()
'''
imshow(seg_sacrum, cmap = 'bone')
plt.show()'''

# %% morphological snake methods
def store_evolution_in(lst):
    """Returns a callback function to store the evolution of the level sets in
    the given list.
    """

    def _store(x):
        lst.append(np.copy(x))

    return _store
img = img

mask_scl = dilation(mask, square(20))
init_ls = mask_scl
evolution = []
callback = store_evolution_in(evolution)
ls = morphological_chan_vese(img, 18, init_level_set=init_ls, smoothing=1,
                             iter_callback=callback)


fig, axes = plt.subplots(2, 2, figsize=(8, 8))
ax = axes.flatten()

ax[0].imshow(img, cmap="gray")
ax[0].set_axis_off()
ax[0].contour(ls, [0.5], colors='r')
ax[0].set_title("Morphological ACWE segmentation", fontsize=12)

ax[1].imshow(ls, cmap="gray")
ax[1].set_axis_off()
contour = ax[1].contour(evolution[2], [0.5], colors='g')
contour.collections[0].set_label("Iteration 2")
contour = ax[1].contour(evolution[7], [0.5], colors='y')
contour.collections[0].set_label("Iteration 7")
contour = ax[1].contour(evolution[-1], [0.5], colors='r')
contour.collections[0].set_label("Iteration 18")
ax[1].legend(loc="upper right")
title = "Morphological ACWE evolution"
ax[1].set_title(title, fontsize=12)

# Morphological GAC
image = img
gimage = gaussian(image, 0.1)

# Initial level set
init_ls = mask_scl
# List with intermediate results for plotting the evolution
evolution = []
callback = store_evolution_in(evolution)
ls_2 = morphological_geodesic_active_contour(gimage, 130, init_ls,
                                           smoothing=1, balloon=-1,
                                           threshold=0.69,
                                           iter_callback=callback)

ax[2].imshow(image, cmap="gray")
ax[2].set_axis_off()
ax[2].contour(ls_2, [0.5], colors='r')
ax[2].set_title("Morphological GAC segmentation", fontsize=12)

ax[3].imshow(ls_2, cmap="gray")
ax[3].set_axis_off()
contour = ax[3].contour(evolution[0], [0.5], colors='g')
contour.collections[0].set_label("Iteration 0")
contour = ax[3].contour(evolution[100], [0.5], colors='y')
contour.collections[0].set_label("Iteration 100")
contour = ax[3].contour(evolution[-1], [0.5], colors='r')
contour.collections[0].set_label("Iteration 130")
ax[3].legend(loc="upper right")
title = "Morphological GAC evolution"
ax[3].set_title(title, fontsize=12)

fig.tight_layout()
plt.show()

#%% Watershed Method

imshow(ls, cmap='gray')
distance = ndi.distance_transform_edt(ls)
coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=ls)
mask = np.zeros(distance.shape, dtype=bool)
mask[tuple(coords.T)] = True
markers, _ = ndi.label(mask)
labels = watershed(-distance, markers, mask=ls)

fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(ls, cmap=plt.cm.gray)
ax[0].set_title('Overlapping objects')
ax[1].imshow(-distance, cmap=plt.cm.gray)
ax[1].set_title('Distances')
ax[2].imshow(labels, cmap=plt.cm.nipy_spectral)
ax[2].set_title('Separated objects')

for a in ax:
    a.set_axis_off()

fig.tight_layout()
plt.show()


# %%
contours = find_contours(mask, 0.8)
line = contours[1]
#imshow(mask)
im1 = imshow(img)
#plt.plot(line[:,1], line[:,0], linewidth = 2)
center_x = np.mean(line[:,1])
center_y = np.mean(line[:,0])

mask_scl = dilation(mask, square(20))


imshow(mask_scl)

cnt_scl = find_contours(mask_scl, 0.8)
line2 = cnt_scl[1]
line2[:,1] = line2[:,1]+7 #offset to center mask
line2[:,0] = line2[:,0]-5


#s = np.linspace(0, 2*np.pi, 60)
#r = center_y + 60*np.sin(s)
#c = center_x + 60*np.cos(s)
#init = np.array([r,c]).T
plt.plot(line[:, 1], line[:, 0], '--r', lw=2)
plt.show(); 

imshow(img)
plt.plot(line2[:, 1], line2[:, 0], '--b', lw=2)
plt.axis('off')

shift = AffineTransform(translation=(15,0))
mask_scl = warp(mask_scl, shift.inverse)
imshow(mask_scl)


# %% common choice of gaussian filter with sigma bet 0.1 and 1
imshow(img)
imshow(mask_scl)
mask_scl = mask_scl > 0.5
cnt_scl = find_contours(mask_scl, 0.8)
line2 = cnt_scl[2]
line2[:,1] = line2[:,1] -10#offset to center mask
line2[:,0] = line2[:,0]- 7

snake = active_contour(gaussian(img,1),
                       line2, w_edge=1, w_line=0, max_iterations=1000)

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(img, cmap=plt.cm.gray)
ax.plot(line2[:, 1], line2[:, 0], '--r', lw=1)
ax.plot(snake[:, 1], snake[:, 0], '-b', lw=1)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, img.shape[1], img.shape[0], 0])

# %%
imshow(img)
ls3 = chan_vese(gaussian(img,1), mu=0.25, lambda1=1, lambda2=2, max_iter=10, dt=0.5,init_level_set = init_ls, 
                                            )
imshow(ls3)
# %%
