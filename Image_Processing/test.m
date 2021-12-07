%read in image in preprocess
im = images.d_pelvis_sagital_crop;
im_eq = adapthisteq(im);
gamma = 0.7; %lower value for brighter pxls
im_enh = imadjust(im_eq, [0.4 0.9],[], gamma);
figure()
subplot(1,6,1)
imshow(im)
title('Original Image')
subplot(1,6,2)
imhist(im)
subplot(1,6,3)
imshow(im_eq)
title('Equalized')
subplot(1,6,4)
imhist(im_eq)
subplot(1,6,5)
imshow(im_enh)
title('Enhanced')
subplot(1,6,6)
imhist(im_enh)


%create dilated mask
se1 =  strel('line',10,0);
se2 = strel('line',10,90);
mask = masks.bw_pelvis_sagital_crop;
mask_dil = imdilate(mask, [se1 se2]);

%num iterations
n = 30;
%method
m = 'Chan-vese';
%m = 'edge';

BW = activecontour(im_enh,mask_dil,n,m,'SmoothFactor', 1, 'ContractionBias', 0.5);

figure2('units','normalized','outerposition',[0.1 0.1 1.25 1.25])
subplot(1,3,1)
imshow(im_enh)
title('Enhanced Image')

subplot(1,3,2)
imshow(im_enh)
hold on
num = uint8(1);
imcontour(mask,1,'b')
imcontour(mask_dil,1,'g')
imcontour(BW,1,'r')
legend('annotated', 'dilated', 'predicted','Location','northwest')
title('Image with Mask Contour')
subplot(1,3,3)
imshow(BW)
title('Mask')