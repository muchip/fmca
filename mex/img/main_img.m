clear all;
close all;
addpath('../')
ImgInput = imread('fo.jpg');
Img = rgb2gray(ImgInput);
y = ([0:size(Img,1)-1] + 0.5) / size(Img,1);
x = ([0:size(Img,2)-1] + 0.5) / size(Img,2);
ybar = ([-1:size(Img,1)] + 0.5) / size(Img,1);
xbar = ([-1:size(Img,2)] + 0.5) / size(Img,2);
[Ty, Iy, Ly] = MEXsampletBasis(y, 3);
Ty = sparse(Ty(:,1), Ty(:,2), Ty(:,3), length(y), length(y));
[Tx, Ix, Lx] = MEXsampletBasis(x, 3);
Tx = sparse(Tx(:,1), Tx(:,2), Tx(:,3), length(x), length(x));
max(Lx)
max(Ly)
figure(1)
lvlx = 6;
lvly = 5;
Txtrunc = Tx;
Txtrunc(find(Lx > lvlx),:) = 0;
Tytrunc = Ty;
Tytrunc(find(Ly > lvly),:) = 0;
TImg = (Tytrunc * double(Img) * Txtrunc');
figure(1)
imshow(TImg);

TImg(find(Ly > lvly), :) = 0;
TImg(:, find(Lx > lvlx)) = 0;

TImg = Ty' * TImg * Tx;
InterpX = splineInterpolator(size(TImg,2));
InterpY = splineInterpolator(size(TImg,1));
coeff = InterpY\[zeros(1, size(TImg,2)); TImg; zeros(1, size(TImg,2))];
coeff = InterpX\[zeros(1,2 + size(TImg,1)); coeff'; zeros(1,2 + size(TImg,1))];
yc = ([0:length(find(Ly <= lvly))-1] + 0.5) / length(find(Ly <= lvly));
xc = ([0:length(find(Lx <= lvlx))-1] + 0.5) / length(find(Lx <= lvlx));
[Xx,Yx] = meshgrid(xbar, xc);
[Xy,Yy] = meshgrid(ybar, yc);
Ex = B3((Xx-Yx) * size(Img,2));
Ey = B3((Xy-Yy) * size(Img,1));
TImg = Ey * coeff' * Ex';
TImg = mat2gray(TImg);
figure(2)
imshow(TImg);
