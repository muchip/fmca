clear all;
close all;
addpath('../')
ImgInput = imread('LuganoMuenster.png');
Img = rgb2gray(ImgInput);
y = [0:size(Img,1)-1] / size(Img,1);
x = [0:size(Img,2)-1] / size(Img,2);
[Ty, Iy, Ly, Ry] = MEXsampletBasis(y, 5);
Ty = sparse(Ty(:,1), Ty(:,2), Ty(:,3), length(y), length(y));
[Tx, Ix, Lx, Rx] = MEXsampletBasis(x, 5);
max(Lx)
max(Ly)
Tx = sparse(Tx(:,1), Tx(:,2), Tx(:,3), length(x), length(x));
figure(1)
lvlx = 6;
lvly = 6;
Txtrunc = Tx;
Txtrunc(find(Lx > lvlx),:) = 0;
Tytrunc = Ty;
Tytrunc(find(Ly > lvly),:) = 0;
TImg = (Tytrunc * double(Img) * Txtrunc');
size(TImg)
imshow(TImg);

TImg(find(Ly > lvly), :) = 0;
TImg(:, find(Lx > lvlx)) = 0;
figure(1)
%TImgC = mat2gray(TImg);
TImg = Ty' * TImg * Tx;
TImg = mat2gray(TImg);
figure(2)
imshow(TImg);
