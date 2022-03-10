clear all;
close all;
addpath('../')
ImgInput = imread('duck.jpg');
Img = rgb2gray(ImgInput);
y = [0:size(Img,1)-1] / size(Img,1);
x = [0:size(Img,2)-1] / size(Img,2);
[Ty, Iy, Ly] = MEXsampletBasis(y, 3);
Ty = sparse(Ty(:,1), Ty(:,2), Ty(:,3), length(y), length(y));
[Tx, Ix, Lx] = MEXsampletBasis(x, 3);
max(Lx)
max(Ly)
Tx = sparse(Tx(:,1), Tx(:,2), Tx(:,3), length(x), length(x));
figure(1)
TImg = (Ty*Ty * double(Img) * Tx' * Tx);
imin = min(min(TImg))
imax = max(max(TImg))
TImg = (TImg - imin) / (imax - imin);
imin = min(min(TImg))
imax = max(max(TImg))
%TImg(find(Ly > 6), :) = 0;
%TImg(:, find(Lx > 6)) = 0;
%TImg = Ty' * TImg * Tx;
imshow(TImg);
