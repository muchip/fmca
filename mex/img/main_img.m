clear all;
close all;
addpath('../')
ImgInput = imread('NCC1701D.jpg');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Img = rgb2gray(ImgInput);
y = ([0:size(Img,1)-1] + 0.5) / size(Img,1);
x = ([0:size(Img,2)-1] + 0.5) / size(Img,2);
ybar = ([-1:size(Img,1)] + 0.5) / size(Img,1);
xbar = ([-1:size(Img,2)] + 0.5) / size(Img,2);
[Ty, Iy, Ly] = MEXsampletBasis(y, 2);
Ty = sparse(Ty(:,1), Ty(:,2), Ty(:,3), length(y), length(y));
[Tx, Ix, Lx] = MEXsampletBasis(x, 2);
Tx = sparse(Tx(:,1), Tx(:,2), Tx(:,3), length(x), length(x));
max_lvl_x = max(Lx)
max_lvl_y = max(Ly)
TImg = (Ty * double(Img) * Tx');
figure(1)
imshow(TImg);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
theimg = [];
% for ly = 0:max_lvl_y
%     row = [];
%     for lx = 0:max_lvl_x
%         filter = TImg;
%         filter(find(Ly ~= ly),:) = 0;
%         filter(:,find(Lx ~= lx)) = 0;
%         filter = Ty' * filter * Tx;
%         InterpX = splineInterpolator(size(filter,2));
%         InterpY = splineInterpolator(size(filter,1));
%         coeff = InterpY\[zeros(1, size(filter,2));...
%                          filter; zeros(1, size(filter,2))];
%         coeff = InterpX\[zeros(1,2 + size(filter,1));...
%                          coeff'; zeros(1,2 + size(filter,1))];
%         yc = ([0:length(find(Ly == ly))-1] + 0.5) / length(find(Ly == ly));
%         xc = ([0:length(find(Lx == lx))-1] + 0.5) / length(find(Lx == lx));
%         [Xx,Yx] = meshgrid(xbar, xc);
%         [Xy,Yy] = meshgrid(ybar, yc);
%         Ex = B3((Xx-Yx) * size(filter,2));
%         Ey = B3((Xy-Yy) * size(filter,1));
%         filter = Ey * coeff' * Ex';
%         filter = mat2gray(filter);
%         row = [row, filter];
%     end
%     theimg = [theimg; row];
% end
%figure(2)
%imshow(theimg);
% lx = max_lvl_x;
% ly = max_lvl_y;
% filter = TImg;
% filter(find(Ly ~= ly),:) = 0;
% filter(:,find(Lx >= lx)) = 0;
% filter = Ty' * filter * Tx;
% InterpX = splineInterpolator(size(filter,2));
% InterpY = splineInterpolator(size(filter,1));
% coeff = InterpY\[zeros(1, size(filter,2));...
%                  filter; zeros(1, size(filter,2))];
% coeff = InterpX\[zeros(1,2 + size(filter,1));...
%                  coeff'; zeros(1,2 + size(filter,1))];
% yc = ([0:length(find(Ly == ly))-1] + 0.5) / length(find(Ly == ly));
% xc = ([0:length(find(Lx == lx))-1] + 0.5) / length(find(Lx < lx));
% [Xx,Yx] = meshgrid(xbar, xc);
% [Xy,Yy] = meshgrid(ybar, yc);
% Ex = B3((Xx-Yx) * size(filter,2));
% Ey = B3((Xy-Yy) * size(filter,1));
% filter = Ey * coeff' * Ex';
% filter = mat2gray(filter);
% imshow(filter)

[cA, cH, cV, cD] = mywavedec2(Img, Tx, Ty, Lx, Ly, max_lvl_x, max_lvl_y-1);
[cA1, cH1, cV1, cD1] = mywavedec2(Img, Tx, Ty, Lx, Ly, max_lvl_x-1, max_lvl_y-1);
%theimg1 = [mat2gray(cA1), mat2gray(cV1); mat2gray(cH1), mat2gray(cD1)];
theimg = [mat2gray(cA), mat2gray(cV); mat2gray(cH), mat2gray(cD)];
figure(1)
imshow(theimg)
figure(2)
imshow(Img)