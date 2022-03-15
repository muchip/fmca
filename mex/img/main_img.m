clear all;
close all;
addpath('../')
ImgInput = imread('LuganoMuenster.png');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Img = rgb2gray(ImgInput);
y = ([0:size(Img,1)-1] + 0.5) / size(Img,1);
x = ([0:size(Img,2)-1] + 0.5) / size(Img,2);
ybar = ([-1:size(Img,1)] + 0.5) / size(Img,1);
xbar = ([-1:size(Img,2)] + 0.5) / size(Img,2);
% samplet basis in vertical direction
[Ty, Iy, Ly] = MEXsampletBasis(y, 10);
Ty = sparse(Ty(:,1), Ty(:,2), Ty(:,3), length(y), length(y));
% samplet basis in horizontal direction
[Tx, Ix, Lx] = MEXsampletBasis(x, 10);
Tx = sparse(Tx(:,1), Tx(:,2), Tx(:,3), length(x), length(x));
max_lvl_x = max(Lx)
max_lvl_y = max(Ly)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
theImg = ImgInput;
for i=1:3
    [cA, cH, cV, cD] = mywavedec2(...
                         ImgInput(:,:,i), Tx, Ty, Lx, Ly, max_lvl_x, max_lvl_y);
    theImg(:,:,i) = [cA, cV; cH, cD];
end
figure(1)
imshow(theImg)
[cA, cH, cV, cD] = mywavedec2(Img, Tx, Ty, Lx, Ly, max_lvl_x, max_lvl_y);
theImg = [mat2gray(cA), mat2gray(cV); mat2gray(cH), mat2gray(cD)];
figure(2)
imshow(theImg)