clear all;
close all;
addpath('../')
ImgInput = imread('LuganoMuenster.png');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sx = 100;
sy = 100;
Img = rgb2gray(ImgInput);
y = ([0:size(Img,1)-1] + 0.5) / size(Img,1);
x = ([0:size(Img,2)-1] + 0.5) / size(Img,2);
ybar = ([-1:size(Img,1)] + 0.5) / size(Img,1);
xbar = ([-1:size(Img,2)] + 0.5) / size(Img,2);
[Ty, Iy, Ly] = MEXsampletBasis(y, 3);
Ty = sparse(Ty(:,1), Ty(:,2), Ty(:,3), length(y), length(y));
[Tx, Ix, Lx] = MEXsampletBasis(x, 3);
Tx = sparse(Tx(:,1), Tx(:,2), Tx(:,3), length(x), length(x));
max_lvl_x = max(Lx)
max_lvl_y = max(Ly)
TImg = (Ty * double(Img) * Tx');
figure(1)
imshow(TImg);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
theimg = zeros(sy, sx);
for ly = 0:max_lvl_y
    for lx = 0:max_lvl_x
        filter = TImg;
        filter(find(Ly ~= ly),:) = 0;
        filter(:,find(Lx ~= lx)) = 0;
        filter = Ty' * filter * Tx;
        InterpX = splineInterpolator(size(filter,2));
        InterpY = splineInterpolator(size(filter,1));
        coeff = InterpY\[zeros(1, size(filter,2));...
                         filter; zeros(1, size(filter,2))];
        coeff = InterpX\[zeros(1,2 + size(filter,1));...
                         coeff'; zeros(1,2 + size(filter,1))];
        yc = ([0:sy-1] + 0.5) / sy;
        xc = ([0:sx-1] + 0.5) / sx;
        [Xx,Yx] = meshgrid(xbar, xc);
        [Xy,Yy] = meshgrid(ybar, yc);
        Ex = B3((Xx-Yx) * size(filter,2));
        Ey = B3((Xy-Yy) * size(filter,1));
        filter = Ey * coeff' * Ex';
        theimg = theimg + filter;
    end

end
theimg = mat2gray(theimg);
figure(2)
imshow(theimg)
figure(1)
imshow(Img)
% single level upscaling
filter = double(Img);
InterpX = splineInterpolator(size(filter,2));
InterpY = splineInterpolator(size(filter,1));
coeff = InterpY\[zeros(1, size(filter,2));...
                 filter; zeros(1, size(filter,2))];
coeff = InterpX\[zeros(1,2 + size(filter,1));...
                 coeff'; zeros(1,2 + size(filter,1))];
yc = ([0:sy-1] + 0.5) / sy;
xc = ([0:sx-1] + 0.5) / sx;
[Xx,Yx] = meshgrid(xbar, xc);
[Xy,Yy] = meshgrid(ybar, yc);
Ex = B3((Xx-Yx) * size(filter,2));
Ey = B3((Xy-Yy) * size(filter,1));
filter = Ey * coeff' * Ex';
filter = mat2gray(filter);
figure(3)
imshow(filter)


