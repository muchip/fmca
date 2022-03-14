function [cA, cH, cV, cD] = mywavedec2(Img, Tx, Ty, Lx, Ly, lx, ly)
ybar = ([-1:size(Img,1)] + 0.5) / size(Img,1);
xbar = ([-1:size(Img,2)] + 0.5) / size(Img,2);
% diagonal block
TImg = Ty * double(Img) * Tx';
TImg(find(Ly > ly),:) = 0;
TImg(:,find(Lx > lx)) = 0;
filter = TImg;
filter(find(Ly < ly),:) = 0;
filter(:,find(Lx < lx)) = 0;
filter = Ty' * filter * Tx;
InterpX = splineInterpolator(size(filter,2));
InterpY = splineInterpolator(size(filter,1));
coeff = InterpY\[zeros(1, size(filter,2));...
                 filter; zeros(1, size(filter,2))];
coeff = InterpX\[zeros(1,2 + size(filter,1));...
                 coeff'; zeros(1,2 + size(filter,1))];
yc = ([0:length(find(Ly == ly))-1] + 0.5) / length(find(Ly == ly));
xc = ([0:length(find(Lx == lx))-1] + 0.5) / length(find(Lx == lx));
[Xx,Yx] = meshgrid(xbar, xc);
[Xy,Yy] = meshgrid(ybar, yc);
Ex = B3((Xx-Yx) * size(filter,2));
Ey = B3((Xy-Yy) * size(filter,1));
filter = Ey * coeff' * Ex';
cD = filter;
% horizontal block
filter = TImg;
filter(find(Ly < ly),:) = 0;
filter(:,find(Lx == lx)) = 0;
filter = Ty' * filter * Tx;
InterpX = splineInterpolator(size(filter,2));
InterpY = splineInterpolator(size(filter,1));
coeff = InterpY\[zeros(1, size(filter,2));...
                 filter; zeros(1, size(filter,2))];
coeff = InterpX\[zeros(1,2 + size(filter,1));...
                 coeff'; zeros(1,2 + size(filter,1))];
yc = ([0:length(find(Ly == ly))-1] + 0.5) / length(find(Ly == ly));
xc = ([0:length(find(Lx < lx))-1] + 0.5) / length(find(Lx < lx));
[Xx,Yx] = meshgrid(xbar, xc);
[Xy,Yy] = meshgrid(ybar, yc);
Ex = B3((Xx-Yx) * size(filter,2));
Ey = B3((Xy-Yy) * size(filter,1));
filter = Ey * coeff' * Ex';
cH = filter;
% vertical block
filter = TImg;
filter(find(Ly == ly),:) = 0;
filter(:,find(Lx < lx)) = 0;
filter = Ty' * filter * Tx;
InterpX = splineInterpolator(size(filter,2));
InterpY = splineInterpolator(size(filter,1));
coeff = InterpY\[zeros(1, size(filter,2));...
                 filter; zeros(1, size(filter,2))];
coeff = InterpX\[zeros(1,2 + size(filter,1));...
                 coeff'; zeros(1,2 + size(filter,1))];
yc = ([0:length(find(Ly < ly))-1] + 0.5) / length(find(Ly < ly));
xc = ([0:length(find(Lx == lx))-1] + 0.5) / length(find(Lx == lx));
[Xx,Yx] = meshgrid(xbar, xc);
[Xy,Yy] = meshgrid(ybar, yc);
Ex = B3((Xx-Yx) * size(filter,2));
Ey = B3((Xy-Yy) * size(filter,1));
filter = Ey * coeff' * Ex';
cV = filter;
% approximation block
filter = TImg;
filter(find(Ly == ly),:) = 0;
filter(:,find(Lx == lx)) = 0;
filter = Ty' * filter * Tx;
InterpX = splineInterpolator(size(filter,2));
InterpY = splineInterpolator(size(filter,1));
coeff = InterpY\[zeros(1, size(filter,2));...
                 filter; zeros(1, size(filter,2))];
coeff = InterpX\[zeros(1,2 + size(filter,1));...
                 coeff'; zeros(1,2 + size(filter,1))];
yc = ([0:length(find(Ly < ly))-1] + 0.5) / length(find(Ly < ly));
xc = ([0:length(find(Lx < lx))-1] + 0.5) / length(find(Lx < lx));
[Xx,Yx] = meshgrid(xbar, xc);
[Xy,Yy] = meshgrid(ybar, yc);
Ex = B3((Xx-Yx) * size(filter,2));
Ey = B3((Xy-Yy) * size(filter,1));
filter = Ey * coeff' * Ex';
cA = filter;
end