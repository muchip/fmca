clear all;
close all;

rot = @(ang) [cos(ang) sin(ang); -sin(ang) cos(ang)];
load bunnySurface.txt

[U,S,V] = svd(bunnySurface,0);

B2D = bunnySurface * V;
B2D = B2D(:,1:2);
R = rot(2.81*pi/4);

B2D = B2D * R;
plot(B2D(:,1),B2D(:,2),'k.')