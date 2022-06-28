clear all;
close all;
max_order = 14;
evalpts = [0:0.001:1]';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
f = @(X, Y) exp(-10 * ((X-0.5).^2+(Y-0.5).^2));
%f = @(X, Y) X.^9 .* Y.^5;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xi = LejaPoints(max_order);
V = LegendrePolynomials(xi', max_order - 1);
invV = inv(V);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
P = LegendrePolynomials(evalpts, max_order - 1);
[X,Y] = meshgrid(xi);
C = invV * f(X,Y) * invV';
% make interpolation sparse
for i = 1:size(C,1)
    for j = size(C,1)-i+2:size(C,2)
        C(i,j)=0;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(1);
vismat(C)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(2)
[X,Y] = meshgrid(evalpts);
surf(X,Y, abs(P * C * P' - f(X,Y)));
shading interp

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(3)
surf(X,Y, P * C * P');
shading interp
