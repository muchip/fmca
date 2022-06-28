clear all;
close all;
deg = 10;
h = 0.001;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Runge function
f = @(x) 1 ./ (1 + 100 * (x-0.5).^2);
% Gaussian
f = @(x) exp(-50 * (x-0.5).^2);
%f = @(x) 20*(x - 0.5).^4;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xi = LejaPoints(deg+1);
w = barycentricWeights(xi);
minw = min(w)
maxw = max(w)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x = [0:h:1]';
P = LegendrePolynomials(xi', length(xi) - 1);
condVandermonde = cond(P)
y = evalBarycentric(x, xi, w, f(xi));
plot(x, y, 'b-')
hold on;
plot (xi, f(xi), 'ro')
norm(y-f(x), 'inf') / norm(f(x), 'inf')
% D = computeDiffMat(xi, w);
% %yp = evalBarycentric(x, xi, w, f(xi) * D');
% yp = evalBarycentricDerivative(x, xi, w, f(xi));
% ypfd = (0.5 / h) * (f(x+h) - f(x-h));
% display('finite difference approximation error')
% norm(ypfd-yp, 'inf') / norm(ypfd, 'inf')
% plot(x, yp, 'm-')
% plot(x, ypfd, 'c-')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%