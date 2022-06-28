clear;
close all;
deg = 5000;
h = 0.01;
point_type = 'Chebyshev'
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Runge function
%f = @(x) 1 ./ (1 + 100 * (x-0.5).^2);
% Gaussian
f = @(x) exp(-50 * (x-0.5).^2);
%f = @(x) 20*(x - 0.5).^4;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
switch point_type
    case 'equidistant'
        % equidistant nodes
        xi = [0:deg] / deg;
        w = xi;
        for i = 0:deg
            if mod(i,2)
                w(i+1) = nchoosek(deg,i);
            else
                w(i+1) = -nchoosek(deg,i);
            end
        end
    case 'Chebyshev'
        %Chebyshev nodes
        xi = 0.5 * (cos((2 * [1:(deg+1)] - 1) / (2 * (deg+1)) * pi) + 1);
        w = sin((2 * [1:(deg+1)] - 1) / (2 * (deg+1)) * pi);
        w(2:2:end) = -w(2:2:end);
    otherwise
        display('choose equidistant or Chebyshev')
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x = [0:h:1]';
L = barycentricLagrangePolynomials(x,xi,w);
P = LegendrePolynomials(xi', length(xi) - 1);
P2 = LegendrePolynomials(x, length(xi) - 1);
cond(P)
% y = evalBarycentric(x, xi, w, f(xi));
% plot(x, y, 'b-')
% hold on;
% plot (xi, f(xi), 'ro')
% D = computeDiffMat(xi, w);
% %yp = evalBarycentric(x, xi, w, f(xi) * D');
% yp = evalBarycentricDerivative(x, xi, w, f(xi));
% ypfd = (0.5 / h) * (f(x+h) - f(x-h));
% display('finite difference approximation error')
% norm(ypfd-yp, 'inf') / norm(ypfd, 'inf')
% plot(x, yp, 'm-')
% plot(x, ypfd, 'c-')
semilogy(x, abs(L * P - P2),'.')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%