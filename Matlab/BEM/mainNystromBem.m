clear all;
close all;
addpath('../')
refSol = @(X,Y) X.^2 - Y.^2 + 10 * X;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% set number of collocation points
i = 1;
n = 2000
% set subdivision of [0,1)
x = (1./n) * [0:n-1]';
% set up the boundary using the cubic B-spline
[b3, db3 d2b3] = cbspline;
fr = @(x) 0.1 + 0.1 * b3(4 * x - 2);
fDr = @(x) db3(4 * x - 2) * 0.4;
fD2r = @(x) d2b3(4 * x - 2) * 1.6;
[g, N, detT, T, D2g] = Gamma (fr, fDr, fD2r, x);
plot([g(:,1); g(1,1)], [g(:,2); g(1,2)], 'b-')
axis square
% init tensors for all combinations of differences
Gx = kron (ones(n,1), g(:,1)');
Gy = kron (ones(n,1), g(:,2)');
Nx = kron (ones(n,1), N(:,1)');
Ny = kron (ones(n,1), N(:,2)');
Gx = Gx' - Gx;
Gy = Gy' - Gy;
S  = kron (ones(n,1), x');
S  = S' - S;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute double layer operator
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute the nystroem matrix and set the diagonal to 0;
K = (Gx .* Nx + Gy .* Ny) ./ (Gx .^2 + Gy .^2 + speye (n));
K = K - spdiags (diag (K),0,n,n);
% compute the diagonal of the nystroem matrix separately
D = (D2g(:,1) .* N(:,1) + D2g(:,2) .* N(:,2)) ./ (2 * detT .^ 2);
% plug diagonal into the nystroem matrix and assemble the discrete operator
K =  0.5 * ((K + spdiags (D,0,n,n)) / (pi * n) - speye (n));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute single layer operator
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute the nystroem matrix and set the diagonal to 0;
V0 = log ((Gx .^2 + Gy .^ 2 + speye(n)) ...
          ./ (4 * sin (pi * S) .^2 + speye (n)));
V0 = V0 - spdiags (diag (V0),0,n,n);
% compute diagonal and add it to V0
d = 2 * log (detT / (2 * pi));
V0 = (V0 + spdiags (d, 0, n, n)) / n;
% compute the analytic part and plug everything together
V1 = getV1 (n);
V = -(V0 + V1) / 4 / pi;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute right hand side by interpolation
rhs = refSol(g(:,1),g(:,2));
% solve linear systems
rhoV = V \ rhs;
rhoK = K \ rhs;
% evaluate the potential at m inner curves of n points and compute
% the error of the solution in the interior of the domain.
m = 10;
[X, Y, uK] = potentialn (g, N, detT, T, D2g, rhoK, m);
[~, ~, uV] = potentialc (g, N, detT, T, D2g, rhoV, m);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sol = refSol(X, Y);
errK(i) = max (abs(uK(:) - sol(:)));
errV(i) = max (abs(uV(:) - sol(:)));
i = i + 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure(2);
% semilogy(errK,'b-')
% hold on;
% semilogy(errV,'r-')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
[~, I, T] = MEXbivariateCompressor(g', 2, 10, 1e-1);
toc
Q = sparse(T(:,1), T(:,2), T(:,3));
clear T;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Qerror = norm(Q * Q' - speye(size(Q)), 'fro') / norm(speye(size(Q)), 'fro')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
KW = Q * K(I, I) * Q';
VW = Q * V(I, I) * Q';
KW(find(abs(KW) < 1e-10)) = 0;
VW(find(abs(VW) < 1e-10)) = 0;
figure(2)
vismat(KW)
figure(3)
vismat(VW)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% visualize solution, since X,Y,u are only defined for the interior
% we have to append the (closed) boundary by hand
% figure(3)
% g(n+1,:) = g(1,:);
% X(:,m+1) = g(:,1);
% Y(:,m+1) = g(:,2);
% rhs(n+1) = rhs(1);
% uV(:,m+1) = rhs;
% clf
% surf (X,Y, uV);
%shading interp;