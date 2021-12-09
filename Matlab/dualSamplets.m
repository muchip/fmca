clear all;
close all;
k = @(X, Y) exp(-20 * abs(X-Y));
%k = @(X, Y) (1 + 50 * abs(X-Y)) .* exp(-50 * abs(X-Y));
%k = @(X, Y) exp(-100 * abs(X-Y).^2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pts = [0:0.02:1];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
[~, I, T] = MEXbivariateCompressor(pts, 3, 0.8, 1e-6);
Q = sparse(T(:,1), T(:,2), T(:,3));
clear T;
toc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Qerror = norm(Q * Q' - speye(size(Q)), 'fro') / norm(speye(size(Q)), 'fro')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(1);
spy(Q);
[X, Y] = meshgrid(pts);
K = k(X, Y);
x = [0:0.00001:1];
[X, Y] = meshgrid(pts,x);
Keval = k(X, Y);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
invI = I;
invI(I) = [1:length(I)]'; 
for i = 1:length(pts)
    figure(2);
    hold on;
    plot(x', Keval(:,i), 'k-', 'linewidth', 2);
end
for i = 1:40
    figure(1);
    clf;
    samplet = Q(i,invI);
    sampletTilde = (K + 1e-12 * eye(size(K))) \ samplet';
    plot(x', Keval* samplet', 'b-', 'linewidth', 2);
    hold on;
    plot(x', Keval* sampletTilde,'r-', 'linewidth', 2);
pause
end