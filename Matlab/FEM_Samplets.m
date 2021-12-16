clear all;
close all;
hat = @(x) max(1-abs(x),0)
h = 0.001;
k = @(X,Y) hat(1/h*(X-Y));
pts = [0:h:1];
[~, I, T] = MEXbivariateCompressor(pts, 1, 0.8, 1e-6);
Q = sparse(T(:,1), T(:,2), T(:,3));
clear T;
D = h * ones(length(pts) + 1,1) * [1/6, 2/3, 1/6];
M = spdiags(D, -1:1, length(pts), length(pts));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Qerror = norm(Q * Q' - speye(size(Q)), 'fro') / norm(speye(size(Q)), 'fro')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(1);
spy(Q);
[X, Y] = meshgrid(pts);
K = k(X, Y);
x = [0:h:1];
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
for i = 1:100
    figure(1);
    clf;
    samplet = Q(i,invI);
    sampletTilde = M \ samplet';
    plot(x', samplet', 'b-', 'linewidth', 2);
    hold on;
    %plot(x', Keval* sampletTilde,'r-', 'linewidth', 2);
pause
end