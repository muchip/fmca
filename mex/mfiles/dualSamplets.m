clear all;
close all;
addpath('../')
k = @(X, Y) exp(-10 * abs(X-Y));
%k = @(X, Y) max(1-10*abs(X-Y),0);
%k = @(X, Y) (1 + 50 * abs(X-Y)) .* exp(-50 * abs(X-Y));
%k = @(X, Y) exp(-100 * abs(X-Y).^2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(0)
pts = rand(400,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
[T, I] = MEXsampletBasis(pts', 3, 10);
toc
Q = sparse(T(:,1), T(:,2), T(:,3));
clear T;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Qerror = norm(Q * Q' - speye(size(Q)), 'fro') / norm(speye(size(Q)), 'fro')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[X, Y] = meshgrid(pts);
K = k(X, Y);
x = [0:0.001:1];
[X, Y] = meshgrid(pts,x);
Keval = k(X, Y);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
invI = I;
invI(I) = [1:length(I)]'; 

    figure(1);
    clf;
    hold on;
for i = 36:36
    samplet = Q(i,invI);
    sampletTilde = (K + 1e-12 * eye(size(K))) \ samplet';
    plot(x', Keval* samplet', 'b-', 'linewidth', 3);
        plot(x', Keval* sampletTilde,'r-', 'linewidth', 3);
    hold on;

    legend('phi','tphi')
    set(gca,'fontsize',30)
end