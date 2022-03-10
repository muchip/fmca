%clear all;
%close all;
alpha = 0.2;
ell = 0.01;
c = 1 ./ (2 * alpha * ell * ell);
ell = 4;
rat_quad = @(X,Y) 1 ./ (1+c*(X-Y).^2).^alpha;


x = [0:300]/300;

[X,Y] = meshgrid(x);

K = rat_quad(X,Y);

semilogy(sort(eig(K),'descend'))
hold on;
%plot(1:1000, [1:1000].^(-2))