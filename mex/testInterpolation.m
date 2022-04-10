clear all;
close all;
[X, Y] = meshgrid([0:0.01:1]);
P = [X(:), Y(:)]';
[Xi, invV, eval] = MEXpolynomialInterpolator(P, [1 4], 10);
cond(invV)

f = zeros(1,size(Xi,2));
for i=1:size(Xi,2)
    f(i) = exp(Xi(1,i)) * sin(0.1 * pi * Xi(2,i));
end
Interp = (f * invV') * eval;

Interp = reshape(Interp, size(X,1), size(X,2));
F = zeros(size(X));
for j = 1:size(X,2)
    for i = 1:size(X,1)
        F(i,j) = exp(X(i,j)) * sin(0.1 * pi * Y(i,j));
    end
end
error = max(max(abs(F-Interp))/max(max(F)))

surf(X,Y,F)