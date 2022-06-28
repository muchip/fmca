function y = barycentricLagrangePolynomials(x, xi, w)
% xi, w, f, are assumed to be row vectors, while we enforce that x is a
% column vector
x = reshape(x, length(x), 1);
% generate matrix of of weights;
W = kron(ones(size(x)), w) ... 
    ./ (kron(x, ones(size(xi))) - kron(ones(size(x)), xi));
% find values which are too large indicating that we evaluated at an
% abscissa
[I,J] = find(abs(W) > 1e20);
% perform interpolation
y = (W) ./ (W * ones(size(xi))');
% fix nan values
y(I, J) = 1;
end