function [L] = orderedCholesky(M, p, tol)
%
%   [L] = orderedCholesky(H, tol)
%
%   computes the pivoted Cholesky decomposition H = LL' of the matrix H up to 
%   a relative threshold of tol
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d = diag(M);
tr = sum(d);
tol = tol * tr;
[~,piv] = max(d);
L = [];
i = 1;
while (tr > tol)
    L(:, end+1) = M(:, piv);
    row = L(piv , 1:end-1);
    L(:, end) = L(:, end) - L(:, 1:end-1) * row';
    L(:, end) = L(:, end) / sqrt(L(piv, end));
    d = d - L(:, end) .* L(:, end);
    [~,piv] = max(d);
    tr = sum(d);
end
end