function [L] = pivotedCholesky(H, tol)
%
%   [L] = pivotedCholesky(H, tol)
%
%   computes the pivoted Cholesky decomposition H = LL' of the matrix H up to 
%   a relative threshold of tol
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d = diag(H);
tr = sum(d);
[~, piv] = max(d);
tol = tol * tr;
L = [];
while (tr > tol)
    L(:, end+1) = H(:, piv);
    row = L(piv , 1:end-1);
    L(:, end) = L(:, end) - L(:, 1:end-1) * row';
    L(:, end) = L(:, end) / sqrt(L(piv, end));
    d = d - L(:, end) .* L(:, end);
    [~, piv] = max(d);
    tr = sum(d);
end
end

