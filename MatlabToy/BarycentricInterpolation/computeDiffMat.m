function [D] = computeDiffMat(xi, w)

D = (kron(w, ones(size(w))') ./ kron(w', ones(size(w)))) ...
    ./ (kron(xi', ones(size(xi))) - kron(xi, ones(size(xi))'));
D(1:length(xi)+1:end) = 0;

D = D - diag(sum(D, 2));

end