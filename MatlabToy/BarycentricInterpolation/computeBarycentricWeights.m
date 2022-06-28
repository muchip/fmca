function W = computeBarycentricWeights(xi)
%
% W = computeBarycentricWeights(xi)
%
% calculates the barycentric interpolation weights for the node sequence xi
%
W = zeros(size(xi));
W(1) = 1;
for j = 2:length(xi)
    for k = 1:j-1
        W(k) = (xi(k) - xi(j)) * W(k);
    end
    W(j) = prod(xi(j) - xi(1:j-1));
end
 W = 1./ W;
end