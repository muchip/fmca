function W = barycentricWeights(xi)
%
% W = barycentricWeights(xi)
%
% calculates the barycentric interpolation weights for the node sequence xi
%
W = zeros(size(xi));
logW = zeros(size(xi));

for i=1:length(xi)
    logW(i) = sum(log(abs(xi(i) - xi(1:i-1)))) ...
        + sum(log(abs(xi(i) - xi(i+1:end))));
    signum(i) = logical(mod(length(find((xi(i)-xi) < 0)), 2));
end
logW = logW - min(logW);
W = 1 ./ exp(logW);
W(signum) = -W(signum);
end