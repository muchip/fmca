function Pols = MonomialPolynomials(x, deg)
%
%   Pols = MonomialPolynomials(x, deg)
%
%
%   evaluates the monomial polynomials up to degree deg
%   at the locations specified by the column vector x
%
P1 = ones(size(x));
Pols = zeros(length(x), deg + 1);
Pols(:,1) = P1;
for i = 2:deg+1
    Pols(:,i) = x.^i;
end
end