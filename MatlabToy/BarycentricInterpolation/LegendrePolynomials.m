function Pols = LegendrePolynomials(x, deg)
%
%   Pols = LegendrePolynomials(x, deg)
%
%   evaluates the L2(0,1)-orthonormal Legendre polynomials up to degree deg
%   at the locations specified by the column vector x
%
P0 = zeros(size(x));
P1 = ones(size(x));
Pols = zeros(length(x), deg + 1);
Pols(:,1) = P1;
for i = 1:deg
    Pols(:,i+1) = (2*i-1)/(i) * (2 * x - 1) .* P1 - (i-1)/(i) * P0;
    P0 = P1;
    P1 = Pols(:,i+1);
end
scal = sqrt((2 * [0:deg]' + 1));
Pols = Pols * spdiags(scal,0,deg+1,deg+1);
end
