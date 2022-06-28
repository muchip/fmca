function Pols = NewtonPolynomials(x, deg)
%
%   Pols = NewtonPolynomials(x, deg)
%
%   evaluates the Newton polynomials up to degree deg
%   at the locations specified by the column vector x 
%   with Horner's Method
%
P1 = ones(size(x));
Pols = zeros(length(x), deg + 1);
Pols(:,1) = P1;
res = P1;
for i = 2:deg+1
    Pols(:,i) = res.*(x - x(i-1));
    res = Pols(:,i);
end
end