 function Pols = Chebychev(x, deg)
P0 = ones(size(x));
P1 = 2*x - 1;
Pols = zeros(length(x), deg + 1);
Pols(:,1) = P0;
Pols(:,2) = P1;
for i = 2:deg
    Pols(:,i+1) = (4 * x - 2) .* P1 - P0;
    P0 = P1;
    P1 = Pols(:,i+1);
end
scal = sqrt((2 * [0:deg]' + 1));
Pols = Pols * spdiags(scal,0,deg+1,deg+1);
end
