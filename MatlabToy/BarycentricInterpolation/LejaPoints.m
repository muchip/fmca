function LP = LejaPoints(n)
%
% LP = LejaPoints(n)
%
% calculates the unweighted Leja points on [0,1] using Rolle's-theorem
% as suggested by Jens Oettershagen
%
leja = [0 1];
if (n > 500)
    error('The computation method is unstable for n > 500');
end
if n < 3
    LP = leja(1:n);
else
    LP = leja;
    for i = 1:n-2
        tval = inf;
        xval = inf;
        for j = 1:length(leja)-1
            [x, fval] = fminbnd(@(x) target(x, leja), leja(j), leja(j+1),...
                optimset('TolFun', 1e-16));
            if (fval < tval)
                tval = fval;
                xval = x;
            end
        end
        leja = sort([leja, xval]);
        LP = [LP, xval];
    end
end
end


function y = target(x, leja)
y = -abs(prod(x - leja));
end