function LP = OrderedCheb2(n)
%
% LP = LejaPoints(n)
pt = (cos(pi*(0:n-1)/(n-1)))';
pt = .5 * pt + .5;
%
% calculates the unweighted Leja points on [0,1] using Rolle's-theorem
% as suggested by Jens Oettershagen
%
leja = [pt(1) pt(end)];
if (n > 500)
    error('The computation method is unstable for n > 500');
end
if n < 3
    LP = sort(leja(1:n));
else
    LP = leja;
    for i = 1:n-2
        tval = inf;
        xval = inf;
        [fval,ind] = min(target(pt, leja));   
        if (fval < tval)
            tval = fval;
            xval = pt(ind);
        end
        leja = sort([leja, xval]);
        LP = [LP, xval];
    end
end
end


function y = target(x, leja)
y = -abs(prod(x - leja,2));
end