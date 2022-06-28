leja = [0 1];
xx = [0:0.0001:1];
for i = 1:200
    i
    tval = inf;
    xval = inf;
    for j = 1:length(leja)-1
        [x, fval] = fminbnd(@(x) target(x, leja), leja(j), leja(j+1));
        if (fval < tval)
            tval = fval;
            xval = x;
        end
    end
    leja = sort([leja, xval]);
    figure(1);
    clf;
    semilogy(leja,0,'ro')
    hold on;
    for i = 1:length(xx)
        yy(i) = -target(xx(i), leja) + eps;
    end
    semilogy(xx, yy,'b-')
    axis([0 1 0 0.1])
    pause
end

