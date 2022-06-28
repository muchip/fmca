clc;
clear;
close all;
addpath("/Users/albe/Desktop/Work/Code(s)/SPQR/") %This is necessary to find the director of SPQR

maxLvl = 12;
dim = 4;

chebnodes = @(n) 0.5 - 0.5 * cos((2*[1:n]-1) / 2 / n * pi);
chebext = @(n) (-0.5 * cos (pi * [0:n-1] / (n - 1 + (n==1))) + 0.5) * (n > 1) + 0.5 * (n==1);
leja = computeLeja(maxLvl+1);
InterpCheb = cell(maxLvl+1,1);
for i = 0:2 * maxLvl
    xi = chebnodes(i+1);
    w = ones(size(xi));
    InterpCheb{i+1} = [xi;w];
end

InterpLeja = cell(maxLvl+1,1);
for i = 0:maxLvl
    xi = leja(1:i+1);
    w = ones(size(leja(1:i+1)));
    InterpLeja{i+1} = [xi;w];
end


[Q,W,sort] = MXsparseQuadrature(maxLvl,dim,'TD', InterpLeja, ones(1,dim));
% npts = size(Q)
% figure;
scatter(Q(1,:),Q(2,:),'fill')

function trueleja = computeLeja(n)
leja = [0 1];
trueleja = leja;
for i = 1:n
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
    trueleja = [trueleja, xval];
end
end


function y = target(x, leja)
 y = -abs(prod(x - leja));
end