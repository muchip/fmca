function [Q] = TensorLeja(maxLvl,dim)
%%This function creates the tensorized nodes of the type Leja
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath("/Users/albe/Desktop/Work/Code(s)/SPQR/") %This is necessary to find the director of SPQR

leja_pts = computeLeja(maxLvl+1);
InterpLeja = cell(maxLvl+1,1);
for k = 0:maxLvl
    xi = leja_pts(1:k+1);
    w = ones(size(leja_pts(1:k+1)));
    InterpLeja{k+1} = [xi;w];
end


[Q,~,~] = MXsparseQuadrature(maxLvl,dim,'TD', InterpLeja, ones(1,dim));
% npts = size(Q)
% figure;
% scatter(Q(1,:),Q(2,:),'fill')

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
end