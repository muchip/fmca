clear all;
close all;
k = @(alpha, X, Y) exp(-alpha * abs(X-Y));
addpath('../');
dat = readtable('val_data_100_month.txt');
headers = dat.Properties.VariableNames;

flength = 100;
pts = [0:length(dat.ME5_BM7)-1]/length(dat.ME5_BM7);
[T, I, L] = MEXsampletBasis(pts(1:flength), 3);
invI = I;
invI(I) = [1:length(I)]';
assert(norm(invI - I) == 0);
T = sparse(T(:,1),T(:,2),T(:,3),flength,flength);

scale =  1;
[Xp, Yp] = meshgrid(pts(1:flength));
x = pts(1:flength+1);
[X, Y] = meshgrid(pts(1:flength),x);
for lvl = 0:max(L)
    Keval{lvl+1} = k(scale, X, Y);
    K{lvl+1} = k(scale, Xp, Yp);
    Ttilde{lvl+1} = K{lvl+1} \ T';
    Ttilde{lvl+1}(:, find(L ~= lvl)) = 0;
    pred{lvl+1} = Keval{lvl+1} * (Ttilde{lvl+1} * dat.ME5_BM7(1:flength));
end
plot(x', T *(pred{1} + pred{2} + pred{3} + pred{4} + pred{5}), 'b-', 'linewidth', 3);
hold on;
plot(pts(1:flength+1), dat.ME5_BM7(1:flength+1), 'r-', 'linewidth', 3);
plot(pts(flength+1:flength+1), dat.ME5_BM7(flength+1:flength+1), 'ko', 'linewidth', 3);