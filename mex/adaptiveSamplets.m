clear all;
close all;
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N = 40000;
NS = 1;
% spiral data set
theta = sqrt(rand(N,1)) * 3 * pi;
psi = 8 * (2 * rand(N,1) -1);
r_a = 2 * theta + 1;
pts = [];
for i = 0:NS-1;
    data_a = [cos(pi - theta + 2 * pi * i / NS) .* r_a,...
            sin(pi - theta + 2 * pi * i / NS) .* r_a];
    x_a = data_a + 1 * psi .* rand(N,2);
    pts = [pts; x_a];
end
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
[T, I] = MEXsampletBasis(pts', 3, 10);
toc
Q = sparse(T(:,1), T(:,2), T(:,3));
clear T;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Qerror = norm(Q * Q' - speye(size(Q)), 'fro') / norm(speye(size(Q)), 'fro')
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sPts = pts(I, :);
rotp = sPts * [1/sqrt(2), 1/sqrt(2); -1/sqrt(2),1/sqrt(2)];
fun = exp(-0.05*abs(rotP(:,2)-2).^2).*(rotp(:,1) > -6)-0.5;
fun = max(fun,0);
scatter3(sPts(:,1),sPts(:,2),fun, 2 * ones(size(fun)),fun)
view(-55,50)