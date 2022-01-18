clear all;
close all;
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N = 80000;
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
f = exp(-0.05*abs(rotp(:,2)-3).^2).*(rotp(:,1) > -6)-0.5;
f = max(f,0);
scatter(sPts(:,1),sPts(:,2), 2 * ones(size(f)),f)
view(-90,90)
axis square
axis tight
axis off

ids = zeros(size(f));
Tf = Q * f;
for i=1:length(f)
    if (abs(Tf(i))>1e-4)
        ids(find(Q(i,:))) = ids(find(Q(i,:))) + 1;
    end
end
figure(2)
scatter(sPts(:,1),sPts(:,2), 4 * ones(size(f)),ids)
view(-90,90)
axis square
axis tight
axis off
