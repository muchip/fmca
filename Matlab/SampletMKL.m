clear all;
close all;
N = 4000;
thresh = 1e-5;
pts = rand(2, N);
K1 = @(x, y) exp(-norm(x - y));
K2 = @(x, y)(1 + sqrt(3) * norm(x - y)) * exp(-sqrt(3) * norm(x - y));
K3 = @(x, y) exp(-norm(x - y).^ 2);

for i = 1:N
  for j = 1:N
    A(i, j) = K1(pts(:,i), pts(:,j));
    B(i, j) = K2(pts( :, i), pts( :, j));
    C(i, j) = K3(pts( :, i), pts( :, j));
  end
end
tic
[~, I, T] = MEXbivariateCompressor(pts, 3, 10, 1e-1);
toc
Q = sparse(T(:,1), T(:,2), T(:,3));
clear T;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Qerror = norm(Q * Q' - speye(size(Q)), 'fro') / norm(speye(size(Q)), 'fro')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
TA = Q * A(I,I) * Q';
TB = Q * B(I,I) * Q';
TC = Q * C(I,I) * Q';
TA(find(abs(TA) < thresh)) = 0;
TB(find(abs(TB) < thresh)) = 0;
TC(find(abs(TC) < thresh)) = 0;
figure(1)
vismat(TA)
figure(2)
vismat(TB)
figure(3)
vismat(TC)