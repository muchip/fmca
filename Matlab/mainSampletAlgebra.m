clear all;
close all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sqrtN = 200;
N = sqrtN * sqrtN;
%t = 2 * [0:(sqrtN-1)]/sqrtN - 1;
%[X,Y] = meshgrid(t);
%pts = [X(:)'; Y(:)'];
pts = haltonset(2, 'skip', 100);
pts = 2 * pts(1:N,:)' - 1;
%pts = 2 * rand(2,N) - 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
[P, I] = MEXbivariateCompressor(pts, 3, 0.8, 1e-6);
toc
Keps = sparse(P(:,1), P(:,2), P(:,3));
Keps = 0.5 * (Keps + Keps');
clear P;
%save('storedK.mat','Keps');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure(1);
% spy(Keps);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% p = dissect(Keps);
% figure(2)
% spy(Keps(p,p))
% L = chol(Keps(p,p), 'lower');
% figure(3)
% spy(L)
% invL = inv(L);
% figure(4)
% spy(invL);
% K2 = Keps * Keps';