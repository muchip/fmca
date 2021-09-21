
clear all;
close all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sqrtN = 100;
N = sqrtN * sqrtN;
%t = 2 * [0:(sqrtN-1)]/sqrtN - 1;
%[X,Y] = meshgrid(t);
%pts = [X(:)'; Y(:)'];
pts = haltonset(2, 'skip', 100);
pts = 2 * pts(1:N,:)' - 1;
%pts = 2 * rand(2,N) - 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
[P, I, T] = MEXbivariateCompressor(pts, 3, 0.6, 1e-6);
toc
Keps = sparse(P(:,1), P(:,2), P(:,3));
clear P;
Q = sparse(T(:,1), T(:,2), T(:,3));
clear T;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Qerror = norm(Q * Q' - speye(size(Q)), 'fro') / norm(speye(size(Q)), 'fro')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(1);
spy(Keps);
figure(2);
spy(Q);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sPts = pts(:, I)';
tic
K = zeros(N,N);
for i = 1:N
    K(:,i) = exp(-10 * sqrt(sum((sPts - ones(N,1) * sPts(i,:)).^2,2))); 
end
toc;
K = Q * K * Q';
compressionError = norm(Keps - K, 'fro') / norm(K, 'fro')