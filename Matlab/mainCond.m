clear all;
close all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sqrtN = 200;
N = sqrtN * sqrtN;
t = 2 * [0:(sqrtN-1)]/sqrtN - 1;
[X,Y] = meshgrid(t);
pts = [X(:)'; Y(:)'];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
[P, I] = MEXbivariateCompressor(pts, 3, 0.6, 1e-8);
toc
Keps = tril(sparse(P(:,1), P(:,2), P(:,3),N,N));
Keps = Keps + tril(Keps,-1)';
clear P;
v = diag(Keps);
M = spdiags(1./sqrt(v),0,N,N);
condest(Keps)
condest(M * Keps * M)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(1);
spy(Keps);
figure(2);
semilogy(v)
hold on
plot(diag(M * Keps * M))