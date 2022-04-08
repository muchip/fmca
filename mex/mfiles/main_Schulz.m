clear all;
close all;
addpath('../');
N = 10000;
P = randn(2, N);
figure(1)
plot(P(1,:), P(2,:), 'r.');
[K,I] = MEXsampletCompressor(P, 3, 0.8, 4, 1e-6);
K = sparse(K(:,1), K(:,2), K(:,3), N, N);
K = K + triu(K,1)';
[pI, pJ] = find(K ~= 0);
nnz(K)
tic
L = K * K;
toc
tic
L2 = K' * K;
toc
display('onpattern')
tic
L3 = K;
ret = zeros(size(pi));
parfor i=1:length(pI)
ret(i) = sum(K(:,pI(i)) .* K(:, pJ(i)));
end
toc
X = 1e-2 * spdiags(1./(diag(K)+1e-6), 0, N, N);
I = speye(size(K));

for i=1:20
    X = X' * (2 * I - K' * X);
    X(find(abs(X) < 1e-4)) = 0;
    X = 0.5 * (X + X');
    X = sparse(X);
    err(i) = norm(K * X - I,'fro') / sqrt(N);
end
