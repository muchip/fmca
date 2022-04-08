clear all;
close all;
n = 40;
e = ones(n,1);
threshold = 1e-2;

T = spdiags([-e, 2*e, -e],-1:1,n,n);
I = speye(n);

TT = kron(T,I) + kron(I,T);
II = kron(I,I);
spy(TT)


X = 0.25 * II;

for i=1:20
    X = X * (2 * II - TT * X);
    X(find(abs(X) < threshold)) = 0;
    X = sparse(X);
    err(i) = norm(TT * X - II,'fro') / n
end
semilogy(err)