function [X, Y, u] = potentialc (g, N, detT, T, D2g, rho, m)

n = length (rho);
y = ((0:m-1) / m);

gx = zeros (n+1, 1);
gy = zeros (n+1, 1);

gx(1:n) = g(:,1);
gx(n+1) = g(1,1);
gy(1:n) = g(:,2);
gy(n+1) = g(1,2);

X = kron (y, gx);
Y = kron (y, gy);
nm = length(X(:));
u = zeros (nm, 1);
Ox = kron (ones(n,1), X(:)') - kron (g(:,1), ones (1, nm));
Oy = kron (ones(n,1), Y(:)') - kron (g(:,2), ones (1, nm));
Nx = kron (N(:,1), ones (1, nm));
Ny = kron (N(:,2), ones (1, nm));
Rho = kron (rho(:,1), ones (1, nm));
u = sum (.5 * (log(Ox .* Ox + Oy .* Oy) .* Rho) / (-2 * pi * n));
u = reshape(u,size(X));
end