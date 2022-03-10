N = 1000;
pts = rand(10000,1);
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
[T, I] = MEXsampletBasis(pts', 1, 10);
toc
Q = sparse(T(:,1), T(:,2), T(:,3));
clear T;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Qerror = norm(Q * Q' - speye(size(Q)), 'fro') / norm(speye(size(Q)), 'fro')

f = @(x) 1 ./ (1+25*x.^2);
x = pts(I);
fx=  f(x);

plot(x, Q * Q * fx)

% plot(x,f(x))
% Tf = Q*ones(size(x));
% wght = Tf(1);
% int = quad(f,0,1)
% Tf = Q*f(x);
% intS = 1/wght * Tf(1)
