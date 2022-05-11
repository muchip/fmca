clear all;
close all;
addpath('../')
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N = 8000;
k = @(alpha, r) exp(-alpha * r);
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
[T, I] = MEXsampletBasis(pts', 4);
toc
T = sparse(T(:,1), T(:,2), T(:,3));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Qerror = norm(T * T' - speye(size(T)), 'fro') / norm(speye(size(T)), 'fro')
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sPts = pts(I, :);
f = sin(0.2*sPts(:,1)) .* cos(0.2*sPts(:,2));
Tf = T*f;
plot3(sPts(:,1), sPts(:,2), f,'r.');
tic
N = size(sPts,1);
K = zeros(N,N);
for i = 1:N
    r = sqrt(sum((sPts - ones(N,1) * sPts(i,:)).^2,2));
    K(:,i) = k(0.2, r);
end
tic
KSigma = T * K * T';
KSigma(find(abs(KSigma)<1e-6)) = 0;
KSigma = sparse(KSigma);
KSigma = 0.5 * (KSigma + KSigma');
Tsol = KSigma \ Tf;
sol = T' * Tsol;
toc
x = [-30:0.5:30];
[X,Y] = meshgrid(x);
Keval = zeros(length(Y(:)), N);
for i = 1:length(sPts)
    r = sqrt(sum(([X(:), Y(:)] - ones(length(Y(:)),1) * sPts(i,:)).^2,2));
    Keval(:,i) = k(0.2, r);
end
sol_eval = Keval * sol;
figure(1);
surf(X,Y,reshape(sol_eval, size(X,1), size(X,2)))
hold on;
plot3(sPts(:,1), sPts(:,2), f,'r.');
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(2)
semilogy(sort(diag(KSigma),'descend'))
I = find(diag(KSigma)>1);
size(I)
KSigma2 = KSigma(I,I);
Tsol_lp = KSigma(I,I) \ Tf(I);
sol_lp = zeros(size(Tsol));
sol_lp(I) = Tsol_lp;
sol_lp = T' * sol_lp;
sol_lp_eval = Keval * sol_lp;
figure(3)
surf(X,Y,reshape(sol_lp_eval, size(X,1), size(X,2)))
hold on;
plot3(sPts(:,1), sPts(:,2), f,'r.');
figure(4)
surf(X,Y,reshape(sol_lp_eval, size(X,1), size(X,2))-reshape(sol_eval, size(X,1), size(X,2)))
