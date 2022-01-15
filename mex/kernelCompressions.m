clear all;
close all;
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N = 20000;
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
[T, I] = MEXsampletBasis(pts', 4, 10);
toc
Q = sparse(T(:,1), T(:,2), T(:,3));
clear T;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Qerror = norm(Q * Q' - speye(size(Q)), 'fro') / norm(speye(size(Q)), 'fro')
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sPts = pts(I, :);
tic
N = size(sPts,1);
K = zeros(N,N);
for i = 1:N
    nrm = sqrt(sum((sPts - ones(N,1) * sPts(i,:)).^2,2));
    nrm2 = sum((0.5 * sPts + 0.5 * ones(N,1) * sPts(i,:)).^2,2);
     K(:,i) = nrm2 .* exp(-1/9 * sqrt(sum((sPts - ones(N,1) * sPts(i,:)).^2,2)));
     % K(:,i) = (sum(sPts .* (ones(N,1) * sPts(i,:)),2) + 0.1).^4;
     %K(:,i) = 1 - nrm.^2 ./ (nrm.^2 + 0.1);
     %K(:,i) = 0.05 ./ nrm .* sin(nrm / 0.05);
     %K(:,i) = exp(-0.5 * sum((sPts - ones(N,1) * sPts(i,:)).^2,2)); 
end
%K(1:N+1:end) = 1;
tic
KSigma = Q * K * Q';
KSigma(find(abs(KSigma)<1e-3)) = 0;
KSigma = sparse(KSigma);
KSigma = 0.5 * (KSigma + KSigma');
spy(KSigma)
toc