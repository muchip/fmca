clear all;
close all;

N = 8001;
%k = @(x,y) exp(-20 * norm(x-y));
% k = @(x,y)  (1 + sqrt(3) * 20 * norm(x - y)) * exp(-20 * sqrt(3) * norm(x-y));
%x = [0:N-1] / N;
% C = zeros(N,N);
% for i=1:length(x)
%     for j  = 1:length(x)
%         C(i,j) = k(x(i),x(j));
%     end
% end
% cond(C)
% max(eig(C))
x = rand(N,2);
[P,I] = MEXcompressCovDebug(x,5);
S = tril(sparse(P(:,1),P(:,2),P(:,3)));
S = S + tril(S,-1)';
% figure(1)
% spy(S)
% min(abs(diag(S)))
% v = 1./sqrt(diag(S));
% P = spdiags(v,0,N,N);
% figure(2)
% semilogy(diag(S),'b.');
% hold on;
% semilogy(diag(P * S * P),'r.')
% [e] = eig(full(P * S * P));
% figure(3)
% semilogy(sort(e))
% condS = condest(S)
% condPSP = condest(P * S * P)
% mine = min(e)
% maxe = max(e)