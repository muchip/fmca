clear all;
close all;

%N = 8001;
sqrtN = 100;
N = sqrtN * sqrtN;
t = [0:(sqrtN-1)]/sqrtN;
[X,Y] = meshgrid(t);
%P = haltonset(2,'skip', 100);
x = [X(:), Y(:)];
x = rand(N,2);
[P,T,I] = MEXcompressCovDebug(x,4);
S = tril(sparse(P(:,1),P(:,2),P(:,3)));
clear P;
S = tril(S) + tril(S,-1)';
Q = sparse(T(:,1),T(:,2),T(:,3));
clear T;

sortx = x(I,:);
f = sin(pi * 3 * sortx(:,1)) .* cos(pi * 2 * sortx(:,2));
fw = Q * f;

D = diag(S);
[d,p] = sort(D,'descend');
p = p(1:1000);
%p = 1:1000;
us = zeros(size(f));
us(p) = S(p,p)\fw(p);



% S2 = S(1:1000,1:1000);
% us = S2 \ fw(1:1000);
% us = [us; zeros(N-1000,1)];
u = Q' * us;

Kx = zeros(N,N);
for i = 1:N
    Kx(:,i) = exp(-20 * sqrt(sum((sortx - ones(N,1) * sortx(i,:)).^2,2))); 
end
f2 = Kx * u;
plot3(sortx(:,1), sortx(:,2), abs(f-f2),'.');     
% S(find(abs(S) < 1e-6))=0;
% p = dissect(S);
% tic
% L = chol(S(p,p)+10*speye(size(S)),'lower');
% toc
% figure(1);
% spy(L)
% tic
% invL = inv(L);
% toc
% figure(2)
% spy(invL)
