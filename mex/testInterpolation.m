clear all;
close all;
[X, Y] = meshgrid([0:0.01:1]);
P = [X(:), Y(:)]';
dim = 5;
P = rand(dim,100000);
tic
[Xi, invV, eval] = MEXpolynomialInterpolator(P,  [1 1 1 1 1], 10);
toc
%cond(invV)
%plot3(Xi(1,:),Xi(2,:),Xi(3,:),'ro')
% 
% f = zeros(1,size(Xi,2));
% for i=1:size(Xi,2)
%     f(i) = exp(Xi(1,i)) * sin(0.1 * pi * Xi(2,i));
% end
% Interp = (f * invV') * eval;
% 
% Interp = reshape(Interp, size(X,1), size(X,2));
% F = zeros(size(X));
% for j = 1:size(X,2)
%     for i = 1:size(X,1)
%         F(i,j) = exp(X(i,j)) * sin(0.1 * pi * Y(i,j));
%     end
% end
% error = max(max(abs(F-Interp))/max(max(F)))
% 
% surf(X,Y,F)