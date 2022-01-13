clear all;
close all;
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N = 30000;
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
axis square;
axis tight;
end
plot(pts(:,1),pts(:,2),'r.')
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
[T, I] = MEXsampletBasis(pts', 5, 10);
toc
Q = sparse(T(:,1), T(:,2), T(:,3));
clear T;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Qerror = norm(Q * Q' - speye(size(Q)), 'fro') / norm(speye(size(Q)), 'fro')
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% clf
% colormap('parula')
% invI = I;
% invI(I) = [1:length(I)]'; 
% for i = 1:40
%     figure(1);
%     clf;
%     samplet = Q(i,invI);
%     scatter3(pts(:,1),pts(:,2),samplet, 2 * ones(size(samplet)),samplet)
%     view(-55,50)
%     axis([-22 28 -19 25 -0.06 0.06])
%     caxis([-0.06 0.06])
%     %axis tight
%     %axis off
%     %set(gca,'fontsize',20)
%     set(gca,'ZTick',[-0.06:0.03:0.06])
%     set(gca,'XTickLabel',[])
%     set(gca,'YTickLabel',[])
%     set(gca,'ZTickLabel',[])
%     saveas(gcf,sprintf('samplet%d.png', i-1),'png');
% end
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sPts = pts(I, :);
tic
N = size(sPts,1);
K = zeros(N,N);
for i = 1:N
     K(:,i) = exp(-1/9 * sqrt(sum((sPts - ones(N,1) * sPts(i,:)).^2,2))); 
     %K(:,i) = exp(-0.5 * sum((sPts - ones(N,1) * sPts(i,:)).^2,2)); 
end
toc;
spy(abs(K)>5e-4);
pause
tic
KSigma = Q * K * Q';
KSigma(find(abs(KSigma)<5e-4)) = 0;
KSigma = sparse(KSigma);
KSigma = 0.5 * (KSigma + KSigma');
spy(KSigma)
toc
p = dissect(KSigma);
L = chol(KSigma(p,p)+speye(N,N),'lower');
plotPattern(KSigma, 1000, 'Kmat_2D.eps');
plotPattern(KSigma(p,p), 1000, 'KmatND_2D.eps');
plotPattern(L, 1000, 'Lmat_2D.eps');
% toc
% tic
% L = pivotedCholesky(K, 1e-8);
% toc
% LSigma = Q * L;
% LSigma(find(abs(LSigma)<1e-8)) = 0;
% figure(1)
% spy(KSigma)
% figure(2)
% spy(LSigma)
