clear all;
close all;
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pts = load('sigma.txt');
pts = pts(1:4000,:)';
%pts = randn(1,40000);
tic
[P, I, T] = MEXbivariateCompressor(pts, 3, 0.8, 1e-6);
toc
clear P;
Q = sparse(T(:,1), T(:,2), T(:,3));
clear T;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Qerror = norm(Q * Q' - speye(size(Q)), 'fro') / norm(speye(size(Q)), 'fro')
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% invI = I;
% invI(I) = [1:length(I)]'; 
% for i = 1:40
%     figure(1);
%     clf;
%     samplet = Q(i,invI);
%     scatter3(pts(1,:),pts(2,:),samplet, 4 * ones(size(samplet)),samplet)
%     axis([-9 7 -10 8.5 -0.05 0.05])
%     view(-15,50)
%     set(gca,'fontsize',20)
%     set(gca,'ZTick',[-0.05:0.01:0.05])
%     set(gca,'XTickLabel',[])
%     set(gca,'YTickLabel',[])
%     set(gca,'ZTickLabel',[])
%     pause(0.5)
%     saveas(gcf,sprintf('samplet%d.png', i-1),'png');
% end
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sPts = pts(:, I)';
m = [mean(sPts(:,1)), mean(sPts(:,2))]
cPts = sPts - ones(length(I),1) * m;
cov = cPts' * cPts / length(I)
plot(cPts(:,1),cPts(:,2),'r.')

