clear all;
close all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sqrtN = 51;
N = sqrtN * sqrtN;
t = [0:(sqrtN-1)]/sqrtN;
[X,Y] = meshgrid(t);
pts = [X(:)'; Y(:)'];
pts = rand(2,4000);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
[P, I, T] = MEXbivariateCompressor(pts, 3, 0.8, 1e-6);
toc
Keps = sparse(P(:,1), P(:,2), P(:,3));
clear P;
Q = sparse(T(:,1), T(:,2), T(:,3));
clear T;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Qerror = norm(Q * Q' - speye(size(Q)), 'fro') / norm(speye(size(Q)), 'fro')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(1);
spy(Keps);
figure(2);
spy(Q);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
invI = I;
invI(I) = [1:length(I)]'; 
for i = 1:40
    figure(1);
    clf;
    samplet = Q(i,invI);
    %samplet = reshape(samplet,sqrtN,sqrtN);
    scatter3(pts(1,:)',pts(2,:),samplet(:), 10 * ones(size(samplet(:))), samplet(:));
   %plot(pts(1,:)',samplet(:),'k.');
    %shading interp
    colormap('jet')
    %axis([0 1 0 1 -0.1 0.1])
    axis off
    saveas(gcf,sprintf('samplet2D%d.png', i-1),'png');
end