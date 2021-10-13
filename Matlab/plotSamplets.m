clear all;
close all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x = rand(1,1000);
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
for i = 1:400
    figure(1);
    clf;
    samplet = Q(i,invI);
    surf(X,Y,reshape(samplet,sqrtN,sqrtN));
    %shading interp
    axis([0 1 0 1 -0.16 0.16])
    saveas(gcf,sprintf('samplet%d.png', i-1),'png');
end