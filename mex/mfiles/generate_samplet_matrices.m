clear all;
close all;
addpath('../');
model = createpde(1);
%importGeometry(model, 'MotherboardFragment1.stl');
%importGeometry(model, 'bunny.stl');
geometryFromEdges(model, @cardg);
generateMesh(model, 'Hmax', 0.09,'GeometricOrder', 'linear');
pdemesh(model);
P = haltonset(2,'skip',100);
P = P(1:4000,:)';
P = 2 * P-1;
[K,I] = MEXsampletCompressor(P, 4, 1, 6, 0);
K = sparse(K(:,1), K(:,2), K(:,3), size(P,2),size(P,2));
K = triu(K) + triu(K,1)' + speye(size(K));
nnz(K) / size(K,2)
tic
Kinv = inv(K);
toc
bla = Kinv;
bla(find(~K)) = 0;
bla = sparse(bla);
norm(bla * K - eye(size(K)),'fro')/norm(K,'fro')
% K = unique(K,'rows');
%  fileID = fopen('A_3D_01.txt','w');
%  fprintf(fileID,'%d %d %d\n', size(K,1), length(I), length(I));
%  for bla = 1:size(K,1);
%      fprintf(fileID,'%d %d %20.16f\n',K(bla,1)-1, K(bla,2)-1, K(bla,3));
%  end
%  fclose(fileID);
