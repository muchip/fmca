clear all;
close all;
addpath('../');
model = createpde(1);
%importGeometry(model, 'MotherboardFragment1.stl');
%importGeometry(model, 'bunny.stl');
geometryFromEdges(model, @cardg);
generateMesh(model, 'Hmax', 0.03,'GeometricOrder', 'linear');
%pdemesh(model);
[K,I] = MEXsampletCompressor(model.Mesh.Nodes, 3, 0.8, 4, 1e-6);
%K = sparse(K(:,1), K(:,2), K(:,3));
K = unique(K,'rows');
 fileID = fopen('A_2D_003.txt','w');
 fprintf(fileID,'%d %d %d\n', size(K,1), length(I), length(I));
 for bla = 1:size(K,1);
     fprintf(fileID,'%d %d %20.16f\n',K(bla,1)-1, K(bla,2)-1, K(bla,3));
 end
 fclose(fileID);
