clear all;
close all;
addpath('../');
addpath('../../examples/');
model = createpde(1);
%importGeometry(model, 'MotherboardFragment1.stl');
%importGeometry(model, 'bunny.stl');
geometryFromEdges(model, @cardg);
generateMesh(model, 'Hmax', 0.05,'GeometricOrder', 'linear');
pdemesh(model)
size(model.Mesh.Nodes)
applyBoundaryCondition(model, 'Edge', 1:model.Geometry.NumEdges, 'u', 0);
%applyBoundaryCondition(model, 'Face', 1:model.Geometry.NumFaces, 'u', 0);
cref = 1;
aref = 0;
fref = 1.;
tic
[A,F] = assempde(model, cref, aref, fref);
[i,j,s] = find(A);
[m,n] = size(A);
K = [[m, n, 0]; [i,j,s]];
fileID = fopen('A_card_005.txt','w');
fprintf(fileID,'%10d %10d %24.16f\n', m, n, 0);
for bla = 1:length(i);
    fprintf(fileID,'%10d %10d %24.16f\n',i(bla)-1, j(bla)-1, s(bla));
end
fclose(fileID);
fileID = fopen('P_card_005.txt','w');
P = model.Mesh.Nodes;
for bla = 1:size(P,2);
    fprintf(fileID,'%24.16f %24.16f\n', P(1,bla), P(2,bla));
      %  fprintf(fileID,'%24.16f %24.16f %24.16f\n', P(1,bla), P(2,bla), P(3,bla));
end
fclose(fileID);

