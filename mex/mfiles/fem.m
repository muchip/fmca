clear all;
close all;
addpath('../');
model = createpde(1);
importGeometry(model, 'MotherboardFragment1.stl');
%importGeometry(model, 'bunny.stl');
%geometryFromEdges(model, @cardg);
generateMesh(model, 'Hmax', 0.004,'GeometricOrder', 'linear');
pdemesh(model)
tic
[T, I] = MEXsampletBasis(model.Mesh.Nodes, 4);
toc
Q = sparse(T(:,1), T(:,2), T(:,3));
clear T;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Qerror = norm(Q * Q' - speye(size(Q)), 'fro') / norm(speye(size(Q)), 'fro')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%applyBoundaryCondition(model, 'Edge', 1:model.Geometry.NumEdges, 'u', 0);
applyBoundaryCondition(model, 'Face', 1:model.Geometry.NumFaces, 'u', 0);

cref = 1;
aref = 0;
fref = 1.;
tic
[A,F] = assempde(model, cref, aref, fref);
u = A\F;
toc
invI = I;
invI(I) = [1:length(I)]';
x = 0 * u;
x = full(Q(100,invI));
%pdeplot(model, "XYData", x, "ZData", x)
pdeplot3D(model, "ColorMapData", x)
tic
TATT = Q * A(I,I) * Q';
IT = inv(TATT);
IT(find(abs(IT)<1e-5)) = 0;
toc
display(sprintf('anz inverse: %d', round(nnz(IT) / size(IT, 1))));
display(sprintf('condition PA: %f', condest(IT * TATT)));
display(sprintf('condition A: %f', condest(A)));
inv_err = norm(IT * TATT - speye(size(A)),'fro') / norm(TATT,'fro');
display(sprintf('error inverse: %f', inv_err));