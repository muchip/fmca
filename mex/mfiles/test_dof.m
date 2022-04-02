clear all;
close all;
model = createpde(1);
geometryFromEdges(model, @cardg);
generateMesh(model, 'Hmax', 0.1,'GeometricOrder', 'linear');
pdemesh(model)
P=model.Mesh.Nodes;
hold on;
plot(P(1,1), P(2,1),'go','linewidth',4)
x = zeros(size(P,2),1);
x(1) = 1;
x(40)  = 1;
pdeplot(model, "XYData", x)
hold on;
plot(P(1,1), P(2,1),'go','linewidth',4)
plot(P(1,40), P(2,40),'go','linewidth',4)