function [Q] = TensorGaussLobatto(maxLvl,dim)
%%This function creates the tensorized nodes of the type Gauss Lobatto
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath("/Users/albe/Desktop/Work/Code(s)/SPQR/") % This is necessary to find the director of SPQR

GL = computeGL(maxLvl)';

InterpGL = cell(maxLvl+1,1);
for i = 0:maxLvl
    xi = GL(1:i+1);
    w = ones(size(GL(1:i+1)));
    InterpGL{i+1} = [xi;w];
end

[Q,~,~] = MXsparseQuadrature(maxLvl,dim,'TD', InterpGL, ones(1,dim));
% npts = size(Q);
% figure;
% scatter(Q(1,:),Q(2,:),'fill')
end



function LP = computeGL(Deg)
int = [-1 1];
En = zeros(Deg,1);
En(end) = En(end) + 1;

J = zeros(Deg);
for i = 1:Deg-1
    J(i,i+1) = i/sqrt(4*i.^2 - 1);
    J(i+1,i) =  i/sqrt(4*i.^2 - 1);
end

gamma = inv(J - int(1)*eye(Deg))*En;
mu = inv(J - int(2)*eye(Deg))*En;

alpha = (int(1)*mu(end) - int(2)*gamma(end))/(mu(end)-gamma(end));
beta = (int(2)-int(1))/(gamma(end)-mu(end));

J_fin = zeros(Deg+1);
for i = 1:Deg-1
    J_fin(i,i+1) = i/sqrt(4*i.^2 - 1);
    J_fin(i+1,i) = i/sqrt(4*i.^2 - 1);
end
J_fin(end,end-1) = sqrt(beta);
J_fin(end-1,end) = sqrt(beta);
J_fin(end,end) = alpha;

LP = eig(J_fin);
LP = .5*LP + .5;
for i = 0:Deg
    V(:,i+1) = LP.^i;
end

[~,~,P] = lu(V);

ord = P*[1:Deg+1]';
LP = LP(ord);
end
