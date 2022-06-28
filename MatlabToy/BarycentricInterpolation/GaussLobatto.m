%% Creation of the Gauss-Lobatto nodes
% Check here for some better definitions: https://math.stackexchange.com/questions/12160/roots-of-legendre-polynomial
function LP = GaussLobatto(Deg)
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


%% In case it comes to do it in C++: https://stackoverflow.com/questions/30228918/rewriting-matlab-eiga-b-generalized-eigenvalues-eigenvectors-to-c-c
LP = eig(J_fin);

for i = 0:Deg
    V(:,i+1) = LP.^i;
end

[~,~,P] = lu(V);

ord = P*[1:Deg+1]';
LP = LP(ord);

LP = .5*LP + .5;
end