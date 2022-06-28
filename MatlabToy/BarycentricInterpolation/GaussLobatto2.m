%% Creation of the Gauss-Lobatto nodes 
function LP = GaussLobatto2(Deg)
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

% for i = 0:Deg
%     V(:,i+1) = LP.^i;
% end
% 
% [~,~,P] = lu(V);
% 
% ord = P*[1:Deg+1]';
% LP = LP(ord);

pt = .5*LP + .5;
LP = 0;

leja = [pt(1) pt(end)];
if (Deg > 1000000)
    error('The computation method is unstable for n > 500');
end
if Deg < 3
    LP = sort(leja(1:Deg));
else
    LP = leja;
    for i = 1:Deg-1
        tval = inf;
        xval = inf;
        [fval,ind] = min(target(pt, leja));   
        if (fval < tval)
            tval = fval;
            xval = pt(ind);
        end
        leja = sort([leja, xval]);
        LP = [LP, xval];
    end
end
end

function y = target(x, leja)
y = -abs(prod(x - leja,2));
end
