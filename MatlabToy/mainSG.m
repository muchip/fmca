%% Implementation of the Sparse Grid Algorithm
clear;
close all;
clc;
%% Set up of the algorithm's inputs 
d = 2;
for N = 1:6
pts = sort(rand(d, 1000));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% dim_tensor_degree = (N+1)^d;
dim_total_degree = nchoosek(N+d,d);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath('./BarycentricInterpolation/');
I = MXsparseIndexSet(d, N) + 1;
% LP = Chebyshevs_second_pts(N+1);
% LP = Chebyshevs_first_pts(N+1);
% LP = LejaPoints(N+1);
% LP = GL_pts(N+1)';
% LP = GaussLobatto(N)';
% p = Newpts(30,d)';
% LP = Chebyshevs_second_pts(N+1)
% LP = OrderedCheb2(N+1);
% LP = FastLeja(N+1);
% SLP = LP(I);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% V1D = NewtonPolynomials(LP', N);
% V1D = LegendrePolynomials(LP', N);
% V1D  = Chebychev(LP', N);
% V1D = MonomialPolynomials(LP', N);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% V = ones(size(I,2), size(I,2)); % size(I,2) in both
% for j = 1:size(V,2)
%     for i = 1:size(V,1) % size(V,2) in both
%         for k = 1:d
%             V(i,j) = V(i,j) * V1D(I(k,i),I(k,j));  % size(I(k,i),I(k,j)) in both
%         end
%     end
% end
% c(d,N) = log(cond(V));
% end
% end

% figure;
% surf(c)
% xlabel('Degrees','FontSize',16)
% ylabel('Dimensions','FontSize',16)
% zlabel('Condition number','FontSize',16)
% title('LGL-Newton log-Condition of V with 1:6 degree and dimension 1:10','FontSize',16)
% colorbar
% grid on
% hold off

init = 1;
for z = init:90
%% Subsample from quasi-random values
p = haltonset(d,'skip',10);
% p = Hammersley(2000,d);
% p = sobolset(d,'skip',10);
p=p(1:z*100,:);  
%% Create the new Vandermonde matrix using tensor product
for i = 1:d
    VD{i} =  LegendrePolynomials(p(:,i),N); %% You need to change it as well at line 132
end
 
V = ones(size(p,1), size(I,2)); % size(I,2) in both
for j = 1:size(V,2)
  for k = 1:d
      V(:,j) = V(:,j) .* VD{k}(:,I(k,j));  % size(I(k,i),I(k,j)) in both
  end
end

%% De Marchi implementation
[Q, ~] = qr(V,0);
[V1,~] = qr(Q,0);
w = V1'\ones(dim_total_degree,1);
ind = find(w(:,1) ~= 0);

p2 = p(ind,:);
V = V(ind,:);

c(z,N) = cond(V);
end
% d
end

figure;
for i = 1:6
    plot(1:90,c(:,i),'LineWidth',2)  
    hold on
end
xlabel('# of points (in hundreds)','FontSize',16)
ylabel('Condition number','FontSize',16)
legend('Deg = 1','Deg = 2','Deg = 3','Deg = 4','Deg = 5','Deg = 6')
grid on
hold off

% figure;
% surf(c)
% xlabel('Degrees','FontSize',16)
% ylabel('Dimensions','FontSize',16)
% zlabel('Condition number','FontSize',16)
% title('Legendre-Halton log-condition number of V with 1:6 degree and dimension 1:10','FontSize',16)
% colorbar
% grid on
% hold off

% 
% SLP = p2'
invV = inv(V);


%% Continues the code
tic;
for k = 1:d
    peval{k} = Chebychev(pts(k,:)', N);
end
toc;
tic;
pevals = ones(size(pts,2), size(I,2));
for j = 1:size(I,2)
    for i = 1:size(pts,2)
        for k = 1:d
            pevals(i,j) = pevals(i,j) * peval{k}(i,I(k,j));
        end
    end
end
toc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
K = zeros(size(SLP,2),size(SLP,2));
for i = 1:size(SLP,2)
     K(:,i) = exp(-0.5 * (sum((SLP' - ones(size(SLP,2),1) * SLP(:,i)').^2,2))); 
end
Kex = zeros(size(pts,2),size(pts,2));
for i = 1:size(pts,2)
     Kex(:,i) = exp(-0.5 * (sum((pts' - ones(size(pts,2),1) * pts(:,i)').^2,2))); 
end
Ktilde = invV * K * invV';
Keval = pevals * Ktilde * pevals';
surf(Keval - Kex)
toc;
cond(Keval)
shading interp

