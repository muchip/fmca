%% Implementation of the Sparse Grid Algorithm
clear;
close all;
clc;
%% Set up of the algorithm's inputs 
d = 4;
N = 15;
pts = sort(rand(d, 1000));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dim_tensor_degree = (N+1)^d;
dim_total_degree = nchoosek(N+d,d);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath('./BarycentricInterpolation/');
I = MXsparseIndexSet(d, N) + 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Subsample from quasi-random values
p = haltonset(d,'skip',10);
% p = Hammersley(2000,d);
% p = sobolset(d,'skip',10);
p=p(1:5000,:);  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Create the new Vandermonde matrix using tensor product
for i = 1:d
    VD{i} =  Chebychev(p(:,i),N); %% You need to change it as well at line 132
end
 
V = ones(size(p,1), size(I,2)); % size(I,2) in both
for j = 1:size(V,2)
  for k = 1:d
      V(:,j) = V(:,j) .* VD{k}(:,I(k,j));  % size(I(k,i),I(k,j)) in both
  end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% De Marchi implementation
[Q, R] = qr(V,0);
[V1,R1] = qr(Q,0);
w = V1'\ones(dim_total_degree,1);
ind = find(w(:,1) ~= 0);
p2 = p(ind,:);
V = V(ind,:);
SLP = p2';
invV = inv(V);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Continues the code
for k = 1:d
    peval{k} = Chebychev(pts(k,:)', N);
end
pevals = ones(size(pts,2), size(I,2));
for j = 1:size(I,2)
    for i = 1:size(pts,2)
        for k = 1:d
            pevals(i,j) = pevals(i,j) * peval{k}(i,I(k,j));
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
cond(Keval);
max(max(Keval - Kex))
shading interp

