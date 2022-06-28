%% Naive implementation for Tensor Product in 2D %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Clean up the mess
clc;
clear;
close all;

%% Addpath
addpath('./BarycentricInterpolation/');
addpath('../Implementing/');

%% Degree of the polynomial 
Deg = 10;
N_pts = 1000;
my_pt = linspace(0,1,N_pts)'; %These are some random points I use for the model 

%% Choose the set of poitn that you find the most interesting
% xi = Chebyshevs_second_pts(Deg+1)';
% xi = Chebyshevs_first_pts(Deg+1)';
% xi = LejaPoints(Deg+1)';
% xi = GaussLobatto2(Deg)';
xi = GaussLobatto(Deg);
% xi = GL_pts(Deg+1)';
% xi = lglnodes(Deg)';
% xi = Chebyshevs_second_pts(Deg+1)';
% xi = OrderedCheb2(Deg+1)';
% xi = FastLeja(Deg+1)';

%% Barycentric weights for the Leja Points
W = barycentricWeights(xi);

%% Vandemoorde matrix
ell = LagrangePolynomial(my_pt, xi, W);
V = Vandermoonde_Matrix(xi, Deg); %Vandermoonde matrix for the nodes

c = (cond(V)); %% Condition number of the Vandermoonde matrix

% for i = 1:10
%     k(i,:) = c*i;
% end


% figure;
% surf(k)
% xlabel('Degrees','FontSize',16)
% ylabel('Dimensions','FontSize',16)
% zlabel('Condition number','FontSize',16)
% title('Tensor Product log-Condition of V with 1:6 degree and dimension 1:10','FontSize',16)
% colorbar
% grid on
% hold off



%% Change of base 
P = ell*V;

%% Evaluate the tensor product
f = @(X, Y) X.^5 .* Y.^4;
% f = @(X, Y) exp(-abs(X - Y)); %% Gaussian Kernel

[X,Y] = meshgrid(xi);

tic;
invV = inv(V);
C = invV * f(X,Y) * invV';

for i = 1:size(C,1)
    for j = size(C,1)-i+2:size(C,2)
        C(i,j)=0;
    end
end
toc;

%% Plot the graphs of the coefficient and final result
figure(1);
vismat(C)

[X,Y] = meshgrid(my_pt);

figure(2);   
surf(X,Y, abs(P * C * P' - f(X,Y)));
shading interp

figure(3);
surf(X,Y, f(X,Y));
shading interp




