%% Flobenius norm error for the Tensor Product %%%%%%%%%%%%%%%%%
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
Deg = 5;
N_pts = 100;
my_pt = linspace(0,.25,N_pts)'; %These are some random points I use for the model 

%% Choose set of Points
% xi = Chebyshevs_second_pts(Deg+1)';
% xi = Chebyshevs_first_pts(Deg+1)';
% xi = sort(LejaPoints(Deg+1)');
% xi = sort(GL_pts(Deg+1));
% xi = sort(GaussLobatto(Deg));
% xi = sort(lglnodes(Deg)); %This is the 
% xi = Chebyshevs_second_pts(Deg+1)';
% xi = OrderedCheb2(Deg+1)';
xi = FastLeja(Deg+1)';
xi = 1/4*xi; 

%% Barycentric weights for the Leja Points
W = barycentricWeights(xi);

%% Vandemoorde matrix
ell = LagrangePolynomial(my_pt, xi, W);
V = Vandermoonde_Matrix(xi, Deg); %Vandermoonde matrix for the nodes

%% Change of base 
P = ell*V;

%% Evaluate the tensor product
% f = @(X, Y) X.^5 .* Y.^4;
f = @(X, Y) exp(-abs(X - Y));

% [X,Y] = meshgrid(xi);
[X,Y] = meshgrid(xi, 1-xi);

% figure(1);
% surf(X,Y,f(X,Y))

% tic;
invV = inv(V);
C = invV * f(X,Y) * invV';

%% Plot the graphs of the coefficient and final result
% figure(1);
% vismat(C)


[X,Y] = meshgrid(linspace(0,0.25,N_pts), 1-linspace(0,.25,N_pts));
% [X,Y] = meshgrid(my_pt);


figure(2);   
surf(X,Y, P * C * P' - f(X,Y));
shading interp
xlabel('X','FontSize',14)
ylabel('Y', 'FontSize',14)
zlabel('Error', 'FontSize',14)

% figure(3);
% surf(X,Y, f(X,Y));
% shading interp
% 
% figure(4)
% surf(X,Y, P * C * P')
% shading interp


res = P * C * P';
z = f(X,Y);

%% Error with the Frobenius Norm
% figure(5);
for k = 1:100
for i = randperm(N_pts,N_pts/2-10)
    for j = randperm(N_pts,1)
%         scatter3(i,j,norm(res(i,j) - z(i,j),'fro')/norm(z(i,j),'fro'),'fill')
        tm(i+j-1) = norm(res(i,j) - z(i,j),'fro')/norm(z(i,j));
%         hold on
    end
end
avg(k) = mean(tm);
mas(k) = max(tm);
end
AVG = mean(avg)
MAS = mean(mas)

% hold off
% xlabel('Random x-coordinate index','FontSize',14)
% ylabel('Random y-coordinate index', 'FontSize',14)
% zlabel('Error', 'FontSize',14)
% title('Error with the Frobenius Norm (Tensor Product) Leja','FontSize',14)
% grid on

