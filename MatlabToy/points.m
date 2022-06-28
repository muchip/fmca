%% % This file is doe to create some nice pictures %%%%%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
close all;
clc;
addpath('./BarycentricInterpolation/');
N = 6;
LP = Chebyshevs_second_pts(100);
% LP = Chebyshevs_first_pts(100);
% LP = LejaPoints(100);
% LP = GaussLobatto(99)';
% OC = (0:10)/10; 
% FLP = FastLeja(N+1);
% scatter(LP,zeros(1,11),'fill','SizeData',60);
% hold on
% scatter(C2,ones(1,11),'fill','SizeData',60);
% scatter(C1,ones(1,11)+1,'fill','SizeData',60);
% scatter(GL,ones(1,11)+2,'fill','SizeData',60);
% scatter(OC,ones(1,11)+3,'fill','SizeData',60);
% scatter(FLP,ones(1,11)+4,'fill','SizeData',60);
% grid on
% xlabel('Interval between 0 and 1','FontSize',14)
% set(gca,'YTick',[0 1 2 3 4 5] );
% set(gca,'YTickLabel',{'Leja points' 'Chebychev 2' 'Chebychev 1' 'LGL' 'Uniform distribution' 'Fast LP'},'FontSize',12 );
% set(gca,'YTickLabel',['Leja points'] );
% hold on
% print -depsc Figure1.eps

% LP = (1:100)/100;

% V1D = NewtonPolynomials(LP', N);
V1D = LegendrePolynomials(LP', N);
% V1D  = Chebychev(LP', N);
% V2D = MonomialPolynomials(LP', N);

figure;
% subplot(1,2,1);
for i = 1:N
    plot(LP,V1D(:,i),'LineWidth',2)
    hold on 
end
grid on;
legend({'\Pi_0','\Pi_1','\Pi_2','\Pi_3','\Pi_4','\Pi_5'},'NumColumns',3)
title('Legendre Polynomial','FontSize',14)
xlabel('X','FontSize',14)
ylabel('Y','FontSize',14)
hold off
% 
subplot(1,2,2);
for i = 1:N
    plot(LP,V2D(:,i),'LineWidth',2)
    hold on 
end
grid on;
legend({'M_0','M_1','M_2','M_3','M_4','M_5'},'NumColumns',3)
title('Monomials','FontSize',14)
xlabel('X','FontSize',14)
ylabel('Y','FontSize',14)












