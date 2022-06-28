%% Compression rate
clc;
close all;
clear;

p{1} = xlsread('compression/1T.xlsx');
p{2} = xlsread('compression/2T.xlsx');
p{3} = xlsread('compression/3T.xlsx');
p{4} = xlsread('compression/4T.xlsx');
p{5} = xlsread('compression/5T.xlsx');
% p = xlsread('compression/2.xlsx');


for i = 3:5
    figure;
    hold on
    for j = 1:2
        plot(1:size(p{i},1),p{i}(:,j),'-x','LineWidth',2);
    end
    grid on
    set (gca,'xtick',[1,2,3,4]);
    set (gca, 'xticklabels', {'10^2';'10^3';'10^4';'10^5'});
    xlabel('Matrix size','FontSize',16)
    ylabel('Set-up time','FontSize',16)
    title('H^2 matrix compression','FontSize',16)
    legend('Chebychev Leja','Chebychev AFP')
    hold off
end

for i = 1:2
    figure;
    hold on
    for j = 1:2
        plot(1:size(p{i},1),p{i}(:,j),'-x','LineWidth',2);
    end
    grid on
    set (gca,'xtick',[1,2,3,4,5]);
    set (gca, 'xticklabels', {'10^2';'10^3';'10^4';'10^5';'10^6'});
    xlabel('Matrix size','FontSize',16)
    ylabel('Set-up time','FontSize',16)
    title('H^2 matrix compression','FontSize',16)
    legend('Chebychev Leja','Chebychev AFP')
    hold off
end


