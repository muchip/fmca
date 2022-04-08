clear all;
close all;
% parameters
alpha = 20;
n_samples = 20;
% define kernel
k = @(alpha, X, Y) exp(-alpha * abs(X-Y).^2);
addpath('../');
dat = readtable('val_data_100_month.txt');

headers = dat.Properties.VariableNames;
pred = [];
mean_yT = [];
data = dat.ME10_BM6;
for lower = 1:length(data)-n_samples-2
    window = [lower, lower+n_samples-1];
    xT = [window(1):window(2)];
    yT = data(window(1):window(2));
    mean_yT(lower) = mean(yT);
    [X,Y] = meshgrid(xT);
    K = k(alpha, X, Y) + 0.000001 * eye(n_samples, n_samples);
    L = pivotedCholesky(K,1e-9);
    LTL = L' * L;
    y = L*(LTL\(LTL\(L'*yT)));
    [X,Y] = meshgrid(xT,window(2)+1);
    KevalT = k(alpha, X, Y);
    pred(lower) = KevalT * y;
%     x = [window(1):window(2)+2];
%     xeval =[window(1):0.1:window(2)+2];
%     [X,Y] = meshgrid(xT,xeval);
%     KevalT = k(alpha, X, Y);
%     fbar = KevalT * y;
%     Kupdate = diag(k(alpha,xeval',xeval')) - KevalT * (L*(LTL\(LTL\(L'*KevalT'))));
%     Kupdate(find(Kupdate < 0)) =0;
%     fvar = 0;
%     figure(1)
%     hold on;
%     fmstd = fbar - 3 * sqrt(diag(Kupdate));
%     fpstd = fbar + 3 * sqrt(diag(Kupdate));
%     x2 = [xeval, fliplr(xeval)];
%     inBetween = [fmstd', fliplr(fpstd')];
%     fill(x2, inBetween, 'g','edgealpha',0,'facealpha',0.5);
%     plot(x, dat.ME5_BM7(window(1):window(2)+2),'b');
%     plot(xeval, fbar,'r','linewidth',1)
%     plot(x, dat.ME5_BM7(window(1):window(2)+2),'bo','linewidth',2);
%     plot(x(end-1:end), dat.ME5_BM7(window(2)+1:window(2)+2),'ks','linewidth',2,'markersize',10);
%     plot(x, mean_yT * ones(size(x)),'m-','linewidth',1);
%     pause(1)
end
figure(1);
hold on;
data2 = data(1:length(dat.ME5_BM7)-n_samples-2);
plot(data2,'k-');
plot(mean_yT,'m-');
plot(pred,'g-');
figure(2);
hold on;
%plot(abs(mean_yT'-data),'k-','linewidth',2);
plot(abs(pred'-data2)-abs(mean_yT'-data2),'m-');
plot(zeros(size(pred)),'k-')
hits = length(find((abs(pred'-data2)-abs(mean_yT'-data2) <= 0)))/length(abs(pred'-data2))


