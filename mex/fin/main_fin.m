clear all;
close all;
k = @(X, Y) exp(-10 * abs(X-Y));
addpath('../');
dat = readtable('val_data_100_month.txt');
headers = dat.Properties.VariableNames;

x = [0:length(dat.ME5_BM7)-1]/length(dat.ME5_BM7);
[T, I] = MEXsampletBasis(x, 3);
T = sparse(T(:,1),T(:,2),T(:,3),length(dat.ME5_BM7),length(dat.ME5_BM7));
ts = dat.Mkt_RF(I);
ts(400:end) = 0;
vec = T * ts;
vec(200:end) = 0;
samplet = T' * vec;
plot(x,dat.Mkt_RF(I),'r-')
hold on;
plot(x,samplet,'b-','linewidth',2);
plot(x,ts,'g-')
ts = dat.Mkt_RF(I);
ts(401:end) = 0;
vec = T * ts;
vec(200:end) = 0;
samplet = T' * vec;
plot(x,samplet,'m-','linewidth',2);

%plot(x,dat.ME5_BM7)
