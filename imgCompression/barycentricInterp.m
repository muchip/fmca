clear all;
close all;
n = 4;
neval = 100;
weight = 2 * pi / (2 * n + 2);

x = [0:neval-1]/(neval-1);
f = @(x) 1 ./ (1 + 100*(x-0.5).^2);

xi = 0.5 * (cos(weight * ([0:n] + 0.5)) + 1);
dat = f(xi);
w = sin(weight * ([0:n] + 0.5));
w(2:2:end) = -w(2:2:end);
xi'
w'
eval = 1./(kron(ones(1, n + 1), x') - kron(xi, ones(neval,1)));
den = eval * w';
num = eval * (w .* dat)';
figure(1)
plot(xi,0,'ro')
hold on;
plot(x, num ./ den, 'b-')
hold on;
plot(x,f(x),'k-')