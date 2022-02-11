k1 = @(R) exp(-10 * R.^2);
k2 = @(R) exp(-R); 

k3 = @(X,Y) 0.5 * (1+sin(pi*(X+Y)).^2) .* k2(abs(X-Y));

[X,Y] = meshgrid ([0:0.001:1]);

K = k3(X,Y);

surf(X,Y,K)