% Initialisation
clc, clear all, close all
diary NDoutput_FL.txt
LASTN = maxNumCompThreads(1)
T = 4;
N = 1000;
k = T/N;
theta = 0.5;
t_wem = [];
t_ND = [];
t_Chol = [];
nnz_S = [];
nnz_L = [];
n_size = [];
bins = 1280;
m=7
[P,F] = gennet(m);
n = size(F,1);
[i,j,l] = WEM(P,F);
S = sparse(j,i,l);
x = dwtkon(ones(n,1),m);
% fractional Laplace 
S = dwtMat(spdiags(idwtkon(full(S*x),m),0,n,n),m)-S;
% Massenmatrix
[i,j,l] = mass(P,F);
M = sparse(j,i,l);
S = M+k*theta*S;
p = dissect(S);
SND = S(p,p);
L = chol(SND);bvd
L = L';
n_size = size(S,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
csize = floor(n_size/bins)
compS = [];
for i=0:bins-1
   for j=0:bins-1
        [I,J] = find(SND(i*csize+1:(i+1)*csize,j*csize+1:(j+1)*csize) ~= 0);
        compS(i+1,j+1) = length(I) / csize() / csize;
   end
end
figure(1)
clf;
[X,Y] = meshgrid(1:bins);
index = find(compS > 0);
scatter(X(index),bins+1-Y(index),1.5,(compS(index)), 'filled')
c = gray;
c = flipud(c(1:200,:));
colormap(c);
axis equal;
axis off;
box on;
title(sprintf('bins: %d, n: %d, binsize: %d, nnz: %d', bins, n_size, csize, nnz(S)));
saveas(gcf,'Spattern.eps','epsc');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
compL = [];
for i=0:bins-1
   for j=0:bins-1
        [I,J] = find(L(i*csize+1:(i+1)*csize,j*csize+1:(j+1)*csize) ~= 0);
        compL(i+1,j+1) = length(I) / csize() / csize;
   end
end
figure(2)
clf;
[X,Y] = meshgrid(1:bins);
index = find(compL > 0);
scatter(X(index),bins+1-Y(index),1.5,(compL(index)), 'filled')
c = gray;
c = flipud(c(1:200,:));
colormap(c);
axis equal;
axis off;
box on;
title(sprintf('bins: %d, n: %d, binsize: %d, nnz: %d', bins, n_size, csize, nnz(L)));
saveas(gcf,'Lpattern.eps','epsc');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
compS = [];
for i=0:bins-1
   for j=0:bins-1
        [I,J] = find(S(i*csize+1:(i+1)*csize,j*csize+1:(j+1)*csize) ~= 0);
        compS(i+1,j+1) = length(I) / csize() / csize;
   end
end
figure(3)
clf;
[X,Y] = meshgrid(1:bins);
index = find(compS > 0);
scatter(X(index),bins+1-Y(index),1.5,(compS(index)), 'filled')
c = gray;
c = flipud(c(1:200,:));
colormap(c);
axis equal;
axis off;
box on;
title(sprintf('bins: %d, n: %d, binsize: %d, nnz: %d', bins, n_size, csize, nnz(S)));
saveas(gcf,'SWpattern.eps','epsc');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





