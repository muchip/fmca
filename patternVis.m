clear all
close all
diary('thisIsRediculous.txt')
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
maxNumCompThreads(1)
bins = 512;
[P,ctime] = MEXmainTestBenchmarkND3(17);
S = tril(sparse(P(:,1),P(:,2),P(:,3)));
clear P;
size(S)
S = S + 10 * speye(size(S));

%T = S + tril(S,-1)';
%condest(T)
%clear T;
whos
p = dissect(S);
L = chol(S(p,p),'lower'); 

S = S + tril(S,-1)';
whos
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SND = S(p,p);
n_size = size(S,1);
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
title(sprintf('bins: %d, n: %d, binsize: %d, nnz: %d',...
              bins, n_size, csize, nnz(S)));
set(gcf,'renderer','Painters');
saveas(gcf,'Spattern.eps','epsc');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
title(sprintf('bins: %d, n: %d, binsize: %d, nnz: %d',...
              bins, n_size, csize, nnz(L)));
set(gcf,'renderer','Painters');
saveas(gcf,'Lpattern.eps','epsc');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
title(sprintf('bins: %d, n: %d, binsize: %d, nnz: %d',...
              bins, n_size, csize, nnz(S)));
set(gcf,'renderer','Painters');
saveas(gcf,'SWpattern.eps','epsc');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



   
