function plotPattern(M, bins, filename)
n_size = size(M,1);
csize = floor(n_size/bins);
compS = [];
for i=0:bins-1
   for j=0:bins-1
        [I,J] = find(M(i*csize+1:(i+1)*csize,j*csize+1:(j+1)*csize) ~= 0);
        compS(i+1,j+1) = length(I) / csize() / csize;
   end
end
[X,Y] = meshgrid(1:bins);
index = find(compS > 0);
scatter(X(index),bins+1-Y(index),1.5,(compS(index)), 'filled')
colormap(flipud(copper));
set(gca,'colorscale','log')
axis equal;
axis off;
box on;
title(sprintf('bins: %d, n: %d, binsize: %d, nnz: %d',...
              bins, n_size, csize, nnz(M)));
set(gcf,'renderer','Painters');
saveas(gcf,filename,'epsc');
end