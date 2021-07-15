function vismat(S)

% function vismat(S)
%
% Stellt die quadratische Matrix S im 
% logarithmischen Massstab graphisch dar.

S = log10(abs(S)+eps);
T = [S,zeros(size(S,1),1);zeros(1,size(S,2)+1)];
pcolor(T);
% shading interp;
colorbar;
colormap(jet);
axis('ij');
shading interp;