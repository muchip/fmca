function vismat(S)

% function vismat(S)
%
% Stellt die quadratische Matrix S im 
% logarithmischen Massstab graphisch dar.

S = log10(abs(S)+eps);
T = [S];
pcolor(T);
% shading interp;
colorbar;
colormap(jet);
axis('ij');
shading interp;
axis square