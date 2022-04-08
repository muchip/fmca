function vismat(S)

% function vismat(S)
%
% Stellt die quadratische Matrix S im 
% logarithmischen Massstab graphisch dar.

%S = log(abs(S)+eps);
T = [S];
pcolor(T);
colorbar;
%colormap(jet);
axis('ij');
shading interp;
axis equal;
axis off;