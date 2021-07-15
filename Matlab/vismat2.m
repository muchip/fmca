function vismat2(S)

% function vismat2(S)
%
% Stellt wie vismat.m die Matrix S im log10-Massstab 
% graphisch dar,allerdings wird hier nicht so rechenintensiv,
% somit schneller, gearbeitet.

T = log10(abs(S)+eps);
n = min(min(T));
T = T-n*ones(size(T));
m = max(max(T))
image(256/m*T);
%colorbar;
colormap(jet);
axis('ij');
