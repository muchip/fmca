function visMatrix(M, fignum,threshold)
figure(fignum);
clf;
M(find(abs(M)<threshold)) = 1e-16;
surf(abs(M));
shading interp;
set(gca,'colorscale','log')
view(0,-90);
axis square;
axis tight;
axis off;
colorbar;
end
