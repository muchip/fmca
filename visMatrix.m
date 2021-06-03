function visMatrix(M, fignum)
figure(fignum);
clf;
surf(M);
shading interp;
view(0,-90);
axis square;
axis tight;
axis off;
end
