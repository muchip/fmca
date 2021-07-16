function visSparsityPattern(S, fignum)
figure(fignum)
clf;
[I,J,K] = find(S);
hp = patch(I, J, K, ...
           'Marker', 's', 'MarkerFaceColor', 'flat', 'MarkerSize', 4, ...
           'EdgeColor', 'none', 'FaceColor', 'none');
axis square;
axis tight;
set(gca,'colorscale','log')
view(0,-90);
colorbar;
end