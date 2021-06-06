function visSparsityPattern(P, fignum)
figure(fignum)
clf;
hp = patch(P(:,1), P(:,2), P(:,3), ...
           'Marker', 's', 'MarkerFaceColor', 'flat', 'MarkerSize', 4, ...
           'EdgeColor', 'none', 'FaceColor', 'none');
axis square;
axis tight;
set(gca,'colorscale','log')
view(0,-90);
colorbar;
end