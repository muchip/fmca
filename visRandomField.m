clear all;
close all;
file = './Points/Halton1D.txt';
dtilde = 2;
rparam = 1e-1;
[P,I] = MEXcompressCov(file, dtilde);
S = sparse(P(:,1),P(:,2),P(:,3));
clear P;
S = 0.5 * (S + S');
v= 1./sqrt(diag(S));
invM = spdiags(1./v,0,size(S,1),size(S,2));
M = spdiags(v,0,size(S,1),size(S,2));
MSM = M * S * M;
p = dissect(MSM);
invp = p;
invp(p) = [1:length(p)]';
%Tp = MEXsampletTransform(file,dtilde,p);
%Tp = MEXinvSampletTransform(file,1,Tp);

%%
L = chol(MSM(p,p) + 10 * speye(size(MSM)));
L = L * invM;
%%
B = load(file);
x = randn(size(B,1),1);
y = L * x;
y = y(invp);
y = MEXinvSampletTransform(file,dtilde,y);
%test = B(I,1).^2 - B(I,2).^2;
figure(1);
clf;
plot(B(I,1),y,'b-')
%hp = patch(B(I,1), B(I,2), y, 'Marker', 's', 'MarkerFaceColor', 'flat', ...
%           'MarkerSize', 4, 'EdgeColor', 'none', 'FaceColor', 'none');
%axis square;
%axis tight;
%set(gca,'colorscale','log')
%view(0,90);
%colorbar;


