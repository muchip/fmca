clear all;
close all;
filename = 'porsche_small.txt';
dtilde = 6;

P = mainMEX(filename, dtilde);

eval(sprintf('pts = load(''%s'');', filename));

S = sparse(P(:,1),P(:,2),P(:,3));
clear P;
S = 0.5 * (S+S');
figure(1)
plot3(pts(:,1),pts(:,2),pts(:,3),'b.');
axis equal;
figure(2)
spy(S)
p = dissect(S);
figure(3)
spy(S(p,p))