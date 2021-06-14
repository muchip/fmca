filename = 'defiant.txt';
dtilde = 2;

P = mainMEX(filename, dtilde);

eval(sprintf('pts = load(''%s'');', filename));

S = sparse(P(:,1)+1,P(:,2)+1,P(:,3));
figure(1)
plot3(pts(:,1),pts(:,2),pts(:,3),'b.');
figure(2)
spy(S)