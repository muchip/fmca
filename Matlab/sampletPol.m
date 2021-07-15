clear all;
close all;

sqrtN = 100;
N = sqrtN * sqrtN;
t = [0:(sqrtN-1)]/sqrtN;
[X,Y] = meshgrid(t);
x = [X(:), Y(:)];
[P,T,I] = MEXbivariateCompressor(x,2);
[P,T,I] = MEXbivariateCompressor(rand(10,1),2);
[P,T,I] = MEXbivariateCompressor(rand(10,4),2);
S = tril(sparse(P(:,1),P(:,2),P(:,3)));
clear P;
S = tril(S) + tril(S,-1)';
Q = sparse(T(:,1),T(:,2),T(:,3));
clear T;
invI = I;
invI(I) = [1:length(I)]'; 

for i = 1:N
    samplet = Q(i,invI);
    surf(X,Y,reshape(samplet,sqrtN,sqrtN));
    shading interp
    pause(1)
end
