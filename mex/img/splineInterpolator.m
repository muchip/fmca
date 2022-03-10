function K = splineInterpolator(N)
e = ones(N,1);
K = sparse(N+2,N+2);
K(2:N+1,2:N+1) = spdiags([B3(-1) * e, B3(0) * e, B3(1) * e], -1:1, N, N);
K(1,1:3) = [1 -2 1];
K(N+2,N:N+2) = [1 -2 1];
K(2,1) = K(2,3);
K(N+1,N+2) = K(N+1,N);

end